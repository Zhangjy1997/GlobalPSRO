#!/usr/bin/env python
import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import setproctitle
import torch
import wandb

from onpolicy.config import get_config
from onpolicy.algorithms.PSRO.train_APSRO import Anytime_PSRO_trainer
from onpolicy.envs.goofspiel.goofspiel import Goofspiel
from onpolicy.envs.goofspiel.goofspiel_gym import goofspiel_symmetry as Env
from onpolicy.envs.goofspiel.goofspiel_policy import Policy_goofspiel_random
from onpolicy.envs.goofspiel.policy_like_spiel import (
    calc_exp,
    gen_mix_spiel_policy,
    goof_policy_dict,
    policy_like_spiel,
)


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "GOOFSPIEL":
                env = Goofspiel(Env(all_args.n_rollout_threads))
            else:
                print("Unsupported environment: " + all_args.env_name)
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env()

    return get_env_fn(0)


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "GOOFSPIEL":
                env = Goofspiel(Env(all_args.n_rollout_threads))
            else:
                print("Unsupported environment: " + all_args.env_name)
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env()

    return get_env_fn(0)


def build_spiel_policies(game, neural_policies, policies_spiel=None):
    if policies_spiel is None:
        policies_spiel = []
    if len(policies_spiel) > len(neural_policies):
        policies_spiel = policies_spiel[: len(neural_policies)]

    start_idx = len(policies_spiel)
    for policy_idx, policy in enumerate(neural_policies[start_idx:], start=start_idx):
        print("convert policy {} to spiel policy".format(policy_idx))
        policies_spiel.append(
            policy_like_spiel(
                copy.deepcopy(game),
                range(2),
                policy,
                random_policy=(policy_idx == 0),
            )
        )
    return policies_spiel


def calc_exp_for_probs(game, policies_spiel, probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs[: len(policies_spiel)]
    support_mask = probs > 1e-5
    if not np.any(support_mask):
        support_mask[0] = True
        probs = np.ones_like(probs) / len(probs)

    policies_support = np.array(policies_spiel, dtype=object)[support_mask].tolist()
    probs_support = probs[support_mask]
    probs_support = probs_support / np.sum(probs_support)
    final_policies = gen_mix_spiel_policy(
        copy.deepcopy(game),
        range(2),
        [policies_support, copy.deepcopy(policies_support)],
        [probs_support, probs_support.copy()],
    )
    exp, expl_per_player = calc_exp(copy.deepcopy(game), final_policies)
    return exp, expl_per_player


def exp_worker_init(num_threads):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)


def exp_job(submit_round, game_snapshot, policies_snapshot, probs_snapshot):
    exp, _ = calc_exp_for_probs(game_snapshot, policies_snapshot, probs_snapshot)
    return submit_round, exp


def save_exp_records(records, save_dir, role_name):
    ordered_rounds = sorted(records)
    values = [records[round_id] for round_id in ordered_rounds]
    np.save(os.path.join(save_dir, "meta_probs_exp_" + str(role_name[0]) + ".npy"), np.asarray(values))


def collect_finished_exp_jobs(exp_futures, exploit_records, save_dir, role_name, use_wandb):
    still_pending = []
    for fut in exp_futures:
        if not fut.done():
            still_pending.append(fut)
            continue

        try:
            exp_round, exp_value = fut.result()
            print("[exploit finished] round={}, exp={}".format(exp_round, exp_value))
            exploit_records[exp_round] = exp_value
            save_exp_records(exploit_records, save_dir, role_name)
            if use_wandb:
                wandb.log({"meta_probs_exp": exp_value, "round": exp_round})
        except Exception as exc:
            print("[exploit worker failed] {}".format(exc))

    return still_pending


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="SPIEL")
    parser.add_argument("--population_size", type=int, default=6)
    parser.add_argument("--eval_episode_num", type=int, default=20)
    parser.add_argument("--use_mix_policy", action="store_false", default=True)
    parser.add_argument("--use_calc_exploit", action="store_false", default=True)
    parser.add_argument("--calc_exp_interval", type=int, default=3)
    parser.add_argument("--exp_max_workers", type=int, default=2)
    parser.add_argument("--exp_max_pending", type=int, default=64)
    parser.add_argument("--exp_worker_threads", type=int, default=1)
    parser.add_argument("--RM_interval", type=int, default=10)
    parser.add_argument("--RM_post_train_steps", type=int, default=0)
    parser.add_argument("--RM_pre_train_steps", type=int, default=0)
    parser.add_argument("--avg_G_last_N", type=int, default=10)
    parser.add_argument("--RM_yita_coef", type=float, default=1.0)
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    parser.set_defaults(
        algorithm_name="mappo",
        n_rollout_threads=200,
        ppo_epoch=15,
        num_mini_batch=2,
        save_interval=10,
        log_interval=1,
        use_proper_time_limits=True,
        gamma=1.0,
        entropy_coef=0.04,
        layer_N=3,
        hidden_size=128,
        recurrent_N=1,
        critic_lr=0.00005,
        lr=0.00005,
    )
    all_args = parse_args(args, parser)
    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda")
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        print("Number of available GPUs = {}".format(torch.cuda.device_count()))
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            notes=socket.gethostname(),
            name="-".join([all_args.algorithm_name, all_args.experiment_name, "seed" + str(all_args.seed)]),
            group="APSRO_goof",
            job_type="training",
            reinit=False,
        )
        save_dir = str(wandb.run.dir)
    else:
        existing_runs = [int(str(folder.name).split("run")[1]) for folder in run_dir.iterdir() if str(folder.name).startswith("run")]
        curr_run = "run1" if len(existing_runs) == 0 else "run%i" % (max(existing_runs) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        save_dir = str(run_dir)

    setproctitle.setproctitle(
        "-".join([all_args.env_name, all_args.scenario_name, all_args.algorithm_name, all_args.experiment_name])
        + "@"
        + all_args.user_name
    )

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as empty_Policy
    from onpolicy.runner.exp3_runner import EXP3_Runner as APSRO_Runner

    envs_p = make_train_env(all_args)
    eval_envs_p = make_eval_env(all_args) if all_args.use_eval else None
    eval_match_envs = make_train_env(all_args)
    all_args.episode_length = envs_p.episode_length

    config = {
        "all_args": all_args,
        "envs": envs_p,
        "eval_envs": eval_envs_p,
        "num_agents": 1,
        "device": device,
        "run_dir": run_dir,
    }
    runner_p = APSRO_Runner(config)
    save_dir = runner_p.save_dir

    policies_p1 = []
    share_policies = []
    policy_anchor = []
    for policy_idx in range(all_args.population_size):
        policy_p1 = empty_Policy(
            all_args,
            envs_p.observation_space[0],
            envs_p.observation_space[0],
            envs_p.action_space[0],
            device,
        )
        if policy_idx == 0:
            policy_rule_p = Policy_goofspiel_random(policy_p1, device)
            policy_anchor.append(policy_rule_p)
            share_policies.append(policy_p1)
            policies_p1.append(policy_rule_p)
        else:
            policies_p1.append(copy.deepcopy(policy_p1))

    goof_policy_dict.set_standard_keys(eval_match_envs.world.standard_game)
    role_name = ["goofspiel"]
    trainer = Anytime_PSRO_trainer(
        all_args,
        policy_anchor,
        share_policies,
        policies_p1,
        runner_p,
        role_name,
        save_dir,
        device=device,
    )

    meta_exp_records = {}
    spiel_policy_cache = []
    wall_time = 0
    exp_executor = None
    exp_futures = []
    if all_args.use_calc_exploit:
        mp_ctx = mp.get_context("spawn")
        exp_executor = ProcessPoolExecutor(
            max_workers=all_args.exp_max_workers,
            mp_context=mp_ctx,
            initializer=exp_worker_init,
            initargs=(all_args.exp_worker_threads,),
        )

    for round_idx in range(all_args.population_size - 1):
        if exp_executor is not None:
            exp_futures = collect_finished_exp_jobs(
                exp_futures,
                meta_exp_records,
                save_dir,
                role_name,
                all_args.use_wandb,
            )

        start_t = time.time()
        re_policies, probs_now, logs = trainer.step()
        wall_time += time.time() - start_t

        should_eval_exp = all_args.use_calc_exploit and round_idx % all_args.calc_exp_interval == 0
        if should_eval_exp:
            print("submit exact exploitability of APSRO probs!")
            game = copy.deepcopy(eval_match_envs.world.standard_game)
            print("APSRO probs = ", probs_now)
            spiel_policy_cache = build_spiel_policies(game, re_policies, spiel_policy_cache)
            unfinished = sum(0 if fut.done() else 1 for fut in exp_futures)
            if unfinished < all_args.exp_max_pending:
                fut = exp_executor.submit(
                    exp_job,
                    round_idx + 2,
                    copy.deepcopy(game),
                    list(spiel_policy_cache),
                    probs_now.copy(),
                )
                exp_futures.append(fut)
                unfinished += 1
                print("[exploit submitted] round={}, unfinished={}".format(round_idx + 2, unfinished))
            else:
                print(
                    "[exploit skipped] round={}, unfinished={}, limit={}".format(
                        round_idx + 2,
                        unfinished,
                        all_args.exp_max_pending,
                    )
                )
            if all_args.use_wandb:
                wandb.log({"round": round_idx + 2, "exp_pending": unfinished})

        if all_args.use_wandb:
            wandb.log({"walltime": wall_time, "round": round_idx + 2})

    if exp_executor is not None:
        for fut in exp_futures:
            try:
                fut.result()
            except Exception:
                pass
        exp_futures = collect_finished_exp_jobs(
            exp_futures,
            meta_exp_records,
            save_dir,
            role_name,
            all_args.use_wandb,
        )
        exp_executor.shutdown(wait=True)

    envs_p.close()
    if eval_match_envs is not envs_p:
        eval_match_envs.close()
    if all_args.use_eval and eval_envs_p is not envs_p:
        eval_envs_p.close()
    if all_args.use_wandb:
        run.finish()
    else:
        runner_p.writter.export_scalars_to_json(str(runner_p.log_dir + "/summary.json"))
        runner_p.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
