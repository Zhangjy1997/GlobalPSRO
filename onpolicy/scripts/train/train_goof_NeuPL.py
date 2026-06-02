#!/usr/bin/env python
import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import socket
import sys
import time

import numpy as np
import setproctitle
import torch
import wandb
from gym import spaces

from onpolicy.config import get_config
from onpolicy.runner.approx_PE_eval import ApproxPEEvaluator
from onpolicy.algorithms.PSRO.utils.eval_match import eval_match as EvalMatch
from onpolicy.algorithms.PSRO.utils.meta_strategy_solver import alpha_rank_MSS, loading_MSS
from onpolicy.algorithms.PSRO.utils.meta_strategy_solver import PRD_MSS, Uniform_MSS
from onpolicy.algorithms.PSRO.utils.meta_strategy_solver import zero_sum_2p_game as nash_MSS
from onpolicy.algorithms.PSRO.train_NeuPL import NeuPL_Trainer
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


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="SPIEL")


    parser.add_argument("--population_size", type=int, default=7)
    parser.add_argument("--eval_episode_num", type=int, default=20)
    parser.add_argument("--use_mix_policy", action="store_false", default=True)
    parser.add_argument("--upper_epsilon", type=float, default=0.0)
    parser.add_argument("--upper_std", type=float, default=0.03)
    parser.add_argument("--estimator_lr", type=float, default=0.00005)
    parser.add_argument("--use_real_estimator", action="store_false", default=True)
    parser.add_argument("--MSS_name", type=str, default="nash")
    parser.add_argument("--simplex_eps", type=float, default=1.0)
    parser.add_argument("--use_uniform_simplex", action="store_true", default=False)
    parser.add_argument("--use_policy_freeze", action="store_true", default=False)
    parser.add_argument("--use_best_model_history", action="store_true", default=False)
    parser.add_argument("--use_soft_kl", action="store_false", default=True)

    parser.add_argument("--RM_interval", type=int, default=10)
    parser.add_argument("--RM_post_train_steps", type=int, default=0)
    parser.add_argument("--RM_pre_train_steps", type=int, default=0)
    parser.add_argument("--avg_G_last_N", type=int, default=10)
    parser.add_argument("--RM_yita_coef", type=float, default=1.0)

    parser.add_argument("--use_calc_exploit", action="store_false", default=True)
    parser.add_argument("--calc_exp_interval", type=int, default=2)
    parser.add_argument("--use_approx_PE_eval", action="store_false", default=True)
    parser.add_argument("--approx_PE_steps", type=int, default=None)
    parser.add_argument("--approx_PE_eval_episodes", type=int, default=20)
    parser.add_argument("--approx_PE_std", type=float, default=0.03)
    parser.add_argument("--use_calc_approx_PE_exp", action="store_false", default=True)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--exp_max_workers", type=int, default=2)
    parser.add_argument("--exp_max_pending", type=int, default=64)
    parser.add_argument("--exp_worker_threads", type=int, default=1)

    all_args = parser.parse_known_args(args)[0]
    all_args.latent_size = all_args.population_size
    if all_args.approx_PE_steps is None:
        all_args.approx_PE_steps = all_args.num_env_steps
    if all_args.approx_PE_eval_episodes is None:
        all_args.approx_PE_eval_episodes = all_args.eval_episode_num
    if all_args.approx_PE_std is None:
        all_args.approx_PE_std = all_args.upper_std
    if all_args.use_calc_approx_PE_exp:
        all_args.use_approx_PE_eval = True
    return all_args


def choose_meta_solver(all_args):
    if all_args.MSS_name == "nash":
        return nash_MSS
    if all_args.MSS_name == "alpharank":
        return alpha_rank_MSS
    if all_args.MSS_name == "anytime":
        return loading_MSS()
    if all_args.MSS_name == "uniform":
        return Uniform_MSS
    if all_args.MSS_name == "PRD":
        return PRD_MSS
    raise NotImplementedError


def transfer2spielpolicy(game, policy_network, gpu_id, random_policy, policy_id, keys, use_cuda):
    goof_policy_dict.set_slots(keys)
    if use_cuda:
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        device = torch.device("cpu")
    if hasattr(policy_network, "transfer_model_to"):
        policy_network.transfer_model_to(device)
    policy = policy_like_spiel(copy.deepcopy(game), range(2), policy_network, random_policy=random_policy)
    serialized_result = pickle.dumps({policy_id: policy})
    return serialized_result


def parallel_trans(game, network_policies, random_policies, max_threads=4):
    if len(network_policies) == 0:
        return []
    if max_threads <= 1:
        goof_policy_dict.set_slots(list(goof_policy_dict.__slots__))
        return [
            policy_like_spiel(copy.deepcopy(game), range(2), policy, random_policy=random_policies[idx])
            for idx, policy in enumerate(network_policies)
        ]

    num_gpus = torch.cuda.device_count()
    use_cuda = torch.cuda.is_available() and num_gpus > 0
    num_devices = max(1, num_gpus)
    standard_keys = list(goof_policy_dict.__slots__)
    spiel_policies = []
    ctx = mp.get_context("spawn")

    for start_idx in range(0, len(network_policies), max_threads):
        policy_buffer = network_policies[start_idx : start_idx + max_threads]
        random_buffer = random_policies[start_idx : start_idx + max_threads]
        with ctx.Pool(processes=len(policy_buffer)) as pool:
            results = [
                pool.apply_async(
                    transfer2spielpolicy,
                    args=(
                        game,
                        policy_buffer[local_idx],
                        local_idx % num_devices,
                        random_buffer[local_idx],
                        local_idx,
                        standard_keys,
                        use_cuda,
                    ),
                )
                for local_idx in range(len(policy_buffer))
            ]
            pool.close()
            pool.join()

        buffer_dict = {}
        for result in results:
            buffer_dict.update(pickle.loads(result.get()))
        for local_idx in range(len(policy_buffer)):
            spiel_policies.append(buffer_dict[local_idx])
        print(
            "transfer to spiel policy: {}/{}".format(
                min(start_idx + len(policy_buffer), len(network_policies)),
                len(network_policies),
            )
        )

    return spiel_policies


def build_spiel_policies_with_cache(game, neural_policies, frozen_cache, frozen_count, max_workers):
    frozen_count = min(frozen_count, len(neural_policies))
    if len(frozen_cache) > frozen_count:
        frozen_cache = frozen_cache[:frozen_count]

    if len(frozen_cache) < frozen_count:
        start_idx = len(frozen_cache)
        new_frozen_policies = neural_policies[start_idx:frozen_count]
        random_masks = np.array([policy_idx == 0 for policy_idx in range(start_idx, frozen_count)], dtype=bool)
        print("convert newly frozen policies: {} -> {}".format(start_idx, frozen_count))
        frozen_cache.extend(
            parallel_trans(
                game,
                new_frozen_policies,
                random_masks,
                max_threads=max_workers,
            )
        )

    active_start = len(frozen_cache)
    active_policies = neural_policies[active_start:]
    active_masks = np.zeros(len(active_policies), dtype=bool)
    if len(active_policies) > 0:
        print("convert active policies: {} -> {}".format(active_start, len(neural_policies)))
    active_spiel_policies = parallel_trans(game, active_policies, active_masks, max_threads=max_workers)
    return frozen_cache + active_spiel_policies, frozen_cache


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


def exp_job(exp_name, submit_round, game_snapshot, policies_snapshot, probs_snapshot, approx_PE=None):
    exp, _ = calc_exp_for_probs(game_snapshot, policies_snapshot, probs_snapshot)
    return {
        "name": exp_name,
        "round": submit_round,
        "exp": exp,
        "approx_PE": approx_PE,
    }


def save_exp_records(records, save_dir, filename):
    ordered_rounds = sorted(records)
    values = [records[round_id] for round_id in ordered_rounds]
    np.save(os.path.join(save_dir, filename), np.asarray(values))


def collect_finished_exp_jobs(
    exp_futures,
    meta_exp_records,
    approx_PE_exp_records,
    approx_PE_records,
    save_dir,
    role_name,
    use_wandb,
):
    still_pending = []
    for fut in exp_futures:
        if not fut.done():
            still_pending.append(fut)
            continue

        try:
            result = fut.result()
            exp_name = result["name"]
            exp_round = result["round"]
            exp_value = result["exp"]
            print("[{} finished] round={}, exp={}".format(exp_name, exp_round, exp_value))

            if exp_name == "meta_probs_exp":
                meta_exp_records[exp_round] = exp_value
                save_exp_records(meta_exp_records, save_dir, "meta_probs_exp_" + str(role_name[0]) + ".npy")
            elif exp_name == "approx_PE_probs_exp":
                approx_PE_exp_records[exp_round] = exp_value
                save_exp_records(approx_PE_exp_records, save_dir, "approx_PE_probs_exp_" + str(role_name[0]) + ".npy")
                if result["approx_PE"] is not None:
                    approx_PE_records[exp_round] = result["approx_PE"]
                    save_exp_records(approx_PE_records, save_dir, "approx_PE_recorded_" + str(role_name[0]) + ".npy")
            else:
                print("[exp worker skipped unknown result] {}".format(exp_name))

            if use_wandb:
                wandb.log({exp_name: exp_value, "round": exp_round})
        except Exception as exc:
            print("[exp worker failed] {}".format(exc))

    return still_pending


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

    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            notes=socket.gethostname(),
            name="-".join([all_args.algorithm_name, all_args.experiment_name, "seed" + str(all_args.seed)]),
            group="NeuPL_goof",
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

    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_obs_latent import R_MAPPOPolicy as empty_Policy

    from onpolicy.runner.br_simplex_runner import BR_Simplex_Runner as BR_Runner
    from onpolicy.runner.exp3_runner import EXP3_Runner as Estimator

    envs_p = make_train_env(all_args)
    eval_envs_p = make_eval_env(all_args) if all_args.use_eval else None
    eval_match_envs = make_train_env(all_args)
    standard_game = copy.deepcopy(envs_p.world.standard_game)
    goof_policy_dict.set_standard_keys(eval_match_envs.world.standard_game)

    all_args.episode_length = envs_p.episode_length
    config = {
        "all_args": all_args,
        "envs": envs_p,
        "eval_envs": eval_envs_p,
        "num_agents": 1,
        "device": device,
        "run_dir": run_dir,
    }
    runner_p = BR_Runner(config)
    save_dir = runner_p.save_dir

    orig_lr = all_args.lr
    orig_critic_lr = all_args.critic_lr
    all_args.lr = all_args.estimator_lr
    all_args.critic_lr = all_args.estimator_lr
    config_em = {
        "all_args": all_args,
        "envs": envs_p,
        "eval_envs": eval_envs_p,
        "num_agents": 1,
        "device": device,
        "run_dir": run_dir,
    }
    em_runner = Estimator(config_em)
    all_args.lr = orig_lr
    all_args.critic_lr = orig_critic_lr

    share_observation_space = envs_p.share_observation_space[0] if all_args.use_centralized_V else envs_p.observation_space[0]
    obs_dim = envs_p.observation_space[0].shape[-1]
    obs_fusion = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim + all_args.latent_size,), dtype=np.float32)
    cent_obs_dim = share_observation_space.shape[-1]
    cent_obs_fusion = spaces.Box(low=-1.0, high=1.0, shape=(cent_obs_dim + all_args.latent_size,), dtype=np.float32)

    policies_p1 = []
    share_policies = []
    policy_anchor = []
    for policy_idx in range(all_args.population_size):
        policy_p1 = empty_Policy(
            all_args,
            obs_fusion,
            cent_obs_fusion,
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

    evaluator = EvalMatch(policies_p1, eval_match_envs)
    role_name = ["goofspiel"]
    meta_solver = choose_meta_solver(all_args)
    approx_PE_evaluator = None
    if all_args.use_approx_PE_eval or all_args.use_calc_approx_PE_exp:
        approx_PE_evaluator = ApproxPEEvaluator(all_args, em_runner, role_name, save_dir, device=device)

    meta_trainer = NeuPL_Trainer(
        all_args,
        policy_anchor,
        share_policies,
        policies_p1,
        runner_p,
        evaluator,
        meta_solver,
        role_name,
        save_dir,
        device=device,
    )

    meta_exp_records = {}
    approx_PE_exp_records = {}
    approx_PE_records = {}
    frozen_spiel_cache = []
    wall_time = 0
    approx_PE_time = 0
    exp_executor = None
    exp_futures = []

    if all_args.use_calc_exploit or all_args.use_calc_approx_PE_exp:
        mp_ctx = mp.get_context("spawn")
        exp_executor = ProcessPoolExecutor(
            max_workers=all_args.exp_max_workers,
            mp_context=mp_ctx,
            initializer=exp_worker_init,
            initargs=(all_args.exp_worker_threads,),
        )

    for round_idx in range(2 * (all_args.population_size - 1)):
        if exp_executor is not None:
            exp_futures = collect_finished_exp_jobs(
                exp_futures,
                meta_exp_records,
                approx_PE_exp_records,
                approx_PE_records,
                save_dir,
                role_name,
                all_args.use_wandb,
            )

        start_t = time.time()
        re_policies, probs_now, logs = meta_trainer.step()
        wall_time += time.time() - start_t

        if all_args.MSS_name == "anytime":
            meta_trainer.meta_solver = loading_MSS([probs_now.copy(), probs_now.copy()])

        should_run_approx_PE = all_args.use_approx_PE_eval and round_idx % all_args.calc_exp_interval == 0
        if should_run_approx_PE:
            approx_PE_evaluator.run(re_policies, len(probs_now), meta_trainer.g_step, meta_trainer.n_threads)

        should_eval_exp = all_args.use_calc_exploit and round_idx % all_args.calc_exp_interval == 0
        should_eval_approx_PE_exp = all_args.use_calc_approx_PE_exp and should_run_approx_PE
        if should_eval_exp or should_eval_approx_PE_exp:
            game = copy.deepcopy(eval_match_envs.world.standard_game)
            frozen_count = getattr(meta_trainer, "lower_i", 0) + 1
            policies_spiel, frozen_spiel_cache = build_spiel_policies_with_cache(
                game,
                re_policies,
                frozen_spiel_cache,
                frozen_count,
                all_args.max_workers,
            )
            unfinished = sum(0 if fut.done() else 1 for fut in exp_futures)

            if should_eval_exp:
                meta_probs = meta_trainer.get_sub_meta_probs()
                print("submit exact exploitability of NeuPL meta probs!")
                print("meta probs = ", meta_probs)
                if unfinished < all_args.exp_max_pending:
                    fut = exp_executor.submit(
                        exp_job,
                        "meta_probs_exp",
                        round_idx + 2,
                        copy.deepcopy(game),
                        list(policies_spiel),
                        meta_probs.copy(),
                    )
                    exp_futures.append(fut)
                    unfinished += 1
                    print("[meta exp submitted] round={}, unfinished={}".format(round_idx + 2, unfinished))
                else:
                    print("[meta exp skipped] round={}, unfinished={}, limit={}".format(round_idx + 2, unfinished, all_args.exp_max_pending))

            if should_eval_approx_PE_exp:
                support = approx_PE_evaluator.last_policy_support
                if support is not None:
                    approx_probs = support["probs"]
                    approx_info = approx_PE_evaluator.last_info
                    approx_PE_value = None
                    if approx_info is not None:
                        approx_PE_value = float(np.asarray(approx_info["approx_PE"]).reshape(-1)[0])
                    print("submit exact exploitability of approx PE probsE!")
                    if unfinished < all_args.exp_max_pending:
                        fut = exp_executor.submit(
                            exp_job,
                            "approx_PE_probs_exp",
                            round_idx + 2,
                            copy.deepcopy(game),
                            list(policies_spiel),
                            approx_probs.copy(),
                            approx_PE_value,
                        )
                        exp_futures.append(fut)
                        unfinished += 1
                        print("[approx PE exp submitted] round={}, unfinished={}".format(round_idx + 2, unfinished))
                    else:
                        print("[approx PE exp skipped] round={}, unfinished={}, limit={}".format(round_idx + 2, unfinished, all_args.exp_max_pending))
                else:
                    print("skip approx PE probsE exp: no approx PE support is available")

            if all_args.use_wandb:
                wandb.log({"round": round_idx + 2, "exp_pending": unfinished})

        if approx_PE_evaluator is not None:
            approx_PE_time = approx_PE_evaluator.approx_PE_time

        if all_args.use_wandb:
            wandb.log(
                {
                    "walltime": wall_time,
                    "approx_PE_time": approx_PE_time,
                    "round": round_idx + 2,
                }
            )

    if exp_executor is not None:
        for fut in exp_futures:
            try:
                fut.result()
            except Exception:
                pass
        exp_futures = collect_finished_exp_jobs(
            exp_futures,
            meta_exp_records,
            approx_PE_exp_records,
            approx_PE_records,
            save_dir,
            role_name,
            all_args.use_wandb,
        )
        exp_executor.shutdown(wait=True)

    envs_p.close()
    if all_args.use_eval and eval_envs_p is not envs_p:
        eval_envs_p.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner_p.writter.export_scalars_to_json(str(runner_p.log_dir + "/summary.json"))
        runner_p.writter.close()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(sys.argv[1:])
