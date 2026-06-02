#!/usr/bin/env python
import os
from pathlib import Path
import sys
import socket
import copy
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import setproctitle
import torch
import wandb
from scipy.optimize import minimize
import multiprocessing as mp

from onpolicy.config import get_config
from onpolicy.envs.poker.leduc_poker_gym import leduc_poker_symmetry as Env_L
from onpolicy.envs.poker.leduc_poker import Poker
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as empty_Policy
from onpolicy.algorithms.PSRO.train_NaiveGlobalPSRO import NaiveGlobal_PSRO_trainer
from onpolicy.envs.poker.leduc_poker_policy import Policy_poker_random
from onpolicy.algorithms.PSRO.utils.eval_match import eval_match as EvalMatch
from onpolicy.algorithms.PSRO.utils.meta_strategy_solver import zero_sum_2p_game as nash_MSS
from onpolicy.envs.poker.policy_like_spiel import policy_like_spiel, gen_mix_spiel_policy, calc_exp
from onpolicy.algorithms.PSRO.utils.meta_strategy_solver import alpha_rank_MSS, loading_MSS
from onpolicy.algorithms.PSRO.utils.meta_strategy_solver import PRD_MSS, Uniform_MSS

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "LEDUC_POKER":
                env = Poker(Env_L(all_args.n_rollout_threads))
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
            if all_args.env_name == "LEDUC_POKER":
                env = Poker(Env_L(all_args.n_rollout_threads))
            else:
                print("Unsupported environment: " + all_args.env_name)
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env()
    return get_env_fn(0)

def make_min_pure_function(game, policy_set):
    def exploit_p(x):
        assert len(policy_set) == len(x), "dimension mismatch!"
        final_policies = gen_mix_spiel_policy(copy.deepcopy(game), range(2), [policy_set, policy_set], [x, x.copy()])
        exp, _ = calc_exp(copy.deepcopy(game), final_policies)
        return exp
    return exploit_p

def find_min_exp(standard_game, policy_set):
    def eq_constraint(x):
        return np.sum(x) - 1
    
    def ineq_constraints(x):
        return x

    constraints = [
        {'type': 'eq', 'fun': eq_constraint},
        {'type': 'ineq', 'fun': ineq_constraints}
    ]

    options = {
        'disp': False,
        'ftol': 1e-5
    }
    target_f = make_min_pure_function(standard_game, policy_set)
    x0 = np.ones(len(policy_set))
    x0 = x0/np.sum(x0)
    result = minimize(target_f, x0, method='SLSQP', constraints=constraints, options=options)

    min_exp = result.fun
    min_probs = result.x

    return min_probs, min_exp

def pe_worker_init(num_threads):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)


def pe_job(submit_round, game_snapshot, policies_snapshot):
    exact_PE_probs, exact_PE = find_min_exp(game_snapshot, policies_snapshot)
    return {
        "round": submit_round,
        "exact_PE": exact_PE,
        "exact_PE_probs": exact_PE_probs,
    }


def collect_finished_pe_jobs(pe_futures, exact_PE_array, exact_PE_probs_array, save_dir, role_name, use_wandb):
    still_pending = []
    for fut in pe_futures:
        if not fut.done():
            still_pending.append(fut)
            continue

        try:
            result = fut.result()
            pe_round = result["round"]
            exact_PE = result["exact_PE"]
            exact_PE_probs = result["exact_PE_probs"]

            print(f"[exact PE finished] round={pe_round}, PE={exact_PE}")
            print("exact PE probs = ", exact_PE_probs)
            exact_PE_array.append(exact_PE)
            exact_PE_probs_array.append(exact_PE_probs.copy())
            np.save(os.path.join(save_dir, "exact_PE_" + str(role_name[0]) + ".npy"), np.asarray(exact_PE_array))
            np.save(os.path.join(save_dir, "PE" + str(role_name[0]) + ".npy"), np.asarray(exact_PE_array))
            np.save(
                os.path.join(save_dir, "exact_PE_probs_" + str(role_name[0]) + ".npy"),
                np.asarray(exact_PE_probs_array, dtype=object),
                allow_pickle=True,
            )

            if use_wandb:
                wandb.log({"exact_PE": exact_PE, "PE": exact_PE, "round": pe_round})
        except Exception as exc:
            print(f"[exact PE worker failed] {exc}")

    return still_pending


def build_spiel_policies(game, neural_policies):
    policies_spiel = []
    for policy_idx, policy in enumerate(neural_policies):
        policies_spiel.append(
            policy_like_spiel(
                copy.deepcopy(game),
                range(2),
                policy,
                random_policy=(policy_idx == 0),
            )
        )
    return policies_spiel


def calc_exact_exp_for_probs(game, policies_spiel, probs):
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


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default="POKER", 
                        help="which scenario to run on.")

    


    parser.add_argument("--population_size", type=int, default=6)
    parser.add_argument("--eval_episode_num", type=int, default=20)
    parser.add_argument("--use_mix_policy", action='store_false', default=True)
    parser.add_argument("--upper_epsilon", type=float, default=0.1)
    parser.add_argument("--upper_std", type=float, default=0.03)
    parser.add_argument("--estimator_lr", type=float, default=0.00005)
    parser.add_argument("--use_calc_exploit", action='store_false', default=True)
    parser.add_argument("--calc_exp_interval", type=int, default=1)
    parser.add_argument("--use_calc_PE", action='store_false', default=True)
    parser.add_argument("--PE_interval", type=int, default=1)
    parser.add_argument("--pe_max_workers", type=int, default=4)
    parser.add_argument("--pe_max_pending", type=int, default=64)
    parser.add_argument("--pe_worker_threads", type=int, default=1)
    parser.add_argument("--max_workers", type = int, default=6)
    parser.add_argument("--RM_interval", type=int, default=10)
    parser.add_argument("--RM_post_train_steps", type=int, default=0)
    parser.add_argument("--RM_pre_train_steps", type=int, default=0)
    parser.add_argument("--MSS_name", type=str, default="nash")
    parser.add_argument("--avg_G_last_N", type=int, default=10)
    parser.add_argument("--RM_yita_coef", type=float, default=1.0)

                        
    all_args = parser.parse_known_args(args)[0]

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

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available")
        else:
            print("Number of available GPUs = {}".format(num_gpus))
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         notes=socket.gethostname(),
                         name="-".join([
                            all_args.algorithm_name,
                            all_args.experiment_name,
                            "seed" + str(all_args.seed)
                         ]),
                         group="NaiveGlobalPSRO_poker_pp",
                         job_type="training",
                         reinit=False)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle("-".join([
        all_args.env_name, 
        all_args.scenario_name, 
        all_args.algorithm_name, 
        all_args.experiment_name
    ]) + "@" + all_args.user_name)

            
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    from onpolicy.runner.br_runner import BR_Runner

    from onpolicy.runner.exp3_runner import EXP3_Runner as Estimator

    

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
        "run_dir": run_dir
    }
    
    runner_p_list = []
    
    for i in range(all_args.max_workers):
        runner_p = BR_Runner(config)
        runner_p_list.append(copy.deepcopy(runner_p))

    all_args.critic_lr = all_args.estimator_lr
    all_args.lr = all_args.estimator_lr

    config_em = {
        "all_args": all_args,
        "envs": envs_p,
        "eval_envs": eval_envs_p,
        "num_agents": 1,
        "device": device,
        "run_dir": run_dir
    }
    
    em_runner = Estimator(config_em)
    runner_p_list[0] = copy.deepcopy(em_runner)

    em_runner_list = []

    for i in range(all_args.max_workers):
        em_runner = Estimator(config_em)
        em_runner_list.append(copy.deepcopy(em_runner))

    policies_p1 = []
    share_policies = []
    policy_anchor = []

    for i in range(all_args.population_size):
        policy_p1 = empty_Policy(all_args,
                            envs_p.observation_space[0],
                            envs_p.observation_space[0],
                            envs_p.action_space[0],
                            device)
        if i == 0:
            policy_rule_p = Policy_poker_random(policy_p1, device)
            policy_anchor.append(policy_rule_p)
            share_policies.append(policy_p1)
            policies_p1.append(policy_rule_p)
        else:
            policies_p1.append(copy.deepcopy(policy_p1))

    evaluator = EvalMatch(policies_p1, eval_match_envs)

    
    role_name = ["poker"]
    exploit_array = []
    exact_PE_array = []
    exact_PE_probs_array = []
    wall_time = 0
    pe_executor = None
    pe_futures = []
    save_dir = str(wandb.run.dir) if all_args.use_wandb else str(run_dir)
    if all_args.use_calc_PE:
        mp_ctx = mp.get_context("spawn")
        pe_executor = ProcessPoolExecutor(
            max_workers=all_args.pe_max_workers,
            mp_context=mp_ctx,
            initializer=pe_worker_init,
            initargs=(all_args.pe_worker_threads,),
        )
    meta_solver = choose_meta_solver(all_args)
    Meta_trainer = NaiveGlobal_PSRO_trainer(all_args, policy_anchor, share_policies, policies_p1, runner_p_list, em_runner_list, evaluator, meta_solver, role_name, save_dir, device=device)
    for i in range(all_args.population_size-1):
        if all_args.use_calc_PE:
            pe_futures = collect_finished_pe_jobs(
                pe_futures,
                exact_PE_array,
                exact_PE_probs_array,
                save_dir,
                role_name,
                all_args.use_wandb,
            )

        start_time = time.time()
        re_policies, probs_now, logs = Meta_trainer.step()
        end_time = time.time()
        delta_time = end_time - start_time
        wall_time += delta_time
        if all_args.MSS_name == "anytime":
            Meta_trainer.meta_solver = loading_MSS([probs_now.copy(), probs_now.copy()])

        should_eval_exp = all_args.use_calc_exploit and i % all_args.calc_exp_interval == 0
        should_eval_pe = all_args.use_calc_PE and i % all_args.PE_interval == 0
        if should_eval_exp or should_eval_pe:
            game = copy.deepcopy(eval_match_envs.world.standard_game)
            policies_spiel = build_spiel_policies(game, re_policies)

            if should_eval_exp:
                print("calculate the exploitability!")
                print("Adopted probs = ", probs_now)
                exp, expl_per_player = calc_exact_exp_for_probs(game, policies_spiel, probs_now)
                print("exploit = ", exp)
                exploit_array.append(exp)
                np.save(os.path.join(save_dir, "exploit_" + str(role_name[0])) + ".npy", np.asarray(exploit_array))

            if should_eval_pe:
                unfinished = sum(0 if fut.done() else 1 for fut in pe_futures)
                if unfinished < all_args.pe_max_pending:
                    fut = pe_executor.submit(
                        pe_job,
                        i + 2,
                        copy.deepcopy(game),
                        copy.deepcopy(policies_spiel),
                    )
                    pe_futures.append(fut)
                    print(f"[exact PE submitted] round={i + 2}, unfinished={unfinished + 1}")
                else:
                    print(
                        f"[exact PE skipped] round={i + 2}, unfinished={unfinished}, "
                        f"limit={all_args.pe_max_pending}"
                    )

            if all_args.use_wandb:
                log_dict = {"round": i + 2}
                if should_eval_exp:
                    log_dict["exploit"] = exp
                if should_eval_pe:
                    log_dict["exact_PE_pending"] = sum(0 if fut.done() else 1 for fut in pe_futures)
                wandb.log(log_dict)

        if all_args.use_wandb:
            wandb.log({"walltime": wall_time, "round": i+2})

    if all_args.use_calc_PE and pe_executor is not None:
        for fut in pe_futures:
            try:
                result = fut.result()
                pe_round = result["round"]
                exact_PE = result["exact_PE"]
                exact_PE_probs = result["exact_PE_probs"]

                print(f"[exact PE finished at end] round={pe_round}, PE={exact_PE}")
                print("exact PE probs = ", exact_PE_probs)
                exact_PE_array.append(exact_PE)
                exact_PE_probs_array.append(exact_PE_probs.copy())
                np.save(os.path.join(save_dir, "exact_PE_" + str(role_name[0]) + ".npy"), np.asarray(exact_PE_array))
                np.save(os.path.join(save_dir, "PE" + str(role_name[0]) + ".npy"), np.asarray(exact_PE_array))
                np.save(
                    os.path.join(save_dir, "exact_PE_probs_" + str(role_name[0]) + ".npy"),
                    np.asarray(exact_PE_probs_array, dtype=object),
                    allow_pickle=True,
                )

                if all_args.use_wandb:
                    wandb.log({"exact_PE": exact_PE, "PE": exact_PE, "round": pe_round})
            except Exception as exc:
                print(f"[exact PE worker failed at end] {exc}")

        pe_executor.shutdown(wait=True)

    envs_p.close()
    if all_args.use_eval and eval_envs_p is not envs_p:
        eval_envs_p.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner_p.writter.export_scalars_to_json(str(runner_p.log_dir + '/summary.json'))
        runner_p.writter.close()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main(sys.argv[1:])
