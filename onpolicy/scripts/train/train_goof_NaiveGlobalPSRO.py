#!/usr/bin/env python
import os
from pathlib import Path
import sys
import socket
import copy
import time

import numpy as np
import setproctitle
import torch
import wandb
import multiprocessing as mp

from onpolicy.config import get_config
from onpolicy.envs.goofspiel.goofspiel_gym import goofspiel_symmetry as Env
from onpolicy.envs.goofspiel.goofspiel import Goofspiel
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as empty_Policy
from onpolicy.algorithms.PSRO.train_NaiveGlobalPSRO import NaiveGlobal_PSRO_trainer
from onpolicy.envs.goofspiel.goofspiel_policy import Policy_goofspiel_random
from onpolicy.algorithms.PSRO.utils.eval_match import eval_match as EvalMatch
from onpolicy.runner.approx_PE_eval import ApproxPEEvaluator
from onpolicy.algorithms.PSRO.utils.meta_strategy_solver import zero_sum_2p_game as nash_MSS
from onpolicy.envs.goofspiel.policy_like_spiel import goof_policy_dict, policy_like_spiel, gen_mix_spiel_policy, calc_exp
from onpolicy.algorithms.PSRO.utils.meta_strategy_solver import alpha_rank_MSS, loading_MSS
from onpolicy.algorithms.PSRO.utils.meta_strategy_solver import PRD_MSS, Uniform_MSS

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
                        default="SPIEL", 
                        help="which scenario to run on.")
    parser.add_argument("--population_size", type=int, default=6)
    parser.add_argument("--eval_episode_num", type=int, default=20)
    parser.add_argument("--use_mix_policy", action='store_false', default=True)
    parser.add_argument("--upper_epsilon", type=float, default=0.1)
    parser.add_argument("--upper_std", type=float, default=0.03)
    parser.add_argument("--estimator_lr", type=float, default=0.00005)
    parser.add_argument("--use_calc_exploit", action='store_false', default=True)
    parser.add_argument("--calc_exp_interval", type=int, default=1)
    parser.add_argument("--use_approx_PE_eval", action="store_false", default=True)
    parser.add_argument("--approx_PE_steps", type=int, default=None)
    parser.add_argument("--approx_PE_eval_episodes", type=int, default=20)
    parser.add_argument("--approx_PE_std", type=float, default=0.03)
    parser.add_argument("--max_workers", type = int, default=6)
    parser.add_argument("--RM_interval", type=int, default=10)
    parser.add_argument("--RM_post_train_steps", type=int, default=0)
    parser.add_argument("--RM_pre_train_steps", type=int, default=0)
    parser.add_argument("--MSS_name", type=str, default="nash")
    parser.add_argument("--avg_G_last_N", type=int, default=10)
    parser.add_argument("--RM_yita_coef", type=float, default=1.0)

                        
    all_args = parser.parse_known_args(args)[0]
    if all_args.approx_PE_steps is None:
        all_args.approx_PE_steps = all_args.num_env_steps
    if all_args.approx_PE_eval_episodes is None:
        all_args.approx_PE_eval_episodes = all_args.eval_episode_num
    if all_args.approx_PE_std is None:
        all_args.approx_PE_std = all_args.upper_std

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
                         group="NaiveGlobalPSRO_goof_pp",
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
            policy_rule_p = Policy_goofspiel_random(policy_p1, device)
            policy_anchor.append(policy_rule_p)
            share_policies.append(policy_p1)
            policies_p1.append(policy_rule_p)
        else:
            policies_p1.append(copy.deepcopy(policy_p1))

    evaluator = EvalMatch(policies_p1, eval_match_envs)
    goof_policy_dict.set_standard_keys(eval_match_envs.world.standard_game)

    role_name = ["goofspiel"]
    save_dir = str(wandb.run.dir) if all_args.use_wandb else str(run_dir)
    approx_PE_evaluator = None
    if all_args.use_approx_PE_eval:
        approx_PE_evaluator = ApproxPEEvaluator(all_args, em_runner, role_name, save_dir, device=device)

    exploit_array = []
    approx_PE_array = []
    wall_time = 0
    approx_PE_time = 0
    meta_solver = choose_meta_solver(all_args)
    Meta_trainer = NaiveGlobal_PSRO_trainer(all_args, policy_anchor, share_policies, policies_p1, runner_p_list, em_runner_list, evaluator, meta_solver, role_name, save_dir, device=device)
    for i in range(all_args.population_size-1):
        start_time = time.time()
        re_policies, probs_now, logs = Meta_trainer.step()
        end_time = time.time()
        delta_time = end_time - start_time
        wall_time += delta_time
        if all_args.MSS_name == "anytime":
            Meta_trainer.meta_solver = loading_MSS([probs_now.copy(), probs_now.copy()])

        should_run_approx_PE = all_args.use_approx_PE_eval and i % all_args.calc_exp_interval == 0
        if should_run_approx_PE:
            approx_PE_evaluator.run(Meta_trainer.eval_policies, Meta_trainer.effect_population_size, Meta_trainer.g_step, Meta_trainer.n_threads)

        should_eval_exp = all_args.use_calc_exploit and i % all_args.calc_exp_interval == 0
        if should_eval_exp:
            game = copy.deepcopy(eval_match_envs.world.standard_game)
            policies_spiel = build_spiel_policies(game, re_policies)

            if should_eval_exp:
                print("calculate the exploitability!")
                print("Adopted probs = ", probs_now)
                exp, expl_per_player = calc_exact_exp_for_probs(game, policies_spiel, probs_now)
                print("exploit = ", exp)
                exploit_array.append(exp)
                np.save(os.path.join(save_dir, "exploit_" + str(role_name[0])) + ".npy", np.asarray(exploit_array))


            if all_args.use_wandb:
                log_dict = {"round": i + 2}
                if should_eval_exp:
                    log_dict["exploit"] = exp
                wandb.log(log_dict)

        if approx_PE_evaluator is not None:
            approx_PE_time = approx_PE_evaluator.approx_PE_time

        if should_run_approx_PE and approx_PE_evaluator.last_info is not None:
            approx_PE = float(np.asarray(approx_PE_evaluator.last_info["approx_PE"]).reshape(-1)[0])
            approx_PE_array.append(approx_PE)
            np.save(os.path.join(save_dir, "approx_PE_recorded_" + str(role_name[0]) + ".npy"), np.asarray(approx_PE_array))

        if all_args.use_wandb:
            log_dict = {"walltime": wall_time, "round": i+2}
            if approx_PE_evaluator is not None:
                log_dict["approx_PE_time"] = approx_PE_time
            if should_run_approx_PE and approx_PE_evaluator.last_info is not None:
                log_dict["approx_PE"] = approx_PE
                log_dict["approx_avg_PE"] = approx_PE_evaluator.last_info["approx_avg_PE"]
            wandb.log(log_dict)


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
