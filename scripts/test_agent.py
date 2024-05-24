#This file is a part of COL864 A2
import os
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


import gym
import numpy as np
import torch


import config as exp_config
import utils.utils as utils
import utils.pytorch_util as ptu
from utils.logger import Logger

MAX_NVIDEO = 2

def setup_agent(args, configs, load_checkpoint):
    print("loading checkpoint", load_checkpoint)
    global env, agent
    
    env = gym.make(args.env_name,render_mode=None)
    env.action_space.seed()
    env.reset()
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    if args.exp_name == "imitation":
        from agents.mujoco_agents import ImitationAgent as Agent    
    elif args.exp_name == "RL":
        from agents.mujoco_agents import RLAgent as Agent
    elif args.exp_name == "imitation-RL":
        from agents.mujoco_agents import ImitationSeededRL as Agent
    else:
        raise ValueError(f"Invalid experiment name {args.exp_name}")

    agent = Agent(ob_dim, ac_dim, args, **configs['hyperparameters'])
    if load_checkpoint is not None:
        state_dict = torch.load(load_checkpoint)
        state_dict = {key: value for key, value in state_dict.items() if not ("expert_policy") in key}
        agent.load_state_dict(state_dict)


def test_agent(args):
    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    logger = Logger(args.logdir)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    total_envsteps = 0
    start_time = time.time()

    agent.to(ptu.device)

    eval_trajs, _ = utils.sample_trajectories(env, agent.get_action, 50000, 1000)
    logs = utils.compute_eval_metrics(eval_trajs)

    for key, value in logs.items():
        print("{} : {}".format(key, value))
        logger.log_scalar(value, key, 0)
    print("Done logging eval...\n\n")

    print("\nCollecting video rollouts...")
    eval_video_trajs = utils.sample_n_trajectories(
        env, agent.get_action, MAX_NVIDEO, 1000, render=True
    )

    logger.log_trajs_as_videos(
        eval_video_trajs,
        1,
        fps=fps,
        max_videos_to_save=MAX_NVIDEO,
        video_title="eval_rollouts",
    )



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, choices = ["imitation", "RL", "imitation-RL"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    # parser.add_argument("--logdir_name", type=str, default="data")

    args = parser.parse_args()

    configs = exp_config.configs[args.env_name][args.exp_name]

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data_test")
    model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models")
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    # logdir = (
    #     args.exp_name
    #     + "_"
    #     + args.env_name
    #     + "_"
    #     + time.strftime("%d-%m-%Y_%H-%M-%S")
    # )
    logdir = data_path
    args.logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("\n\n =====================================================")
    print("Testing agent..........")
    # setup_agent(args, configs, f"best_models/model_{args.exp_name}_{args.env_name}.pth")
    if os.path.exists(f"./best_models/model_{args.exp_name}_{args.env_name}.pth"):
        setup_agent(args, configs, f"best_models/model_{args.exp_name}_{args.env_name}.pth")
    elif os.path.exists(f"./best_models/model_{args.env_name}_{args.exp_name}.pth"):
        setup_agent(args, configs, f"./best_models/model_{args.env_name}_{args.exp_name}.pth")
    else:
        setup_agent(args, configs, f"../data_train/model_{args.env_name}_{args.exp_name}.pth")

    test_agent(args)
    

if __name__ == "__main__":
    main()