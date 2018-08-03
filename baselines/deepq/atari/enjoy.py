import argparse
import gym
import os
import numpy as np

from gym.monitoring import VideoRecorder

import baselines.common.tf_util as U

from baselines import deepq
from baselines.common.misc_util import (
    boolean_flag,
    SimpleMonitor,
)
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.deepq.atari.model import model, dueling_model


def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, default="Amidar", help="name of the game")
    parser.add_argument("--model-dir", type=str, required=True, help="load model from this directory. ")
    parser.add_argument("--video", type=str, default=None, help="Path to mp4 file where the video of first episode will be recorded.")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "clipped", default=False, help="whether you want to record results in a clipped environment")


    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    env = SimpleMonitor(env)
    env = wrap_dqn(env)
    return env


def play(env, act, stochastic, video_path, clipped, num_trials = 10):
    num_episodes = 0
    video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)
    obs = env.reset()
    reward = 0
    num_played = 0
    rewardArray = []
    while num_played < num_trials:
        env.unwrapped.render()
        video_recorder.capture_frame()
        action = act(np.array(obs)[None], stochastic=stochastic)[0]
        obs, rew, done, info = env.step(action)
        if clipped:
            rew = clip_score(rew)
        reward += rew
        if done:
            obs = env.reset()
        if len(info["rewards"]) > num_episodes:
            if len(info["rewards"]) == 1 and video_recorder.enabled:
                # save video of first episode
                print("Saved video.")
                video_recorder.close()
                video_recorder.enabled = False
            print(info["rewards"][-1])
            rewardArray.append(reward)
            reward = 0
            num_played += 1
            num_episodes = len(info["rewards"])
    return {"Nonclipped": info["rewards"], "Clipped": rewardArray}

def clip_score(reward):
    if reward > 0:
        return 1
    elif reward < 0:
        return (-1)
    return 0

if __name__ == '__main__':
    with U.make_session(4) as sess:
        args = parse_args()
        env = make_env(args.env)
        act = deepq.build_act(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=dueling_model if args.dueling else model,
            num_actions=env.action_space.n)
        U.load_state(os.path.join(args.model_dir, "saved"))
        trial_rewards = play(env, act, args.stochastic, args.video, args.clipped)

        with open("results.txt", "a") as text_file:
            # text_file.write("Clipped Agent: " + str(trial_rewards) + "\n")        
            text_file.write("NonClipped Agent: " + str(trial_rewards) + "\n")        


