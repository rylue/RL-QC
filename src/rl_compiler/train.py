"""
Example training script for the RLCompiler environment using Stable Baselines 3.

This file demonstrates a very basic, bare-bones framework for how RLCompiler could be trained. In
practice, this training cycle would be MUCH more detailed, including hyperparameter tuning, a formalized
ML pipeline, RLCompiler env wrappers for tasking, etc.

Proximal Policy Optimization (PPO) is expected to work well for this project, given its success
in discretized action spaces.
"""

import gymnasium as gym
from stable_baselines3 import PPO

from rlcompiler import RLCompiler

# circuit/backend placeholders
circuit = ...
backend_info = ...

def make_env():
    """
    Construct the RLCompiler environment. This wrapper encodes the quantum compilation problem
    for a given circuit to train on and given backend info.

    In practice, for robustness, the circuit and backend info should be from a list of different
    configurations.
    """
    env = RLCompiler(
        circuit=circuit,
        backend_info=backend_info,
        reward_mode="murali_proxy"
    )
    return env


def train():
    """
    Example PPO training procedure.
    """
    env = make_env()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1
    )

    model.learn(total_timesteps=100000)

    model.save("ppo_rlcompiler")


def evaluate():
    """
    Example evaluation/usage loop on a trained model
    """

    env = make_env()
    model = PPO.load("ppo_rlcompiler")

    obs, _ = env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print("Final layout:", info.get("final_layout"))

if __name__ == "__main__":
    train()