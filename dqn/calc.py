import base64
from collections import deque
import os
import pathlib
import shutil

from IPython import display as ipydisplay
import torch

from utils_env import MyEnv
from utils_drl import Agent

for target in range(0,348):
    print(target,end="  ")
    model_name = f"model_{target:03d}"
    model_path = f"./models/{model_name}"
    device = torch.device("cuda")
    env = MyEnv(device)
    agent = Agent(env.get_action_dim(), device, 0.99, 0, 0, 0, 1, model_path)
    obs_queue = deque(maxlen=5)
    avg_reward, frames = env.evaluate(obs_queue, agent, render=True)
   
    print(f"Avg. Reward: {avg_reward:.1f}")