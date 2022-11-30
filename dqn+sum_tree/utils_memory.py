import random
from typing import (
    Tuple,
)

import torch
import numpy as np
from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)
from SumTree import  SumTree

class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            a: float,
            e: float,
            device: TorchDevice,
    ) -> None:

        self.__device = device
        self.__capacity = capacity
        self.__a = a
        self.__e = e

        self.tree = SumTree(capacity)

        self.__size = 0
        self.__pos = 0

        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:


        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done
        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)

        #state = folded_state[:4].to(self.__device).float()
        #next_s = folded_state[ 1:].to(self.__device).float()
        data = (folded_state , action , reward , done)
        p = (np.abs(self.__capacity) + self.__e) ** self.__a
        self.tree.add(p,data)

    def sample(self, batch_size: int) :
        '''
        indices = torch.randint(0, high=self.__size, size=(batch_size,))
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        '''
        #actions, rewards , dones =  [], [], []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        folded_states = torch.zeros((batch_size, 5, 84, 84), dtype=torch.uint8)
        actions = torch.zeros((batch_size, 1), dtype=torch.long)
        rewards = torch.zeros((batch_size, 1), dtype=torch.int8)
        dones = torch.zeros((batch_size, 1), dtype=torch.bool)

        for i in range(batch_size):
            a = segment*i
            b = segment*(i+1)
            s = random.uniform(a,b)
            idx , p , data =self.tree.get(s)

            folded_state , action , reward ,  done  = data

            folded_states[i] = folded_state

            actions[i, 0] = action

            rewards[i, 0] = reward

            dones[i, 0] = done

            priorities.append(p)

            idxs.append(idx)



        return idxs ,folded_states, actions, rewards , dones

    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p)

    def __len__(self) -> int:
        return self.tree.n_entries
