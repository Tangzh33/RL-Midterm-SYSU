from typing import (
    Optional,
)

import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DuelingDQN as DQN

class Agent(object):

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,

            eps_start: float,
            eps_final: float,
            eps_decay: float,

            restore: Optional[str] = None,
    ) -> None:
        self.__action_dim = action_dim
        self.__device = device
        self.__gamma = gamma

        self.__eps_start = eps_start
        self.__eps_final = eps_final
        self.__eps_decay = eps_decay

        self.__eps = eps_start
        self.__r = random.Random()
        self.__r.seed(seed)

        self.__policy = DQN(action_dim, device).to(device)
        self.__target = DQN(action_dim, device).to(device)


        if restore is None:
            self.__policy.apply(DQN.init_weights)
        else:
            self.__policy.load_state_dict(torch.load(restore))
        self.__target.load_state_dict(self.__policy.state_dict())
        self.__optimizer = optim.Adam(
            self.__policy.parameters(),
            lr=0.0000625,
            eps=1.5e-4,
        )
        self.__target.eval()

    def run(self, state: TensorStack4, training: bool = False) -> int:
        """run suggests an action for the given state."""
        if training:
            self.__eps -= \
                (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)

        if self.__r.random() > self.__eps:
            with torch.no_grad():
                return self.__policy(state).max(1).indices.item()
        return self.__r.randint(0, self.__action_dim - 1)

    def learn(self, memory: ReplayMemory, batch_size: int) -> float:
        """learn trains the value network via TD-learning."""
        '''
        state_batch, action_batch, reward_batch, next_batch, done_batch = \
            memory.sample(batch_size)
        values = self.__policy(state_batch.float()).gather(1, action_batch)
        values_next = self.__target(next_batch.float()).max(1).values.detach()
        expected = (self.__gamma * values_next.unsqueeze(1)) * \
            (1. - done_batch) + reward_batch
        loss = F.smooth_l1_loss(values, expected)

        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.__policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()
        '''
        idxs, folded_states, actions, rewards, dones = memory.sample(batch_size)


        #states = torch.cat(states)
        #nexts = torch.cat(nexts)

        #t_state = torch.zeros((batch_size, 4 , 84, 84), dtype=torch.uint8)

        indices = torch.randint(0, high=32, size=(batch_size,))
        state_batch = folded_states[indices, :4].to(self.__device).float()
        next_batch = folded_states[indices, 1:].to(self.__device).float()
        done = dones[indices].to(self.__device).float()
        reward = rewards[indices].to(self.__device).float()
        action = actions[indices].to(self.__device)

        values = self.__policy(state_batch.float()).gather(1, action)
        values_next = self.__target(next_batch.float()).max(1).values.detach()
        expected = (self.__gamma * values_next.unsqueeze(1)) * \
                   (1. - done) + reward
        loss = F.smooth_l1_loss(values, expected)
        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.__policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()

        '''
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones).bool()



        values = self.__policy(states.float()).gather(1, actions)

        values_for_action = values[range(states.shape[0]), actions]

        values_next = self.__target(nexts.float()).max(1).values.detach()
        target_qvalues_for_actions = rewards + self.__gamma * values_next.unsqueeze(1)
        target_qvalues_for_actions = torch.where(dones , rewards , target_qvalues_for_actions)
        errors = (values_for_action - target_qvalues_for_actions).detach().cpu().squeeze().tolist()

        memory.tree.update(idxs,errors)
        loss = F.smooth_l1_loss(values_for_action, target_qvalues_for_actions.detach())

        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.__policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()
        '''
        return loss.item()

    def sync(self) -> None:
        """sync synchronizes the weights from the policy network to the target
        network."""
        self.__target.load_state_dict(self.__policy.state_dict())

    def save(self, path: str) -> None:
        """save saves the state dict of the policy network."""
        torch.save(self.__policy.state_dict(), path)
