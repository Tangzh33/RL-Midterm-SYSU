import torch
import torch.nn as nn
import torch.nn.functional as F


# class DQN(nn.Module):

#     def __init__(self, action_dim, device):
#         super(DQN, self).__init__()
#         self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
#         self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
#         self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
#         self.__fc1 = nn.Linear(64*7*7, 512)
#         self.__fc2 = nn.Linear(512, action_dim)
#         self.__device = device

#     def forward(self, x):
#         x = x / 255.
#         x = F.relu(self.__conv1(x))
#         x = F.relu(self.__conv2(x))
#         x = F.relu(self.__conv3(x))
#         x = F.relu(self.__fc1(x.view(x.size(0), -1)))
#         return self.__fc2(x)

#     @staticmethod
#     def init_weights(module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
#             module.bias.data.fill_(0.0)
#         elif isinstance(module, nn.Conv2d):
#             torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")


            
class DuelingDQN(nn.Module):
    def __init__(self, action_dim, device):

        super(DuelingDQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2,bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1,bias=False)
        self.state_value = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1),
        )
        self.action_value = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=action_dim),
        )
        self.__device = device


    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = x.view(x.size(0), -1)

        action_value = self.action_value(x)
        state_value = self.state_value(x)
        # Q(s,a) = V(s) + (A(s,a)-avgA(s,a))
        return state_value + action_value - action_value.mean()


    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight,nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight,nonlinearity="relu")
