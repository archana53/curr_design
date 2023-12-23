import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes

        # TODO remove the hardcoding of input dimensions of proprio and extero
        self._proprio_net = nn.Linear(168, 16)
        total_concat_size += 16

        self._extero_net = nn.Sequential(nn.Linear(51, 16), nn.Linear(16, 16))
        total_concat_size += 16

        self._final_net = nn.Linear(32, 32)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        encoded_tensor_list.append(self._proprio_net(observations["proprio"]))
        encoded_tensor_list.append(self._extero_net(observations["extero"]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return self._final_net(torch.cat(encoded_tensor_list, dim=1))
