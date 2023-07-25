import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)
        # super().__init__(observation_space)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "blockworld_map":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                input_channels = subspace.shape[0]
                extractors[key] = nn.Sequential(SimpleCNN(input_channels), nn.Flatten())

                # total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
                total_concat_size += 128
            # elif key == "object_map":
            #     extractors[key] = nn.Sequential(SimpleCNN(4), nn.Flatten())
            #     # total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            #     total_concat_size += 128
            else:
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.

        return th.cat(encoded_tensor_list, dim=1)

class SimpleCNN(nn.Module):
    def __init__(self, input_channels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Reshape to [batch_size, 64 * 2 * 2]
        x = self.fc1(x)
        return x

# class SimpleCNN(nn.Module):
#     def __init__(self, input_channels):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(2)
#         # self.conv2 = nn.Conv2d(64, 128, kernel_size=1) 
#         # self.relu = nn.ReLU()
#         # self.maxpool = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(64 * 2 * 2, 128)
#         # self.relu = nn.ReLU()
#         # self.fc2 = nn.Linear(1000, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         # x = self.conv2(x)
#         # x = self.relu(x)
#         # x = self.maxpool(x)
#         x = x.view(-1, 64 * 2 * 2)
#         x = self.fc1(x)
#         # x = self.relu(x)
#         # x = self.fc2(x)
#         return x
