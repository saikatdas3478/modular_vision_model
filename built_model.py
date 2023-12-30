

import torch
import torch.nn as nn
from torchinfo import summary


class CompactVGG(nn.Module):

    """Creates the CompactVGG architecture.

    Replicates the CompactVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.

    Returns:
    A model instance on the given shapes and randomly initiliazed weights and biases
  """

    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()

        self.first_conv_block = nn.Sequential(
            nn.Conv2d(in_channels = input_shape,
                      out_channels =  hidden_units,
                      kernel_size = (3, 3),
                      stride = 1,
                      padding = 0),

            nn.ReLU(),

            nn.Conv2d(in_channels = hidden_units,
                      out_channels =  hidden_units,
                      kernel_size = (3, 3),
                      stride = 2,
                      padding = 1),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size = (2, 2),
                         stride = 2)
        )

        self.second_conv_block = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units,
                      out_channels = hidden_units,
                      kernel_size = (3,3),
                      stride = 1,
                      padding = 0),

            nn.ReLU(),

            nn.Conv2d(in_channels = hidden_units,
                      out_channels = hidden_units,
                      kernel_size = (3, 3),
                      stride = 2,
                      padding = 1),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size = (2,2),
                         stride = 2)
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features = hidden_units * 13 * 13, #shape = (3,224,224)
                      out_features = output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.fc_block(self.second_conv_block(self.first_conv_block(x)))

def shape_summary(model: torch.nn.Module,
                  demo_shape: torch.Tensor):

        return summary(model, demo_shape)
