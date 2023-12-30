
import torch
import torch.nn as nn
from torchinfo import summary


def pretrained_model(model, weights):

    weights = weights.DEFAULT
    model = model(weights = weights)

    for params in model.parameters():

        params.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p = 0.2, inplace = True),
        nn.Linear(in_features = 1280,
                  out_features = 3)
        )

    return model

def pre_trained_summary(model: torch.nn.Module,
                        demo_shape: torch.Tensor):

    return summary(model, demo_shape)
