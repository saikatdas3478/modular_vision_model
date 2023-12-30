
import torch
import os
import argparse
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from data_collection import get_image_dir
from data_loader import preparation_dataloaders
from built_model import CompactVGG, shape_summary
from pre_trained import pretrained_model
from engine import Train
from urllib.request import urlretrieve
from utils import save_model
from prediction import single_predict
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

#Single image link

ADDED_IMAGE = 'https://www.eatingwell.com/thmb/PhRj8Sp6g5m-Cn9AJL2zeLi1LM4=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/easy-vegan-pizza-1x1-002-a224f13696b3483d8099b7ae5b494250.jpg'

# hyperparameters parsing with arguments

parser = argparse.ArgumentParser(description = 'Getting hyperparameters')

parser.add_argument('--batch_size',
                  default = 32,
                  type = int)

parser.add_argument('--num_epochs',
                  default = 5,
                  type = int)

parser.add_argument('--learning_rate',
                    default = 0.001,
                    type = float)

parser.add_argument('--hidden_units',
                    default = 10,
                    type = int)

parser.add_argument('--added_image',
                    default = ADDED_IMAGE)

parser.add_argument('--model_name',
                    default = 'pre_model')

parse = parser.parse_args()

BATCH_SIZE = parse.batch_size
LEARNING_RATE = parse.learning_rate
HIDDEN_UNITS = parse.hidden_units
NUM_EPOCHS = parse.num_epochs
MODEL_SELECTED = parse.model_name
ADDED_IMAGE = parse.added_image

# Image data link

data_link = 'https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip'

# Getting the train and test directory images

train_dir, test_dir = get_image_dir(dir_path = 'data', data_link = data_link)

# Checking either gpu or cpu avaliable

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# setting transform object for resizing and tensor conversion of images
data_transform = transforms.Compose([
    transforms.Resize(size = (224, 224)),
    transforms.ToTensor()
])

# creating train_dataloaders

train_dataloader, test_dataloader, class_names = preparation_dataloaders(train_dir = train_dir,
                                                                        test_dir = test_dir,
                                                                        transform = data_transform,
                                                                        batch_size = BATCH_SIZE)

if MODEL_SELECTED == 'new_model':

    # Compact VGG model

    model_name = 'compact_vgg_2conv_1fc.pth'
    model_dir = 'compactvgg_model'

    model = CompactVGG(input_shape = 3,
                        hidden_units = HIDDEN_UNITS,
                        output_shape = len(class_names)).to(device)

    # Setting loss and optimizer

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params = model.parameters(),
                                lr = LEARNING_RATE)

    all_metrics = Train(model = model,
                        train_dataloader = train_dataloader,
                        test_dataloader = test_dataloader,
                        loss_func = loss,
                        optimizer = optimizer,
                        epochs = NUM_EPOCHS,
                        device = device)

    print('CompactVGG Model is Trained.\n')

    save_model(model = model,
                model_dir = model_dir,
                model_name = model_name)

    print('CompactVGG Model is Saved.\n')

    new_model = CompactVGG(input_shape = 3,
                            hidden_units = HIDDEN_UNITS,
                            output_shape = len(class_names)).to(device)


elif MODEL_SELECTED == 'pre_model':

    # Pretrained EfficientNet with B1 version

    model_name = 'efficientnet_b1.pth'
    model_dir = 'efficientnet_b1_model'

    model = pretrained_model(model = efficientnet_b1,
                            weights = EfficientNet_B1_Weights.DEFAULT).to(device)

    # Setting loss and optimizer

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params = model.parameters(),
                             lr = LEARNING_RATE)

    # Training and testing model at same time

    all_metrics = Train(model = model,
                        train_dataloader = train_dataloader,
                        test_dataloader = test_dataloader,
                        loss_func = loss,
                        optimizer = optimizer,
                        epochs = NUM_EPOCHS,
                        device = device)

    print('EfficientNet Model is Trained.\n')

    # Finally saving the model

    save_model(model = model,
           model_dir = model_dir,
           model_name = model_name)

    print('EfficientNet Model is Saved.\n')

    new_model = pretrained_model(model = efficientnet_b1,
                                weights = EfficientNet_B1_Weights.DEFAULT).to(device)


else:
    print('such model does not exists.')

# Predicting based on a new image

if MODEL_SELECTED in ['pre_model', 'new_model']:

    img_dir = urlretrieve(ADDED_IMAGE)[0]

    pred = single_predict(img_dir = img_dir,
                            model_path = f'{model_dir}/{model_name}',
                            model_instance = new_model,
                            data_transform = data_transform,
                            class_names = class_names,
                            device = device)

    print(f'According to {model_name}, This is a {pred}.')

else:
    print('Try Again with correct model name!')
