# Training a model with one line of code

I have created a modular vision model, which can get commands and train as well as test the model with one line of command line code. Just like:
| !python --batch_size 32 --model_name 'pre_model' | 
this simple command can train our vision model with a pre_trained efficientnet_b1 with a custom food101 dataset (20% of actual dataset). There are also different other commands such as: 
|--num_epochs integer --hidden_units integer --learning_rate float --added_image image url|
The possibility is endless here. If you want to train the model on different other models, you just need to add another .py file with the model implemened on it. If you want to use another dataset (in zip format), just change the link in train file. Also if you want another dataset format just change some of codes in data_collection.py file and the rest of the code is reusable. 
Although, in order to get full benefit of this model. Some work needs to be done beforehand. I am listing some instructions below:

1. The first and foremost work that is needed is that it needs to be in a single directory or nothing will work.
2. You need to install all necessary libraries besides I mentioned in the requirments file.
3. You open the terminal and run "python modular_vision_model/train.py" or whatever directory you have have put into or order you followed
4. There are two models available one is pre_trained effiicentnet_bt1 and another one is a small VGG variant I built from scratch (although the accuracy is not that great, around 61%).
5. You can use pre_trained model by specifying "--model_name 'pre_model'" or the model built from scratch by specifying "--model_name 'new_model'".
6. You can add any image url for prediction (within three class of images pizza, steak and sushi) using "--added_image '{image url}'".
7. You can run any number of epochs with with "--num_epochs {any integer}", do based on any number of batch sizes with "--batch_size {integer}", train the model based on any number of hidden channels with "--hidden_units {integer}" or any learning rate with "--learning_rate {float}".
8. The possibility is endless as you can also add other models on your own, use other arguments and the changes in code is endless.
9. Make sure you have a GPU, or the wait will get tedious. 

Enjoy!!
