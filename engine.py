
import os
import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

if not os.path.exists('experiment_tracking'):
    os.mkdir('experiment_tracking')

writer = SummaryWriter(log_dir = 'experiment_tracking')

def accuracy_func(y_pred, y_true):

    correctness = torch.eq(y_true, y_pred).sum().item()

    return (correctness/ len(y_true)) * 100



def training_step(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_func: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  device: torch.device,
                  acc_func = accuracy_func):


    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """

    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X_train, y_train) in enumerate(dataloader):

        X_train, y_train = X_train.to(device), y_train.to(device)

        y_pred = model(X_train)

        loss = loss_func(y_pred, y_train)

        y_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)

        train_acc += acc_func(y_class, y_train)

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(dataloader)

    train_acc /= len(dataloader)

    return train_loss, train_acc


def testing_step(model: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 loss_func: torch.nn.Module,
                 device: torch.device,
                 acc_func = accuracy_func):

    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
  """

    model.eval()

    test_loss, test_acc = 0, 0

    for batch, (X_test, y_test) in enumerate(dataloader):

        X_test, y_test = X_test.to(device), y_test.to(device)

        y_pred = model(X_test)

        loss = loss_func(y_pred, y_test)

        test_loss += loss.item()

        y_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)

        test_acc += acc_func(y_class, y_test)

    test_loss /= len(dataloader)

    test_acc /= len(dataloader)

    return test_loss, test_acc


def Train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_func: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device):

    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through training_step() and testing_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]}
        For example if training for epochs=2:
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]}
    """

    all_metrics = {'train_loss': [],
                   'train_acc': [],
                   'test_loss': [],
                   'test_acc': []}


    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = training_step(model,
                                              train_dataloader,
                                              loss_func,
                                              optimizer,
                                              device)

        test_loss, test_acc = testing_step(model,
                                           test_dataloader,
                                           loss_func,
                                           device)


        print(f'Epoch: {epoch+1} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Train Accuracy: {train_acc:.2f} | '
            f'Test Loss: {test_loss:.4f} | '
            f'Test Accuracy: {test_acc:.2f}')


        all_metrics['train_loss'].append(train_loss)
        all_metrics['train_acc'].append(train_acc)
        all_metrics['test_loss'].append(test_loss)
        all_metrics['test_acc'].append(test_acc)

        writer.add_scalars(main_tag = 'Loss',
                           tag_scalar_dict = {'Train Loss': train_loss,
                                              'Test Loss': test_loss},
                           global_step = epoch)

        writer.add_scalars(main_tag = 'Accuracy',
                           tag_scalar_dict = {'Train Accuracy' : train_acc,
                                              'Test Accuracy': test_acc},
                           global_step = epoch)

        writer.add_graph(model = model,
                         input_to_model = torch.randn(32, 3, 224, 224).to(device))

    writer.close()
    return all_metrics

