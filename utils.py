

import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               model_dir: str,
               model_name: str):

    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,
                    target_dir="models",
                    model_name="tingvgg_model.pth")
    """

    assert model_name.endswith('.pth') or model_name.endswith('.pt'), 'Model is not in .pth or .pt format\n'


    model_dir_path = Path(model_dir)

    model_dir_path.mkdir(parents = True,
                     exist_ok = True)

    model_save_path = model_dir_path / model_name

    print(f'Saving model to: {model_save_path}\n')


    torch.save(obj = model.state_dict(),
               f = model_save_path)

    print(f'Model Saved!')
