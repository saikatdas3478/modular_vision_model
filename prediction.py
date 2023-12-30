
import torch
from PIL import Image
from torchvision import transforms

def single_predict(img_dir: str,
                   model_path: str,
                   model_instance: torch.nn.Module,
                   data_transform: transforms.Compose,
                   class_names: list,
                   device: torch.device):


    """Gives Prediction on a single image a PyTorch model from a target directory.

    Args:
        img_dir: Directory path of a target image for prediction.
        model_path: Directory path of a saved PyTorch model.
        model_instance: new model instance created to store all weights and biases
        data_transform: Transform function for resizing and preprocessing image.
        class_names: list of strings of all prediction classes or objects

    Example usage:
        single_predict(img_dir = '/image_directory/img.jpg',
                        model_path = '/model/model.pth' or '/model/model.pt',
                        model_instance = CompactVGG(...),
                        data_transform = data_transform,
                        class_names = ['pizza', 'steak', 'sushi'])
    """

    img = Image.open(fp = img_dir)

    transformed_img = data_transform(img).unsqueeze(dim = 0).to(device)

    model_instance.load_state_dict(torch.load(f = model_path))

    model_instance.eval()

    with torch.inference_mode():

        y_pred = model_instance(transformed_img)

        pred_index = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1).item()

    prediction = class_names[pred_index]

    return prediction
