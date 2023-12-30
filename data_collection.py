
import requests
from zipfile import ZipFile
from pathlib import Path
import os


def get_image_dir(dir_path: str,
                  data_link: str):

    """
    Downloads raw image data from a specified given link into specified directory
    and also unzipps the file to place under a given directory in particular
    order.

    Args:
    dir_path: Path to the desired directory where the data needs to be downloaded
    data_link: raw URL link of data in zip format

    Returns:
    A folder where train and test data are separated in separated directory

    Example usage:
    get_image_dir(dir_path = 'data',
                  data_link = 'https://xyz.com')
    """

    img_path = Path('data')

    img_save_path = img_path / 'img_directory'

    if Path.is_dir(img_save_path):

        print(f'{img_save_path} already exists.. Skipping the creation of directory\n')

    else:

        print(f'{img_save_path} does not exist.. Creating the path..\n')

        img_save_path.mkdir(parents = True, exist_ok = True)

    print('Downloading the zip file.\n')


    with open(img_path / 'zipfile.zip', 'wb') as f:

        zip_file = requests.get(data_link)

        f.write(zip_file.content)

        print('Download Complete.\n')


    with ZipFile(img_path / 'zipfile.zip', 'r') as f:

        f.extractall(img_save_path)

        print('Unzipping Complete.\n')

    os.remove(img_path / 'zipfile.zip')

    train_dir = img_save_path / 'train'

    test_dir = img_save_path / 'test'

    return train_dir, test_dir

