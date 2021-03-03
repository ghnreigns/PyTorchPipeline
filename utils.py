"""Some utility functions."""
import glob
import os
import random
from collections import Counter
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def check_file_type(image_folder_path, allowed_extensions: Optional[List] = None):
    if allowed_extensions is None:
        allowed_extensions = [".jpg", ".png", ".jpeg"]

    no_files_in_folder = len(glob.glob(os.path.join(image_folder_path, "*")))
    extension_type = ""
    no_files_allowed = 0

    for ext in allowed_extensions:
        no_files_allowed = len(
            glob.glob(os.path.join(image_folder_path, "*.{}".format(ext)))
        )
        if no_files_allowed > 0:
            extension_type = ext
            break

    assert (
        no_files_in_folder == no_files_allowed
    ), "The extension in the folder should all be the same, but found more than one extensions"
    return extension_type


def get_file_type(image_folder_path: str, allowed_extensions: Optional[List] = None):
    """Get the file type of images in a folder."""
    if allowed_extensions is None:
        allowed_extensions = [".jpg", ".png", ".jpeg"]

    file_list = os.listdir(image_folder_path)
    extension_type = [os.path.splitext(file)[-1].lower() for file in file_list]
    extension_dict = Counter(extension_type)
    assert (
        len(extension_dict.keys()) == 1
    ), "The extension in the folder should all be the same, "
    "but found {} extensions".format(extension_dict.keys)
    extension_type = list(extension_dict.keys())[0]
    assert extension_type in allowed_extensions
    return extension_type


""" Consider modifying this function below to check if the dataframe's
image id column has extension or not """


def check_df_ext(
    df: pd.DataFrame, col_name: str, allowed_extensions: Optional[List] = None
):
    """Get the image file extension used in a data frame."""
    if allowed_extensions is None:
        allowed_extensions = [".jpg", ".png", ".jpeg"]
    # check if the col has an extension, this is tricky.
    # if no extension, it gives default ""
    image_id_list = df[col_name].tolist()
    extension_type = [
        # Review Comments: os.path.splitext is guaranteed to return a 2-tuple,
        # so no need to use -1 index.
        os.path.splitext(image_id)[1].lower()
        for image_id in image_id_list
    ]

    assert (
        len(set(extension_type)) == 1
    ), "The extension in the image id should all be the same"
    if "" in extension_type:
        return False
    assert list(set(extension_type))[0] in allowed_extensions
    return True


# Check the image folder for corrupted images.


def image_corruption(image_folder_path, img_type):
    """Find images in a folder that are corrupted."""
    corrupted_images = filter(
        lambda path_name: cv2.imread(path_name) is None,
        glob.glob(os.path.join(image_folder_path, img_type)),
    )
    for image_name in corrupted_images:
        print("This image {} is corrupted!".format(os.path.basename(image_name)))


def check_image_size(image_folder_path, height=None, width=None):
    """Count the number of images having differing dimensions."""
    total_img_list = glob.glob(os.path.join(image_folder_path, "*"))
    counter = 0
    for image in tqdm(total_img_list, desc="Checking in progress"):
        try:
            img = cv2.imread(image)

            # Review Comments:
            #
            # I assume you were trying to initialize width and height
            # if they are not defined by the caller. I have rewritten
            # your code to do this successfully - before you were just
            # comparing the height and width of each image with
            # itself.
            if height is None:
                height = img.shape[1]

            if width is None:
                width = img.shape[0]

            if not (height == img.shape[1] and width == img.shape[0]):
                counter += 1
        # Review Comments: What exception are you trying to catch here?
        # In general, you should not have a bare except block.
        except:
            print("this {} is corrupted".format(image))
            continue
    return counter


def seed_all(seed: int = 1930):
    """Seed all random number generators."""
    print("Using Seed Number {}".format(seed))

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(seed)  # set fixed value for python built-in pseudo-random generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def seed_worker(_worker_id):
    """Seed a worker with the given ID."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
