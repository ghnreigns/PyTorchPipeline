"""A dataset loader."""
import os
from typing import Optional, List, Dict

import cv2
import numpy as np
import pandas as pd
import torch

from utils import check_df_ext, get_file_type


class CustomDataset(torch.utils.data.Dataset):
    """The Custom Dataset. transforms is now an abstract class"""

    def __init__(
        self,
        config: type,
        df: pd.DataFrame = None,
        file_list: List = None,
        transforms: type = None,
        transform_norm: bool = True,
        meta_features: bool = None,
        mode: str = "train",
    ):
        """Construct a Custom dataset."""

        self.df = df
        self.file_list = file_list
        self.config = config
        self.transforms = transforms
        self.transform_norm = transform_norm
        self.meta_features = meta_features
        self.mode = mode

        if self.transforms is None:
            assert self.transform_norm is False
            print(
                "Transforms is None and Transform Normalization is not " "initialized!"
            )

        self.image_extension = get_file_type(
            image_folder_path=config.paths["train_path"], allowed_extensions=None
        )

        self.df_has_ext = check_df_ext(df=self.df, col_name=config.image_col_name)
        """
        This part here says that if the df has extension for all the image name, like each cell has
        image.jpg behind, then we set self.image_extension to empty string so that os.path.join in
        getitem won't throw error.
        """

        if self.df_has_ext is True:
            self.image_extension = ""

    def __len__(self):
        """Get the dataset length."""
        return len(self.df) if self.df is not None else len(self.file_list)

    def __getitem__(self, idx: int):
        """Get a row from the dataset."""

        image_id = self.df[self.config.image_col_name].values[idx]
        """Setting label=None is a simple hack to bypass testset - because I want to use
           this class for both train and test. So I have to anticipate that testset's df has no
           column called label."""

        # label = None

        """ It seems like you cannot use label = None as when you initiate DataLoader, the collate function inside expects anything
        inside to be numpy or tensor form, which label is not; So I set label to be tensor[0] and if we are in train mode, it will overwrite anyways."""
        label = torch.zeros(1)

        ### Problem 2: I suddenly have a dataset whereby the images are named 1.2.826.0.1.3680043.8.498.jpg =.=|||
        ### This cause my check_df_ext(df=self.df, col_name=config.image_col_name) function above (called in utils)
        ### to throw error, let me know if there is an elegant fix. I did not really anticipate weird(?) symbols
        ### to appear in image name, so maybe next time I will see something like x12donal?...?trum_!.jpg

        if self.mode == "train":
            ### Problem 3: Encountered a situation where by dytpe of label must be torch.float32 when you are using BCE loss.
            ### This part here is quite perplexing because usually labels are int format. What is the best way to catch the dtype error here?
            label = self.df[self.config.class_col_name].values[idx]
            label = torch.as_tensor(data=label, dtype=torch.int64, device=None)
            image_path = os.path.join(
                self.config.paths["train_path"],
                "{}{}".format(image_id, self.image_extension),
            )

        else:
            image_path = os.path.join(
                self.config.paths["test_path"],
                "{}{}".format(image_id, self.image_extension),
            )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # return transpose if image is not square
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        if not self.transform_norm:
            image = image.astype(np.float32) / 255.0

        if self.transforms is not None:
            image = self.transforms.augment(image)
        else:
            image = torch.as_tensor(data=image, dtype=torch.float32, device=None)

        if self.meta_features is not None:
            meta = np.array(
                self.df.iloc[idx][self.meta_features].values, dtype=np.float32
            )
            return image_id, (image, meta), label

        # Note this is important if you use BCE loss. Must make labels to float for some reason
        if self.config.criterion_train == "BCEWithLogitsLoss":
            label = torch.as_tensor(data=label, dtype=torch.float32, device=None)

        return image_id, image, label


class CustomDataLoader:

    """
    Class which will return dictionary containing dataloaders for training and validation.

    Arguments:

        config : A dictionary which contains the following keys and corresponding values.
        Keys and Values of config
            train_paths : list of paths of images for training.
            valid_paths : list of paths of images for validation.
            train_targets : targets for training.
            valid_targets : targets for validation.
            train_augmentations : Albumentations augmentations to use while training.
            valid_augmentations : Albumentations augmentations to use while validation.

        Reason why using dictionary ? -> It will keep all of the data pipeline clean and simple.

    Return :
            Dictionary containing training dataloaders and validation dataloaders
    """

    def __init__(self, config: type, data_dict: Dict):
        self.config = config
        self.data_dict = data_dict
        self.train_dataset = CustomDataset(
            self.config, **self.data_dict["dataset_train_dict"]
        )
        self.valid_dataset = CustomDataset(
            self.config, **self.data_dict["dataset_val_dict"]
        )

    def get_loaders(self):

        """
        Function which will return dictionary of dataloaders
        Arguments:

            train_bs : Batch Size for train loader.
            valid_bs : Batch Size for valid loader.
            num_workers : num_workers to be used by dataloader.
            drop_last : whether to drop last batch or not.
            shuffle : whether to shuffle inputs
            sampler : if dataloader is going to use a custom sampler pass the sampler argument.

        Returns :

            Dictionary with Training and Validation Loaders.

        """

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, **self.data_dict["dataloader_train_dict"]
        )

        val_loader = torch.utils.data.DataLoader(
            self.valid_dataset, **self.data_dict["dataloader_val_dict"]
        )

        dataloader_dict = {"Train": train_loader, "Validation": val_loader}

        return dataloader_dict