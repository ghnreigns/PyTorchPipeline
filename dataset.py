"""A dataset loader."""
import os
from typing import Optional
import cv2
import numpy as np
import pandas as pd
import torch

from utils import check_df_ext, get_file_type


class Melanoma(torch.utils.data.Dataset):
    """The Melanoma dataset. transforms is now an abstract class"""
    def __init__(
        self,
        df: pd.DataFrame,
        config: type,
        transforms: type = None,
        test: bool = False,
        transform_norm: bool = True,
        meta_features=None,
    ):
        """Construct a Melanoma dataset."""

        self.df = df
        self.config = config
        self.transforms = transforms
        self.test = test
        self.transform_norm = transform_norm
        self.meta_features = meta_features

        if self.transforms is None:
            assert self.transform_norm is False
            print("Transforms is None and Transform Normalization is not "
                  "initialized!")

        self.image_extension = get_file_type(
            image_folder_path=config.paths["train_path"],
            allowed_extensions=None)

        self.df_has_ext = check_df_ext(df=self.df,
                                       col_name=config.image_col_name)
        """
        This part here says that if the df has extension for all the image name, like each cell has
        image.jpg behind, then we set self.image_extension to empty string so that os.path.join in
        getitem won't throw error.
        """

        if self.df_has_ext is True:
            self.image_extension = ""

    def __len__(self):
        """Get the dataset length."""
        return len(self.df)

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
        
        if self.test:
            image_path = os.path.join(
                self.config.paths["test_path"],
                "{}{}".format(image_id, self.image_extension))
        else:
            label = self.df[self.config.class_col_name].values[idx]
            label = torch.as_tensor(data=label, dtype=torch.int64, device=None)
            image_path = os.path.join(
                self.config.paths["train_path"],
                "{}{}".format(image_id, self.image_extension))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.transform_norm:
            image = image.astype(np.float32) / 255.0

        if self.transforms is not None:
            image = self.transforms.augment(image)
        else:
            image = torch.as_tensor(data=image,
                                    dtype=torch.float32,
                                    device=None)

        if self.meta_features is not None:
            meta = np.array(self.df.iloc[idx][self.meta_features].values,
                            dtype=np.float32)
            return image_id, (image, meta), label

        return image_id, image, label
