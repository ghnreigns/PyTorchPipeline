"""A module for performing image augmentations."""
from abc import ABC, abstractmethod
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
import torchtoolbox
from torchtoolbox.transform import Cutout


class Augmentation(ABC):
    @abstractmethod
    def augment(image):
        """Augment an image."""


class AlbumentationsAugmentation(Augmentation):
    def __init__(self, transforms: albumentations.core.composition.Compose):
        self.transforms = transforms

    def augment(self, image):
        albu_dict = {"image": image}
        transform = self.transforms(**albu_dict)
        return transform["image"]


class TorchTransforms(Augmentation):
    def __init__(self, transforms: torchvision.transforms.transforms.Compose):
        self.transforms = transforms

    def augment(self, image):
        if isinstance(image, np.ndarray):
            image = torchvision.transforms.ToPILImage()(image)
        transformed_image = self.transforms(image)
        return transformed_image


class TorchToolBoxTransforms(Augmentation):
    def __init__(self, transforms: torchtoolbox.transform.transforms.Compose):
        self.transforms = transforms

    def augment(self, image):
        transformed_image = self.transforms(image)
        return transformed_image


######For Ian#########
"""Technically, this seems more elegant, but KIV on where to put augment_config"""


class augment_config:
    train_augmentations = [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        AutoAugment(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    val_augmentations = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    test_augmentations = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


"""This function is exactly the same as the one below"""


def get_transforms_torchvision(config):
    transforms_train = torchvision.transforms.Compose([*augment_config.train_augmentations])

    transforms_val = torchvision.transforms.Compose([*augment_config.val_augmentations])

    return transforms_train, transforms_val


def get_transforms_torchvision(config):
    transforms_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            AutoAugment(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transforms_val = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transforms_train, transforms_val


def get_albu_transforms(config):
    transforms_train = albumentations.Compose(
        [
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightness(limit=0.2, p=0.75),
            albumentations.RandomContrast(limit=0.2, p=0.75),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            albumentations.Resize(height=config.image_size, width=config.image_size, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    transforms_val = albumentations.Compose(
        [
            albumentations.Resize(height=config.image_size, width=config.image_size, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    return transforms_train, transforms_val


def get_torchtoolbox_transforms(config):
    transforms_train = torchtoolbox.transform.Compose(
        [
            DrawHair(),
            torchtoolbox.transform.RandomResizedCrop(size=config.image_size, scale=(0.8, 1.0)),
            torchtoolbox.transform.RandomHorizontalFlip(),
            torchtoolbox.transform.RandomVerticalFlip(),
            Microscope(p=0.4),
            torchtoolbox.transform.ToTensor(),
            torchtoolbox.transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transforms_val = torchtoolbox.transform.Compose(
        [
            torchtoolbox.transform.ToTensor(),
            torchtoolbox.transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transforms_train, transforms_val
