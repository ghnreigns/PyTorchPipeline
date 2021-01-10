"""A module for performing image augmentations."""
from abc import ABC, abstractmethod
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from discolight.disco import disco
import numpy as np
import torch
import torchvision
import torchtoolbox.transform


class Augmentation(ABC):
    """A standard interface for performing augmentations."""

    # The object that contains the augmentations that may be specified
    # in the configuration information. This must be specified by the
    # implementing class.
    augmentations_store = None
    # The function or callable object that is used to construct a composition
    # of augmentations. This must be specified by the implementing class.
    compose_constructor = None

    @abstractmethod
    def augment(self, image):
        """Augment an image."""

    @classmethod
    def from_config(klass, augmentations):
        """Construct an augmentation from configuration data.

        This function takes a list of augmentations in the form

            [ {"name": "Augmentation1", "params": {"param1": 1.0, ...}},
              {"name": "Augmentation2"},
              ...
            ]

        and returns an Augmentation class which will perform a
        composition of the specified augmentations. The name of each
        augmentation corresponds to a member of the
        augmentations_store object. The function or callable object in
        compose_constructor will be used to construct the composition
        augmentation. This composition augmentation will then be
        supplied to the constructor of the Augmentation class to
        return a new class instance.
        """
        augmentation_objs = [
            getattr(klass.augmentations_store, augmentation["name"])(**augmentation.get("params", {}))
            for augmentation in augmentations
        ]

        return klass(klass.compose_constructor(augmentation_objs))


class AlbumentationsAugmentation(Augmentation):
    class AlbumentationsStore:
        """A wrapper that exposes ToTensorV2 alongside other augmentations."""

        def __getattr__(self, name):

            if name == "ToTensorV2":
                return ToTensorV2

            return getattr(albumentations, name)

    augmentations_store = AlbumentationsStore()
    compose_constructor = albumentations.Compose

    def __init__(self, transforms: albumentations.core.composition.Compose):
        self.transforms = transforms

    def augment(self, image):
        albu_dict = {"image": image}
        transform = self.transforms(**albu_dict)
        return transform["image"]


class DiscolightAugmentation(Augmentation):

    augmentations_store = disco

    @staticmethod
    def compose_constructor(augmentations):
        return disco.Sequence(augmentations=augmentations)

    def __init__(self, seq):
        self.seq = seq

    def augment(self, image):
        aug_image = self.seq.get_img(image)

        tensor = torch.as_tensor(data=aug_image, dtype=torch.float32, device=None).permute(2, 0, 1)
        return tensor


class TorchTransforms(Augmentation):
    class TorchTransformsStore:
        def __getattr__(self, name):

            if name == "AutoAugment":
                return AutoAugment

            return getattr(torchvision.transforms, name)

    augmentations_store = TorchTransformsStore()
    compose_constructor = torchvision.transforms.Compose

    def __init__(self, transforms: torchvision.transforms.transforms.Compose):
        self.transforms = transforms

    def augment(self, image):
        if isinstance(image, np.ndarray):
            image = torchvision.transforms.ToPILImage()(image)
        transformed_image = self.transforms(image)
        return transformed_image


class TorchToolBoxTransforms(Augmentation):

    augmentations_store = torchtoolbox.transform.transforms
    compose_constructor = torchtoolbox.transform.transforms.Compose

    def __init__(self, transforms: torchtoolbox.transform.transforms.Compose):
        self.transforms = transforms

    def augment(self, image):
        transformed_image = self.transforms(image)
        return transformed_image
