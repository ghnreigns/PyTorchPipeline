"""A module for constructing machine learning models."""
import functools

import torch

import geffnet
import timm
from utils import rsetattr


class CustomModel(torch.nn.Module):
    """A custom model."""

    def __init__(
        self,
        config: type,
        pretrained: bool = True,
        load_weight: bool = False,
        load_url: bool = False,
    ):
        """Construct a custom model."""
        super().__init__()
        self.config = config
        self.pretrained = pretrained
        self.load_weight = load_weight
        self.load_url = load_url
        self.out_features = None

        def __setattr__(self, name, value):
            self.model.__setattr__(self, name, value)

        _model_name_list = [
            "regnet",
            "csp",
            "res",
            "efficientnet",
            "densenet",
            "senet",
            "inception",
            "nfnet",
            "vit",
        ]
        _model_factory = (
            geffnet.create_model
            if config.model_factory == "geffnet"
            else timm.create_model
        )

        self.model = _model_factory(
            model_weight_path_folder=config.paths["model_weight_path_folder"],
            model_name=config.model_name,
            pretrained=self.pretrained,
        )

        # load pretrained weight that are not available on timm or geffnet; for example, when NFNet just came out, we do not have timm's pretrained weight
        if self.load_weight:
            self.model.load_state_dict(torch.load(config.paths["pretrained_weight"]))
        if self.load_url:
            # using torch hub to load url, can be beautified. https://pytorch.org/docs/stable/hub.html
            checkpoint = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f1-fc540f82.pth"
            self.model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    checkpoint, progress=True, map_location="cpu"
                )
            )

        # attributes, self.out_features = self.get_last_layer()
        # setattr(self.model, '.'.join(attributes[:-1]), torch.nn.Linear(self.out_features, config.num_classes))

        last_layer_attr_name, self.out_features, _ = self.get_last_layer()
        last_layer_attr_name = ".".join(last_layer_attr_name)
        #         self.model.head.fc = torch.nn.Linear(self.out_features, config.num_classes)
        rsetattr(
            self.model,
            last_layer_attr_name,
            torch.nn.Linear(self.out_features, config.num_classes),
        )
        # n_features = self.model.classifier.in_features
        # self.model.classifier = torch.nn.Linear(self.out_features, config.num_classes)

    def forward(self, input_neurons):
        """Define the computation performed at every call."""
        # TODO: add dropout layers, or the likes.
        output_predictions = self.model(input_neurons)
        return output_predictions

    def get_last_layer(self):
        last_layer_name = None
        for name, param in self.model.named_modules():
            last_layer_name = name

        last_layer_attributes = last_layer_name.split(".")  # + ['in_features']
        linear_layer = functools.reduce(getattr, last_layer_attributes, self.model)
        # reduce applies to a list recursively and reduce
        in_features = functools.reduce(
            getattr, last_layer_attributes, self.model
        ).in_features
        return last_layer_attributes, in_features, linear_layer


### Problem: In timm module, some of the models have different names for the below two lines of code.
### If model name = resnext50_32x4d, then the below should be like this.
### n_features = self.model.fc.in_features
### self.model.fc = nn.Linear(n_features, config.num_classes)
### A possible approach is as follows, extracted from my friend's notebook.


class Backbone(torch.nn.Module):
    """Backbone refers to the model's feature extractor. It is not a well defined word in my opinion, but it is
    so often used in Deep Learning papers so that it is probably coined to mean what it means, literally - the
    backbone of a model. In other words, if we are using a pretrained EfficientNetB6, we will definitely strip off
    the last layer of the network, and replace it with our own layer as seen in the code below; however, we are using
    EfficientNetB6 as the FEATURE EXTRACTOR/BACKBONE because we are using almost all its layers, except for the last layer."""

    def __init__(self, name="resnet18", pretrained=True):
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained)

        if "regnet" in name:
            self.out_features = self.net.head.fc.in_features
        elif "csp" in name:
            self.out_features = self.net.head.fc.in_features
        elif "res" in name:  # works also for resnest
            self.out_features = self.net.fc.in_features
        elif "efficientnet" in name:
            self.out_features = self.net.classifier.in_features
        elif "densenet" in name:
            self.out_features = self.net.classifier.in_features
        elif "senet" in name:
            self.out_features = self.net.fc.in_features
        elif "inception" in name:
            self.out_features = self.net.last_linear.in_features

        else:
            self.out_features = self.net.classifier.in_features

    def forward(self, x):
        x = self.net.forward_features(x)

        return x


class Net(torch.nn.Module):
    def __init__(self):

        super().__init__()
        self.backbone = Backbone(name=config.name, pretrained=True)

        if config.pool == "gem":
            self.global_pool = GeM(p_trainable=config.p_trainable)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.neck = nn.Sequential(
            torch.nn.Linear(
                self.backbone.out_features, config.embedding_size, bias=True
            ),
            torch.nn.BatchNorm1d(config.embedding_size),
            torch.nn.PReLU(),
        )
        self.head = ArcMarginProduct(config.embedding_size, config.num_classes)

    def forward(self, x):

        bs, _, _, _ = x.shape
        x = self.backbone(x)
        x = self.global_pool(x).reshape(bs, -1)

        x = self.neck(x)

        logits = self.head(x)

        return logits