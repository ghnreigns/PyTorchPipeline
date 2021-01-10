"""A module for constructing machine learning models."""
import torch
import geffnet
import timm


class CustomModel(torch.nn.Module):
    """A custom model."""

    def __init__(self, config: type, pretrained: bool = True):
        """Construct a custom model."""
        super().__init__()
        self.config = config

        model_factory = geffnet.create_model if config.model_factory == "geffnet" else timm.create_model

        self.model = model_factory(
            model_weight_path_folder=config.paths["model_weight_path_folder"],
            model_name=config.model_name,
            pretrained=pretrained,
        )

        n_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(n_features, config.num_classes)

    def forward(self, input_neurons):
        """Define the computation performed at every call."""
        # TODO: add dropout layers, or the likes.
        output_predictions = self.model(input_neurons)
        return output_predictions


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
            torch.nn.Linear(self.backbone.out_features, config.embedding_size, bias=True),
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