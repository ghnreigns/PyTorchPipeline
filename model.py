"""A module for constructing machine learning models."""
import torch.nn as nn

import geffnet
import timm


class CustomEfficientNet(nn.Module):
    """A custom EfficientNet model."""
    def __init__(self, config: type, pretrained: bool = True):
        """Construct a custom EfficientNet model."""
        super().__init__()
        self.config = config

        model_factory = (geffnet.create_model if config.model_factory
                         == "geffnet" else timm.create_model)

        self.model = model_factory(
            model_weight_path_folder=config.paths["model_weight_path_folder"],
            model_name=config.effnet,
            pretrained=pretrained,
        )
        ### Problem: In timm module, some of the models have different names for the below two lines of code.
        ### If model name = resnext50_32x4d, then the below should be like this.
        ### n_features = self.model.fc.in_features
        ### self.model.fc = nn.Linear(n_features, config.num_classes)
        
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, config.num_classes)

    def forward(self, input_neurons):
        """Define the computation performed at every call."""
        # TODO: add dropout layers, or the likes.
        output_predictions = self.model(input_neurons)
        return output_predictions
