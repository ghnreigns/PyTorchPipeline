"""A module for constructing machine learning models."""
import torch.nn as nn
import sys

#sys.path.insert(0, "./gen-efficientnet-pytorch-master-hongnan")
import geffnet

######For Ian#########
"""
I want to be slightly more versatile, as some people uses geffnet, some uses timm from the same author;
Should I add a simpe if-else clause here?
"""
# Review Comments:
#
# Yes. You could do something like
#
# if config.model_factory == "geffnet":
#     import geffnet as model_factory
# else
#     import timm as model_factory
#
# Then, in CustomEfficientNet you can call
#
# model_factory.create_model(...) as long as geffnet and timm provide the same
# model creation interface. If not, you should provide a small wrapper so
# they do have the same interface.


class CustomEfficientNet(nn.Module):
    """A custom EfficientNet model."""
    def __init__(self, config: type, pretrained: bool = True):
        """Construct a custom EfficientNet model."""
        super().__init__()
        self.config = config
        self.model = geffnet.create_model(
            model_weight_path_folder=config.paths["model_weight_path_folder"],
            model_name=config.effnet,
            pretrained=pretrained,
        )
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, config.num_classes)

    # TODO: Change x to input_neurons, I feel it is more apt.
    def forward(self, input_neurons):
        """Define the computation performed at every call."""
        # TODO: add dropout layers, or the likes.
        output_predictions = self.model(input_neurons)
        return output_predictions
