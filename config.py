"""A perser for YAML configuration."""
import torch
import yamale


class YAMLConfig:
    """Stores configuration loaded from a YAML file."""

    _schema = yamale.make_schema(content="""
seed: int()
num_classes: int()
class_list: list(int())
batch_size: int()
n_epochs: int()
scheduler: str()
scheduler_params: map()
train_step_scheduler: bool()
val_step_scheduler: bool()
optimizer: str()
optimizer_params: map()
criterion: str()
criterion_val: str()
criterion_params: map()
image_size: int()
crop_size: map(int(), key=int())
verbose: int()
verbose_step: int()
num_folds: int()
image_col_name: str()
class_col_name: str()
group_kfold_split: str()
paths:
  log_path: str()
  train_path: str()
  csv_path: str()
  save_path: str()
  model_weight_path_folder: str()
effnet: str()
device: str()
augmentations_class: str()
augmentations_train: map(list(include('augmentation')), key=str())
augmentations_val: map(list(include('augmentation')), key=str())
---
augmentation:
  name: str()
  params: map(required=False)
""")

    def __init__(self, config_path):
        """Load configuration information from a YAML file."""
        config = yamale.make_data(config_path)

        yamale.validate(self._schema, config)

        self._config = config[0][0]

        self._config["device"] = torch.device(
            self._config["device"]
        ) if self._config["device"] != "auto" else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def __getattr__(self, name):
        """Return the given configuration parameter."""
        if name not in self._config:
            raise AttributeError(
                "No such configuration parameter {}".format(name))

        try:
            return self._config[name]
        except KeyError:
            raise AttributeError(
                "No such configuration parameter {}".format(name))
