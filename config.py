"""A perser for YAML configuration."""
import inspect
import multiprocessing
import torch
import yaml
import yamale

import results


class YAMLConfig:
    """Stores configuration loaded from a YAML file."""

    _schema = yamale.make_schema(
        content="""
seed: int()
num_workers: any(enum('auto'), int(min=0))
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
results_val: list(str())
results_train: list(str())
monitored_result: str()
result_params: map()
image_size: int()
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
model_factory: enum('geffnet', 'timm')
model_name: str()
device: str()
augmentations_class: str()
augmentations_train: map(list(include('augmentation')), key=str())
augmentations_val: map(list(include('augmentation')), key=str())
---
augmentation:
  name: str()
  params: map(required=False)
"""
    )

    def _generate_default_configuration(self, item_name, params_key, factory):
        """Generate default configuration for a selected item.

        If an entry for the given item is not found under the given params_key,
        this function invokes the factory to get the function for constructing
        the given item. The default parameters are then inserted into the
        configuration. Parameters with no default values are ignored.

        :param item_name:
            The item name
        :param params_key:
            The key in the configuration that stores the configuration for
            each item type.
        :param factory:
            A function that, given an item name, returns the function used
            to construct that item.

        :returns: True if the configuration was updated, and False otherwise
        """
        if item_name in self._config[params_key]:
            return False

        item = factory(item_name)

        default_values = {
            param.name: param.default
            for param in inspect.signature(item).parameters.values()
            if param.default != inspect.Parameter.empty
        }

        # Some PyTorch optimizers mark required paramaters by setting
        # the default value to the special object
        # torch.optim.optimizer.required. Since this value is not
        # actually accessible to us (its hidden within the torch.optim
        # module), we check its stringified signature.
        found_required_value = False

        for name, default in default_values.items():
            if str(default) == "<required parameter>":
                default_values[name] = "VALUE REQUIRED"
                found_required_value = True

        if found_required_value:
            raise ValueError(
                "Failed to generate a default configuration for "
                "{} because some of its required parameters "
                "do not have an associated default value. You "
                "should check the documentation for {} and "
                "manually enter its configuration under {}. Here "
                "are the values you must provide (some with "
                "defaults we found):\n\n{}".format(
                    item_name,
                    item_name,
                    params_key,
                    "\n".join(["{}: {}".format(name, value) for (name, value) in default_values.items()]),
                )
            )

        self._config[params_key][item_name] = default_values

        return True

    def __init__(self, config_path, save_updated_configuration=True):
        """Load configuration information from a YAML file."""
        config = yamale.make_data(config_path)

        yamale.validate(self._schema, config)

        self._config = config[0][0]

        configuration_updated = False

        for result in self._config["results_val"]:
            configuration_updated |= self._generate_default_configuration(
                result, "result_params", lambda metric: results._results[result].__init__
            )

        for result in self._config["results_train"]:
            configuration_updated |= self._generate_default_configuration(
                result, "result_params", lambda metric: results._results[result].__init__
            )

        configuration_updated |= self._generate_default_configuration(
            self._config["scheduler"],
            "scheduler_params",
            lambda scheduler: getattr(torch.optim.lr_scheduler, scheduler).__init__,
        )

        configuration_updated |= self._generate_default_configuration(
            self._config["criterion"], "criterion_params", lambda criterion: getattr(torch.nn, criterion).__init__
        )

        configuration_updated |= self._generate_default_configuration(
            self._config["criterion_val"], "criterion_params", lambda criterion: getattr(torch.nn, criterion).__init__
        )

        configuration_updated |= self._generate_default_configuration(
            self._config["optimizer"], "optimizer_params", lambda optimizer: getattr(torch.optim, optimizer).__init__
        )

        configuration_updated |= self._generate_default_configuration(
            self._config["scheduler"], "scheduler_params", lambda scheduler: getattr(torch.optim.lr_scheduler).__init__
        )

        if configuration_updated and save_updated_configuration:
            with open(config_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(self._config, yaml_file)

        self._config["device"] = (
            torch.device(self._config["device"])
            if self._config["device"] != "auto"
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if self._config["num_workers"] == "auto":
            self._config["num_workers"] = multiprocessing.cpu_count()

    def __getattr__(self, name):
        """Return the given configuration parameter."""
        if name not in self._config:
            raise AttributeError("No such configuration parameter {}".format(name))

        try:
            return self._config[name]
        except KeyError:
            raise AttributeError("No such configuration parameter {}".format(name))
