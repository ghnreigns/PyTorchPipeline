"""Model training."""

import datetime
import os
import random
import time

import numpy as np
import pandas as pd
import pytz
import sklearn
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

import metrics
import results
import transforms
from config import YAMLConfig
from cross_validate import make_folds
from dataset import CustomDataset
from dataset import CustomDataLoader
from loss import LabelSmoothingLoss
from model import CustomModel
from oof import get_oof_acc, get_oof_roc
from utils import seed_all, seed_worker


class Trainer:
    """A class to perform model training."""

    def __init__(self, model, config, early_stopping=None):
        """Construct a Trainer instance."""
        self.model = model
        self.config = config
        self.early_stopping = early_stopping
        self.epoch = 0
        # loss history and monitored metrics history stores each epoch's results, and save it to the weights, so later we can access it easily, we can also access it by calling the attribute
        self.loss_history = []
        self.monitored_metrics_history = []
        self.save_path = config.paths["save_path"]
        if not os.path.exists(self.save_path):
            print("new save folder created")
            os.makedirs(self.save_path)

        # Uncomment this if needed to use different val loss #
        # self.criterion = LabelSmoothingLoss(**config.criterion_params[config.criterion]).to(self.config.device)
        self.criterion_train = getattr(torch.nn, config.criterion_train)(
            **config.criterion_params[config.criterion_train]
        )
        self.criterion_val = getattr(torch.nn, config.criterion_val)(
            **config.criterion_params[config.criterion_val]
        )
        self.optimizer = getattr(torch.optim, config.optimizer)(
            self.model.parameters(), **config.optimizer_params[config.optimizer]
        )
        self.scheduler = getattr(torch.optim.lr_scheduler, config.scheduler)(
            optimizer=self.optimizer, **config.scheduler_params[config.scheduler]
        )

        """scaler is only used when use_amp is True, use_amp is inside config."""
        if config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.selected_results_val = [
            results.construct_result(result, self.config)
            for result in self.config.results_val
        ]
        # print(self.selected_results_val)
        # -> [<results.average_loss object at 0x0000020CF63C67F0>, <results.average_accuracy object at 0x0000020CF63C68E0>,
        #  <results.val_preds_softmax_array object at 0x0000020CF63C6940>, <results.val_roc_auc_score object at 0x0000020CF63C69A0>,
        #  <results.multi_class_roc_auc_score object at 0x0000020CF63C69D0>]

        self.validation_results = results.ValidationResults(
            self, self.selected_results_val, self.config
        )
        # print(self.validation_results)
        # -> <results.ValidationResults object at 0x000001BCFA265850>

        self.selected_results_train = [
            results.construct_result(result, self.config)
            for result in self.config.results_train
        ]
        # print(self.selected_results_train)
        # -> [<results.average_loss object at 0x000001BCFA265880>, <results.average_accuracy object at 0x000001BCFA265970>,
        #  <results.val_preds_softmax_array object at 0x000001BCFA2659D0>, <results.val_roc_auc_score object at 0x000001BCFA265A30>,
        #  <results.multi_class_roc_auc_score object at 0x000001BCFA265A60>]

        self.training_results = results.TrainingResults(
            self, self.selected_results_train, self.config
        )
        # print(self.training_results)
        # -> <results.TrainingResults
        # The current-known best values for selected ComparableResult
        # validation results
        self.best_val_results = {}
        # The current-known values for selected savable validation results
        self.saved_val_results = {}
        """https://stackoverflow.com/questions/1398674/display-the-time-in-a-different-time-zone"""
        self.date = datetime.datetime.now(pytz.timezone("Asia/Singapore")).strftime(
            "%Y-%m-%d"
        )

        self.log(
            "[Trainer prepared]: We are using {} device with {} worker(s).\nThe monitored metric is {}\n".format(
                self.config.device,
                self.config.num_workers,
                self.config.monitored_result,
            )
        )

        self.log(
            "We are dealing with a Multiclass/label problem, using softmax/sigmoid, etc"
        )

    def fit(self, train_loader, val_loader, fold: int):
        """Fit the model on the given fold."""
        self.log(
            "Training on Fold {} and using {}".format(fold, self.config.model_name)
        )

        for _epoch in range(self.config.n_epochs):
            # Getting the learning rate after each epoch!
            lr = self.optimizer.param_groups[0]["lr"]

            timestamp = datetime.datetime.now(pytz.timezone("Asia/Singapore")).strftime(
                "%Y-%m-%d %H-%M-%S"
            )
            # printing the lr and the timestamp after each epoch.
            self.log("\n{}\nLR: {}".format(timestamp, lr))

            # start time of training on the training set
            train_start_time = time.time()
            # train one epoch on the training set
            train_results_computed = self.train_one_epoch(train_loader)
            # end time of training on the training set
            train_end_time = time.time()
            # formatting time to make it nicer
            train_elapsed_time = time.strftime(
                "%H:%M:%S", time.gmtime(train_end_time - train_start_time)
            )

            train_reported_results = [
                result.report(train_results_computed[result.__class__.__name__])
                for result in self.selected_results_train
                if isinstance(result, results.ReportableResult)
            ]
            # print(train_reported_results)
            # -> ['Avg Validation Summary Loss: 0.527126', 'Validation Accuracy: 0.791667']
            train_result_str = " | ".join(
                [
                    "Training Epoch: {}".format(self.epoch + 1),
                    *train_reported_results,
                    "Time Elapsed: {}".format(train_elapsed_time),
                ]
            )

            self.log("[TRAIN RESULT]: {}".format(train_result_str))

            val_start_time = time.time()
            """
            It suffices to understand self.valid_one_epoch(val_loader)
            So val_results_computed returns the following:
            
            """
            val_results_computed = self.valid_one_epoch(val_loader)
            # print(val_results_computed)
            # -> {'average_loss': tensor(0.1868, device='cuda:0'),
            #     'average_accuracy': 0.95,
            #     'val_preds_softmax_array': array([[0.985, 0.014],[0.977, 0.0229]], dtype=float32),
            #     'val_roc_auc_score': 0.521,
            #     'multi_class_roc_auc_score': 0.521}
            # print(val_results_computed["val_preds_softmax_array"].shape)
            val_end_time = time.time()
            val_elapsed_time = time.strftime(
                "%H:%M:%S", time.gmtime(val_end_time - val_start_time)
            )
            ## Important addition for plotting
            self.loss_history.append(val_results_computed["average_loss"])
            self.monitored_metrics_history.append(
                val_results_computed[self.config.monitored_result]
            )
            ## end

            # Save the current value of all savable validation results
            # This for loop must be after valid one epoch for apparent reason you need the results computed.
            for result in self.selected_results_val:
                # print(result, isinstance(result, results.SavableResult))
                # <results.average_loss object at 0x000001B174223880> False
                # <results.average_accuracy object at 0x000001B174223970> False
                # <results.val_preds_softmax_array object at 0x000001B1742239D0> True
                # <results.val_roc_auc_score object at 0x000001B174223A30> False
                # <results.multi_class_roc_auc_score object at 0x000001B174223A60> False

                if not isinstance(result, results.SavableResult):
                    continue

                # gets name
                savable_name = result.get_save_name(
                    val_results_computed[result.__class__.__name__]
                )
                # print(savable_name) -> "oof_preds" refer to results.py val_preds_softmax_array class, this signifies we want to save this.

                # If savable_name is reported as None, we don't save the
                # result value.
                if savable_name is None:
                    continue

                self.saved_val_results[savable_name] = val_results_computed[
                    result.__class__.__name__
                ]
                # print(len(self.saved_val_results))
                # print(self.saved_val_results) here get the savable name oof_preds, and then get the class name to be val_preds_softmax_array
                # ->
                # {'oof_preds': array([[0.8899242 , 0.11007578],
                #                      [0.72081256, 0.27918747],
                #                      [0.68776596, 0.31223404],
                #                      [0.7657404 , 0.23425962]}

            val_reported_results = [
                result.report(val_results_computed[result.__class__.__name__])
                for result in self.selected_results_val
                if isinstance(result, results.ReportableResult)
            ]
            # print(val_reported_results) - > ['Avg Validation Summary Loss: 0.035370', 'Validation Accuracy: 0.996047', 'Validation ROC: 0.261905', 'MultiClass ROC: 0.26190476190476186']
            val_result_str = " | ".join(
                [
                    "Validation Epoch: {}".format(self.epoch + 1),
                    *val_reported_results,
                    "Time Elapsed: {}".format(val_elapsed_time),
                ]
            )

            self.log("[VAL RESULT]: {}".format(val_result_str))

            if self.early_stopping is not None:
                best_score, early_stop = self.early_stopping.should_stop(
                    curr_epoch_score=val_results_computed[self.config.monitored_result]
                )
                """
                Be careful of self.best_loss here, when our monitered_metrics is val_roc_auc, then we should instead write
                self.best_auc = best_score. After which, if early_stop flag becomes True, then we break out of the training loop.
                """

                self.best_val_results[self.config.monitored_result] = best_score
                self.save(
                    "{}_best_{}_fold_{}.pt".format(
                        self.config.model_name, self.config.monitored_result, fold
                    )
                )
                if early_stop:
                    break

            """
            Compute the new best value for all selected ComparableMetric validation results.
            If we find a new best value for the selected monitored result, save the model.
            """

            for result in self.selected_results_val:
                if not isinstance(result, results.ComparableResult):
                    continue

                old_value = self.best_val_results.get(result.__class__.__name__, None)

                if old_value is None:
                    self.best_val_results[
                        result.__class__.__name__
                    ] = val_results_computed[result.__class__.__name__]

                    if result.__class__.__name__ == self.config.monitored_result:
                        self.save(
                            os.path.join(
                                self.save_path,
                                "{}_{}_best_{}_fold_{}.pt".format(
                                    self.date,
                                    self.config.model_name,
                                    self.config.monitored_result,
                                    fold,
                                ),
                            )
                        )

                    continue

                new_value = val_results_computed[result.__class__.__name__]

                if result.compare(old_value, new_value):
                    self.best_val_results[result.__class__.__name__] = new_value

                    if result.__class__.__name__ == self.config.monitored_result:
                        self.log(
                            "Saving epoch {} of fold {} as best weights".format(
                                self.epoch + 1, fold
                            )
                        )
                        self.save(
                            os.path.join(
                                self.save_path,
                                "{}_{}_best_{}_fold_{}.pt".format(
                                    self.date,
                                    self.config.model_name,
                                    self.config.monitored_result,
                                    fold,
                                ),
                            )
                        )
            """
            Usually, we should call scheduler.step() after the end of each epoch. In particular, we need to take note that
            ReduceLROnPlateau needs to step(monitered_metrics) because of the mode argument.
            """
            if self.config.val_step_scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(
                        val_results_computed[self.config.monitored_result]
                    )
                else:
                    self.scheduler.step()

            """End of training, epoch + 1 so that self.epoch can be updated."""
            self.epoch += 1

        """
        curr_fold_best_checkpoint: We have ended the training for the current fold/model after the configured number of 
        epochs. We then load the best weight we have saved earlier in the training. The reason is that we want to use 
        this checkpoint later to extract the predictions (OOF predictions).
        """
        curr_fold_best_checkpoint = self.load(
            os.path.join(
                self.save_path,
                "{}_{}_best_{}_fold_{}.pt".format(
                    self.date,
                    self.config.model_name,
                    self.config.monitored_result,
                    fold,
                ),
            )
        )
        return curr_fold_best_checkpoint

    def train_one_epoch(self, train_loader):
        """Train one epoch of the model."""
        # set to train mode
        self.model.train()

        return self.training_results.compute_results(train_loader)

    def valid_one_epoch(self, val_loader):
        """Validate one training epoch."""
        # set to eval mode
        self.model.eval()
        # print("len", len(val_loader))
        # it is returning the method in Results class which takes in a loader
        return self.validation_results.compute_results(val_loader)

    def save_model(self, path):
        """Save the trained model."""
        self.model.eval()
        torch.save(self.model.state_dict(), path)

    def save(self, path):
        """Save the weight for the best evaluation loss (and monitored metrics) with corresponding OOF predictions.
        OOF predictions for each fold is merely the best score for that fold."""
        self.model.eval()

        best_results = {
            "best_{}".format(best_result): value
            for (best_result, value) in self.best_val_results.items()
        }

        # print(best_results)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": self.epoch,
                **best_results,
                **self.saved_val_results,
                "loss_history": self.loss_history,
                self.config.monitored_result: self.monitored_metrics_history,
            },
            path,
        )

    def load(self, path):
        """Load a model checkpoint from the given path."""
        checkpoint = torch.load(path)
        return checkpoint

    def log(self, message):
        """Log a message."""
        if self.config.verbose:
            print(message)
        with open(self.config.paths["log_path"], "a+") as logger:
            logger.write(f"{message}\n")


def train_on_fold(df_folds: pd.DataFrame, config, fold: int):
    """Train the model on the given fold."""
    model = CustomModel(config=config, pretrained=True)
    model.to(config.device)

    augmentations_class = getattr(transforms, config.augmentations_class)

    transforms_train = augmentations_class.from_config(
        config.augmentations_train[config.augmentations_class]
    )
    transforms_val = augmentations_class.from_config(
        config.augmentations_val[config.augmentations_class]
    )

    train_df = df_folds[df_folds["fold"] != fold].reset_index(drop=True)
    val_df = df_folds[df_folds["fold"] == fold].reset_index(drop=True)
    val_df.to_csv("val_df.csv")
    # print(len(val_df))
    data_dict = {
        "dataset_train_dict": {
            "df": train_df,
            "transforms": transforms_train,
            "transform_norm": True,
            "meta_features": None,
            "mode": "train",
        },
        "dataset_val_dict": {
            "df": val_df,
            "transforms": transforms_val,
            "transform_norm": True,
            "meta_features": None,
            "mode": "train",
        },
        "dataloader_train_dict": {
            "batch_size": config.batch_size,
            "shuffle": True,
            "num_workers": config.num_workers,
            "worker_init_fn": seed_worker,
            "pin_memory": True,
        },
        "dataloader_val_dict": {
            "batch_size": config.batch_size,
            "shuffle": False,
            "num_workers": config.num_workers,
            "worker_init_fn": seed_worker,
            "pin_memory": True,
        },
    }

    # returns {"Train": train_loader, "Validation": val_loader}
    dataloader_dict = CustomDataLoader(config=config, data_dict=data_dict).get_loaders()
    train_loader, val_loader = dataloader_dict["Train"], dataloader_dict["Validation"]
    hongnan_classifier = Trainer(model=model, config=config)

    curr_fold_best_checkpoint = hongnan_classifier.fit(train_loader, val_loader, fold)
    # print(len(curr_fold_best_checkpoint["oof_preds"]))
    val_df[[str(c) for c in range(config.num_classes)]] = curr_fold_best_checkpoint[
        "oof_preds"
    ]
    # val_df["preds"] = curr_fold_best_checkpoint["oof_preds"].argmax(1)

    return val_df


### Ian: Might need some help cleaning up this part, especially when my monitored metric is roc for example,
### then it will be good to call get_oof_roc in the train_loop instead of hard coding it.


def train_loop(
    df_folds: pd.DataFrame, config, fold_num: int = None, train_one_fold=False
):
    """Perform the training loop on all folds. Here The CV score is the average of the validation fold metric.
    While the OOF score is the aggregation of all validation folds."""

    cv_score_list = []
    oof_df = pd.DataFrame()
    if train_one_fold:
        _oof_df = train_on_fold(df_folds=df_folds, config=config, fold=fold_num)
        # curr_fold_best_score = get_oof_roc(config, _oof_df)
        # print("Fold {} OOF Score is {}".format(fold_num, curr_fold_best_score))
    else:
        """The below for loop code guarantees fold starts from 1 and not 0. https://stackoverflow.com/questions/33282444/pythonic-way-to-iterate-through-a-range-starting-at-1"""
        for fold in (number + 1 for number in range(config.num_folds)):
            _oof_df = train_on_fold(df_folds=df_folds, config=config, fold=fold)
            oof_df = pd.concat([oof_df, _oof_df])
            curr_fold_best_score_dict, curr_fold_best_score = get_oof_roc(
                config, _oof_df
            )
            cv_score_list.append(curr_fold_best_score)
            print(
                "\n\n\nOOF Score for Fold {}: {}\n\n\n".format(
                    fold, curr_fold_best_score
                )
            )

        print("CV score", np.mean(cv_score_list))
        print("Variance", np.var(cv_score_list))
        print("Five Folds OOF", get_oof_roc(config, oof_df))
        oof_df.to_csv(os.path.join(config.paths["save_path"], "oof.csv"))


if __name__ == "__main__":
    colab = True
    COMPETITIONS = ["MELANOMA", "CASSAVA", "RANZCR"]
    if colab is True:
        # uncomment this if you do not create new folder, else create mkdir reighns
        # yaml_config = YAMLConfig("/content/Pytorch-Pipeline/config.yaml")

        yaml_config = YAMLConfig("/content/reighns/config_RANZCR.yaml")
        # yaml_config.paths[
        #     "log_path"
        # ] = "/content/drive/My Drive/Melanoma/weights/tf_effnet_b2_ns/5th-Mar-V1/log.txt"
        # yaml_config.paths["train_path"] = "/content/train/"
        # yaml_config.paths[
        #     "csv_path"
        # ] = "/content/drive/My Drive/Melanoma/siim-isic-melanoma-classification/train.csv"
        # yaml_config.paths[
        #     "save_path"
        # ] = "/content/drive/My Drive/Melanoma/weights/tf_effnet_b2_ns/5th-Mar-V1"
        # yaml_config.paths[
        #     "model_weight_path_folder"
        # ] = "/content/drive/My Drive/pretrained-weights/pretrained-effnet-weights"
        # yaml_config.num_workers = 4
        # yaml_config.batch_size = 32
        # yaml_config.debug = False
    else:
        yaml_config = YAMLConfig("./config.yaml")

    seed_all(seed=yaml_config.seed)
    train_csv = pd.read_csv(yaml_config.paths["csv_path"])

    df_folds = make_folds(train_csv, yaml_config)
    if yaml_config.debug:
        df_folds = df_folds.sample(frac=0.05)
        yaml_config.batch_size = 4
        # print(df_folds.groupby(["fold", yaml_config.class_col_name]).size())
        # print(len(df_folds))

        train_all_folds = train_loop(
            df_folds=df_folds, config=yaml_config, fold_num=1, train_one_fold=True
        )
    else:
        train_all_folds = train_loop(df_folds=df_folds, config=yaml_config)
