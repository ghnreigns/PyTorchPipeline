### Problem 5: If one uses custom loss functions like LabelSmoothingLoss, for example, then it is necessarily to use two different loss, one for train
### one for val. This is because while training, the custom loss function will be a deciding factor to optimize(minimization in fact) the loss. However,
### during validation mode, we will use back our normal loss function to obtain a more accurate result of what we would expect when predicting on an unseen test.
### For now, I hardcoded this in config to be something like:     criterion_train = 'LabelSmoothingLoss' and criterion_val = 'CrossEntropyLoss' and call
# self.criterion_train = LabelSmoothingLoss(**config.criterion_params[config.criterion]).to(self.config.device)
# self.criterion_val = getattr(torch.nn, config.criterion_val)(**config.criterion_params[config.criterion_val])
### Subsequently, changing the above attribute accordingly in train and val.

"""A configurable system for computing model training and validation results."""
import abc
from enum import Enum
import inspect
import time

import networkx as nx
import numpy as np
import sklearn
import torch

import metrics


class Mode(Enum):
    TRAINING = "training"
    VALIDATION = "validation"


class PerStepResult(abc.ABC):
    """A result computed at each step with no overall summary."""

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        """Compute the result value at the current step.

        Named parameters to this function in implementing classes
        refer to other results that the implementing class relies
        on. These results will be computed first, and the result of
        computing them will be supplied to the implementing class as
        arguments.

        Implementing classes must provide a step() method that accepts
        **kwargs in addition to the named parameters. Otherwise, the
        automated dependency injection in Results will not work.
        """

    @abc.abstractmethod
    def reset(self):
        """Reset the result for computing in a new epoch."""


class Result(abc.ABC):
    """A result computed at each step with an overall summary."""

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        """Compute the result value at the current step.

        Named parameters to this function in implementing classes
        refer to other results that the implementing class relies
        on. These results will be computed first, and the result of
        computing them will be supplied to the implementing class as
        arguments.

        Implementing classes must provide a step() method that accepts
        **kwargs in addition to the named parameters. Otherwise, the
        automated dependency injection in Results will not
        work.
        """

    @abc.abstractmethod
    def compute(self, *args, **kwargs):
        """Compute the summary value for this result.

        Named parameters to this function in implementing classes
        refer to other results that the implementing class relies
        on. These results will be computed first, and the result of
        computing them will be supplied to the implementing class as
        arguments.

        Implementing classes must provide a step() method that accepts
        **kwargs in addition to the named parameters. Otherwise, the
        automated dependency injection in Results will not
        work.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset the result for computing in a new epoch."""


class PerStepReportableResult(abc.ABC):
    """A result whose per-step computations may be reported."""

    @abc.abstractmethod
    def report_step(self, step_value):
        """Get the current step value for the result as a string."""


class ReportableResult(abc.ABC):
    """A result whose summary value may be reported."""

    @abc.abstractmethod
    def report(self, computed_value):
        """Get the summary value for the result as a string."""


class ComparableResult(abc.ABC):
    """A result whose summary values may be compared."""

    @abc.abstractmethod
    def compare(self, old_value, new_value):
        """Determine whether the new_value is better than the old_value."""


class SavableResult(abc.ABC):
    """A result whose value may be saved when the model is saved."""

    @abc.abstractmethod
    def get_save_name(self, computed_value):
        """Get the name this result should be saved under in the model dict.

        If the computed_value should not be saved in the model dict, this
        function should return None.
        """


class average_loss(Result, PerStepReportableResult, ReportableResult, ComparableResult):
    """A result for computing average loss."""

    def __init__(self):
        self.curr_batch_avg_loss = 0
        self.avg = 0
        self.running_total_loss = 0
        self.count = 0

    def step(self, loss, batch_size, **kwargs):
        self.curr_batch_avg_loss = loss
        self.running_total_loss += loss * batch_size
        self.count += batch_size
        self.avg = self.running_total_loss / self.count
        return self.avg

    def compute(self, **kwargs):
        return self.avg

    def reset(self):
        self.curr_batch_avg_loss = 0
        self.avg = 0
        self.running_total_loss = 0
        self.count = 0

    def report_step(self, step_value):
        return "summary_loss: {:.3f}".format(step_value)

    def report(self, computed_value):
        return "Avg Validation Summary Loss: {:.6f}".format(computed_value)

    def compare(self, old_value, new_value):
        """@Ian, since here returns a tensor, I will use tensor comparison"""
        # quick hack to bypass inappropriate comparison between none type and tensor
        if old_value is None:
            old_value = torch.as_tensor(np.inf)
        return torch.gt(old_value, new_value)


class average_accuracy(Result, PerStepReportableResult, ReportableResult, ComparableResult):
    """A result for computing average prediction accuracy."""

    def __init__(self):
        self.score = 0
        self.count = 0
        self.sum = 0

    def step(self, y_true, y_preds, batch_size, **kwargs):

        # so we just need to count total num of images / batch_size
        # self.count += num_steps
        self.count += batch_size
        # this part here already got an acc score for the 4 images, so
        # no need divide batch size
        self.score = sklearn.metrics.accuracy_score(y_true, y_preds)
        total_score = self.score * batch_size

        self.sum += total_score

        return self.sum / self.count

    def compute(self, **kwargs):
        return self.sum / self.count

    def reset(self):
        self.score = 0
        self.count = 0
        self.sum = 0

    def report_step(self, step_value):
        return "acc: {:.3f}".format(step_value)

    def report(self, computed_value):
        return "Validation Accuracy: {:.6f}".format(computed_value)

    def compare(self, old_value, new_value):
        return new_value > old_value


class val_preds_roc_array(Result):
    def __init__(self):
        self.roc_list = []

    def step(self, softmax_preds, **kwargs):
        self.roc_list.append(softmax_preds[:, 1])

    def compute(self, **kwargs):
        return np.concatenate(self.roc_list, axis=0)

    def reset(self):
        self.roc_list = []


class val_preds_softmax_array(Result, SavableResult):
    def __init__(self):
        self.softmax_list = []

    def step(self, softmax_preds, **kwargs):
        self.softmax_list.append(softmax_preds)

    def compute(self, **kwargs):
        return np.concatenate(self.softmax_list, axis=0)

    def reset(self):
        self.softmax_list = []

    def get_save_name(self, _computed_value):
        return "oof_preds"


class val_preds_argmax_array(Result):
    def __init__(self):
        self.argmax_list = []

    def step(self, y_preds, **kwargs):
        self.argmax_list.append(y_preds)

    def compute(self, **kwargs):
        return np.concatenate(self.argmax_list, axis=0)

    def reset(self):
        self.argmax_list = []


class val_gt_label_array(Result):
    def __init__(self):
        self.gt_label_list = []

    def step(self, y_true, **kwargs):
        self.gt_label_list.append(y_true)

    def compute(self, **kwargs):
        return np.concatenate(self.gt_label_list, axis=0)

    def reset(self):
        self.gt_label_list = []


"""Note that val_roc_auc_score should be the same as multi_class_roc_auc_score if we are training on binary."""


class val_roc_auc_score(Result, ReportableResult, ComparableResult):
    """A result for computing the validation ROC score."""

    def step(self, **kwargs):
        pass

    def compute(self, val_gt_label_array, val_preds_roc_array, **kwargs):
        return sklearn.metrics.roc_auc_score(y_true=val_gt_label_array, y_score=val_preds_roc_array)

    def reset(self):
        pass

    def report(self, computed_value):
        return "Validation ROC: {:.6f}".format(computed_value)

    def compare(self, old_value, new_value):
        return new_value > old_value


class multi_class_roc_auc_score(Result, ReportableResult, ComparableResult):
    """A result for computing the multi-class validation ROC score."""

    def step(self, **kwargs):
        pass

    def compute(self, val_gt_label_array, val_preds_softmax_array, config, **kwargs):
        score, _ = metrics.multiclass_roc(
            y_true=val_gt_label_array, y_preds_softmax_array=val_preds_softmax_array, config=config
        )

        return score

    def reset(self):
        pass

    def report(self, computed_value):
        return "MultiClass ROC: {}".format(computed_value)

    def compare(self, old_value, new_value):
        return new_value > old_value


class y_true(PerStepResult):
    def step(self, labels, **kwargs):
        return labels.cpu().numpy()

    def reset(self):
        pass


class softmax_preds(PerStepResult):
    def step(self, logits, mode, **kwargs):

        if mode == Mode.VALIDATION:
            return torch.nn.Softmax(dim=1)(input=logits).to("cpu").numpy()
        else:
            return torch.nn.Softmax(dim=1)(input=logits).to("cpu").detach().numpy()

    def reset(self):
        pass


class y_preds(PerStepResult):
    def step(self, softmax_preds, **kwargs):
        return np.argmax(a=softmax_preds, axis=1)

    def reset(self):
        pass


def get_function_param_names(func):
    """Get the positional and keyword parameter names of a function."""
    return [
        param.name
        for param in inspect.signature(func).parameters.values()
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]


# Using globals() bit of a hack that allows us to dynamically generate
# a dictionary of all available results even if this code is run in a
# Jupyter Notebook or similar interactive Python environment rather
# than as a module in the filesystem.
_results = {
    name: result
    for (name, result) in globals().items()
    if (inspect.isclass(result) and not inspect.isabstract(result) and issubclass(result, (Result, PerStepResult)))
}


def construct_result(name, config):
    """Construct a result with the given name and configuration."""
    return _results[name](**config.result_params.get(name, {}))


class Results(abc.ABC):
    """A class for performing validation or training with selected results.

    Results have two computation phases. First, during each
    step the step() method of all selected results is called to
    produce the per-step value for each result. Then, after completing
    all validation steps, the compute() method of the selected
    _summary results_ is invoked to produce the final summary value
    for the selected results.

    The internal state of each result is reset at the beginning of
    validation via the reset() method. During validation, summary
    results may update their internal state in the step() method to
    aid in the computation of the final summary value in compute().

    In both step() and compute(), results may rely on the per-step and
    computed values of other results, respectively. We use automated
    dependency injection to determine the order in which the selected
    results and their dependencies must be computed. The named
    parameters of step() and compute() refer to other classes
    implementing Result and PerStepResult. We use this to build a
    dependency graph that is topologically sorted to determine the
    computation order in the per-step and summarizing phases. This
    design is modular and extensible, and inspired by the PyTest
    implementation of fixtures (see
    https://docs.pytest.org/en/reorganize-docs/fixture.html#fixtures-as-function-arguments
    for more information).

    If a selected result relies on a result that is not selected in
    step() or compute(), the result will automatically be constructed
    and computed for you. Only the results of the _selected summary
    results_ will be returned when results are computed.
    """

    def __init__(
        self,
        results_type,
        computation_context,
        built_in_dependencies_per_step,
        built_in_dependencies_summary,
        trainer,
        results,
        config,
    ):
        """Construct a new Results instance.

        All selected results in `results` _must_ be summary results (i.e.,
        implement the Result class), not per-step results.
        """
        self.results_type = results_type
        self.computation_context = computation_context
        self.trainer = trainer
        self.results = results

        # All metrics that must be computed, not just selected
        # ones. When we are finished calling
        # compute_computation_order, this list will contain all of the
        # selected metrics and their dependencies.
        self.used_results = [*results]

        self.config = config

        self.trainer.log("Solving {} results computation order...".format(results_type))

        def compute_computation_order(classes, func_from_class, built_in_dependencies):
            """Determine the computation order of results for a phase.

            :param classes:
                The selected result class instances to compute
            :param func_from_class:
                A function that, given a result class, returns the function
                to be called for the current phase that we need to get
                dependencies from

                e.g., lambda result: result.step
            :param built_in_dependencies:
                A set of string names for dependencies that will be
                pre-computed and available to all results during this phase

            :returns A list of results in the order they need to be computed
                     for the phase.
            """
            # We use a directed graph for the dependency graph.
            # Nodes are results, and edges show dependencies between them.
            g = nx.DiGraph()
            # A set of metrics that we have or will compute
            # dependencies for.  Used to determine when we need to
            # recursively compute dependencies.
            used_results = set(classes)

            def compute_dependencies(result):
                """Compute the dependencies for a result."""

                name = result.__class__.__name__
                # The result function (e.g., step or compute) for the phase
                # we are considering
                func = func_from_class(result)

                # Result dependencies are the function arguments minus
                # the pre-computed dependencies.
                deps = set(get_function_param_names(func)) - built_in_dependencies

                # If we've come across dependencies that hasn't been
                # selected by the user, we need to add it to the
                # dependencies _and_ compute its dependencies as well.
                uncomputed_deps = deps - used_results

                for dep in uncomputed_deps:
                    new_result = construct_result(dep, config)
                    used_results.add(new_result)
                    self.used_results.append(new_result)

                    compute_dependencies(new_result)

                # We explicitly add the result to the dependency graph
                # as a node in case no other results rely on
                # it. Explicitly adding it ensures that it will appear
                # somewhere in the topologically sorted listed of
                # nodes.
                g.add_node(name)

                for dep in deps:
                    # The directedness of the edges is important! We want
                    # dependencies to be computed first, so the edge must flow
                    # from dependency to current result.
                    g.add_edge(dep, name)

            for result in classes:
                compute_dependencies(result)

            # Turn the set of all needed results into a map for easily
            # fetching the needed results by name.
            used_results = {result.__class__.__name__: result for result in used_results}

            # Get the order the results must be computed in
            solved_order = nx.algorithms.dag.topological_sort(g)

            return [used_results[result] for result in solved_order]

        # The order here is important! We must solve the computation
        # order of the selected summary results _first_, because we
        # will likely end up needing to compute other summary results
        # as dependencies of these. These dependency summary results
        # may have additional dependencies in the per-step phase,
        # which must be properly accounted for.
        self.summary_results = compute_computation_order(
            results, lambda result: result.compute, built_in_dependencies_summary
        )

        self.per_step_results = compute_computation_order(
            self.summary_results, lambda result: result.step, built_in_dependencies_per_step
        )

        self.trainer.log(
            "Per-Step Results: {}".format(", ".join([metric.__class__.__name__ for metric in self.per_step_results]))
        )
        self.trainer.log(
            "Summary Results: {}".format(", ".join([metric.__class__.__name__ for metric in self.summary_results]))
        )
        self.trainer.log(
            "Selected Results: {}".format(", ".join([metric.__class__.__name__ for metric in self.results]))
        )

    @abc.abstractmethod
    def compute_built_in_per_step_results(self, step, image_ids, images, labels):
        """Compute built-in results for a step."""

    @abc.abstractmethod
    def compute_built_in_summary_results(self):
        """Compute built-in results for the summary phase."""

    def compute_results(self, loader):
        """Compute the selected results from the given loader."""

        for result in self.used_results:
            result.reset()

        start_time = time.time()

        with self.computation_context():
            for step, (image_ids, images, labels) in enumerate(loader):

                # Compute all the pre-computed dependencies available
                # to all results
                step_computed = self.compute_built_in_per_step_results(step, image_ids, images, labels)

                for result in self.per_step_results:

                    # This is why all step() methods must accept **kwargs,
                    # because it allows us to be a bit lazy here with the
                    # parameter passing. If we didn't accept **kwargs in
                    # step(), we would have to select only the step results
                    # explicitly asked for in step(), or we would get an
                    # invalid keyword argument exception.
                    step_computed[result.__class__.__name__] = result.step(**step_computed)

                if self.config.verbose and step % self.config.verbose_step == 0:
                    end_time = time.time()
                    reported_computed = [
                        result.report_step(step_computed[result.__class__.__name__])
                        for result in self.per_step_results
                        if isinstance(result, PerStepReportableResult)
                    ]

                    results_str = ", ".join(
                        [
                            "{} steps: {} / {}".format(self.results_type, step, len(loader)),
                            *reported_computed,
                            "time: {:.3f}".format(end_time - start_time),
                        ]
                    )

                    print(results_str, end="\r")

        # Compute the pre-computed dependencies available to all metrics
        summary_computed = self.compute_built_in_summary_results()

        # Now compute the summary phase of all summary results
        for result in self.summary_results:
            summary_computed[result.__class__.__name__] = result.compute(**summary_computed)

        return {result.__class__.__name__: summary_computed[result.__class__.__name__] for result in self.results}


class TrainingResults(Results):
    """A class for performing model training with selected results."""

    def __init__(self, trainer, results, config):
        super().__init__(
            "training",
            EmptyContextManager,
            {"images", "labels", "batch_size", "logits", "loss", "config", "mode"},
            {"config", "mode"},
            trainer,
            results,
            config,
        )

    def compute_built_in_per_step_results(self, step, _image_ids, images, labels):
        images = images.to(self.config.device)
        labels = labels.to(self.config.device)
        batch_size = images.shape[0]

        """using amp, https://pytorch.org/docs/stable/notes/amp_examples.html FYI Ian."""
        if config.use_amp:
            """I would think clearing gradients here is the correct way, as opposed to calling it last."""
            self.trainer.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = self.trainer.model(images)
                loss = self.trainer.criterion(input=logits, target=labels)
            loss_value = loss.item()
            self.trainer.scaler.scale(loss).backward()
            self.trainer.scaler.step(self.trainer.optimizer)
            self.trainer.scaler.update()

        else:
            logits = self.trainer.model(images)
            loss = self.trainer.criterion(input=logits, target=labels)
            loss_value = loss.item()
            self.trainer.optimizer.zero_grad()
            loss.backward()
            self.trainer.optimizer.step()

        return {
            "images": images,
            "labels": labels,
            "batch_size": batch_size,
            "logits": logits,
            "loss": loss_value,
            "config": self.config,
            "mode": Mode.TRAINING,
        }

    def compute_built_in_summary_results(self):
        return {"config": self.config, "mode": Mode.TRAINING}


class ValidationResults(Results):
    """A class for performing model validation with selected results."""

    def __init__(self, trainer, results, config):
        super().__init__(
            "validation",
            torch.no_grad,
            {"images", "labels", "batch_size", "logits", "loss", "config", "mode"},
            {"config", "mode"},
            trainer,
            results,
            config,
        )

    def compute_built_in_per_step_results(self, step, _image_ids, images, labels):

        images = images.to(self.config.device)
        labels = labels.to(self.config.device)
        logits = self.trainer.model(images)
        loss = self.trainer.criterion(input=logits, target=labels)

        return {
            "images": images,
            "labels": labels,
            "batch_size": images.shape[0],
            "logits": logits,
            "loss": loss,
            "config": self.config,
            "mode": Mode.VALIDATION,
        }

    def compute_built_in_summary_results(self):
        return {"config": self.config, "mode": Mode.VALIDATION}


class EmptyContextManager:
    """A context manager that does nothing."""

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        pass
