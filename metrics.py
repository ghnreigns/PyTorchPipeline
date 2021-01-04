"""A configurable system for model validation metrics."""
import abc
import inspect
import time

import networkx as nx
import numpy as np
import sklearn
import torch

import roc


class PerStepMetric(abc.ABC):
    """A metric computed at each step with no overall summary."""
    @abc.abstractmethod
    def step(self, *args, **kwargs):
        """Compute the metric value at the current step.

        Named parameters to this function in implementing classes
        refer to other metrics that the implementing class relies
        on. These metrics will be computed first, and the result of
        computing them will be supplied to the implementing class as
        arguments.

        Implementing classes must provide a step() method that accepts
        **kwargs in addition to the named parameters. Otherwise, the
        automated dependency injection in ValidationMetrics will not
        work.
        """

    @abc.abstractmethod
    def reset(self):
        """Reset the metric for computing in a new epoch."""


class Metric(abc.ABC):
    """A metric computed at each step with an overall summary."""
    @abc.abstractmethod
    def step(self, *args, **kwargs):
        """Compute the metric value at the current step.

        Named parameters to this function in implementing classes
        refer to other metrics that the implementing class relies
        on. These metrics will be computed first, and the result of
        computing them will be supplied to the implementing class as
        arguments.

        Implementing classes must provide a step() method that accepts
        **kwargs in addition to the named parameters. Otherwise, the
        automated dependency injection in ValidationMetrics will not
        work.
        """

    @abc.abstractmethod
    def compute(self, *args, **kwargs):
        """Compute the summary value for this metric.

        Named parameters to this function in implementing classes
        refer to other metrics that the implementing class relies
        on. These metrics will be computed first, and the result of
        computing them will be supplied to the implementing class as
        arguments.

        Implementing classes must provide a step() method that accepts
        **kwargs in addition to the named parameters. Otherwise, the
        automated dependency injection in ValidationMetrics will not
        work.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset the metric for computing in a new epoch."""


class PerStepReportableMetric(abc.ABC):
    """A metric whose per-step computations may be reported."""
    @abc.abstractmethod
    def report_step(self, step_value):
        """Get the current step value for the metric as a string."""


class ReportableMetric(abc.ABC):
    """A metric whose summary value may be reported."""
    @abc.abstractmethod
    def report(self, computed_value):
        """Get the summary value for the metric as a string."""


class ComparableMetric(abc.ABC):
    """A metric whose summary values may be compared."""
    @abc.abstractmethod
    def compare(self, old_value, new_value):
        """Determine whether the new_value is better than the old_value."""


class SavableMetric(abc.ABC):
    """A metric whose value may be saved when the model is saved."""
    @abc.abstractmethod
    def get_save_name(self, computed_value):
        """Get the name this metric should be saved under in the model dict.

        If the computed_value should not be saved in the model dict, this
        function should return None.
        """


class average_loss(Metric, PerStepReportableMetric, ReportableMetric,
                   ComparableMetric):
    """A metric for computing average loss."""
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
        return new_value < old_value


class average_accuracy(Metric, PerStepReportableMetric, ReportableMetric,
                       ComparableMetric):
    """A metric for computing average prediction accuracy."""
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


class val_preds_roc_array(Metric):
    def __init__(self):
        self.roc_list = []

    def step(self, softmax_preds, **kwargs):
        self.roc_list.append(softmax_preds[:, 1])

    def compute(self, **kwargs):
        return np.concatenate(self.roc_list, axis=0)

    def reset(self):
        self.roc_list = []


class val_preds_softmax_array(Metric, SavableMetric):
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


class val_preds_argmax_array(Metric):
    def __init__(self):
        self.argmax_list = []

    def step(self, y_preds, **kwargs):
        self.argmax_list.append(y_preds)

    def compute(self, **kwargs):
        return np.concatenate(self.argmax_list, axis=0)

    def reset(self):
        self.argmax_list = []


class val_gt_label_array(Metric):
    def __init__(self):
        self.gt_label_list = []

    def step(self, y_true, **kwargs):
        self.gt_label_list.append(y_true)

    def compute(self, **kwargs):
        return np.concatenate(self.gt_label_list, axis=0)

    def reset(self):
        self.gt_label_list = []


class val_roc_auc_score(Metric, ReportableMetric, ComparableMetric):
    """A metric for computing the validation ROC score."""
    def step(self, **kwargs):
        pass

    def compute(self, val_gt_label_array, val_preds_roc_array, **kwargs):
        return sklearn.metrics.roc_auc_score(y_true=val_gt_label_array,
                                             y_score=val_preds_roc_array)

    def reset(self):
        pass

    def report(self, computed_value):
        return "Validation ROC: {:.6f}".format(computed_value)

    def compare(self, old_value, new_value):
        return new_value > old_value


class multi_class_roc_auc_score(Metric, ReportableMetric, ComparableMetric):
    """A metric for computing the multi-class validation ROC score."""
    def step(self, **kwargs):
        pass

    def compute(self, val_gt_label_array, val_preds_softmax_array, config,
                **kwargs):
        score, _ = roc.multiclass_roc(
            y_true=val_gt_label_array,
            y_preds_softmax_array=val_preds_softmax_array,
            config=config)

        return score

    def reset(self):
        pass

    def report(self, computed_value):
        return "MultiClass ROC: {}".format(computed_value)

    def compare(self, old_value, new_value):
        return new_value > old_value


class y_true(PerStepMetric):
    def step(self, labels, **kwargs):
        return labels.cpu().numpy()

    def reset(self):
        pass


class softmax_preds(PerStepMetric):
    def step(self, logits, **kwargs):
        return torch.nn.Softmax(dim=1)(input=logits).to("cpu").numpy()

    def reset(self):
        pass


class y_preds(PerStepMetric):
    def step(self, softmax_preds, **kwargs):
        return np.argmax(a=softmax_preds, axis=1)

    def reset(self):
        pass


def get_function_param_names(func):
    """Get the positional and keyword parameter names of a function."""
    return [
        param.name for param in inspect.signature(func).parameters.values()
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]


# Using globals() bit of a hack that allows us to dynamically generate
# a dictionary of all available metrics even if this code is run in a
# Jupyter Notebook or similar interactive Python environment rather
# than as a module in the filesystem.
_metrics = {
    name: metric
    for (name, metric) in globals().items()
    if (inspect.isclass(metric) and not inspect.isabstract(metric)
        and issubclass(metric, (Metric, PerStepMetric)))
}


def construct_metric(name, config):
    """Construct a metric with the given name and configuration."""
    return _metrics[name](**config.metric_params.get(name, {}))


class ValidationMetrics:
    """A class for performing model validation with selected metrics.

    Metrics have two computation phases. First, during each validation
    step the step() method of all selected metrics is called to
    produce the per-step value for each metric. Then, after completing
    all validation steps, the compute() method of the selected
    _summary metrics_ is invoked to produce the final summary value
    for the selected metrics.

    The internal state of each metric is reset at the beginning of
    validation via the reset() method. During validation, summary
    metrics may update their internal state in the step() method to
    aid in the computation of the final summary value in compute().

    In both step() and compute(), metrics may rely on the per-step and
    computed values of other metrics, respectively. We use automated
    dependency injection to determine the order in which the selected
    metrics and their dependencies must be computed. The named
    parameters of step() and compute() refer to other classes
    implementing Metric and PerStepMetric. We use this to build a
    dependency graph that is topologically sorted to determine the
    computation order in the per-step and summarizing phases. This
    design is modular and extensible, and inspired by the PyTest
    implementation of fixtures (see
    https://docs.pytest.org/en/reorganize-docs/fixture.html#fixtures-as-function-arguments
    for more information).

    If a selected metric relies on a metric that is not selected in
    step() or compute(), the metric will automatically be constructed
    and computed for you. Only the results of the _selected summary
    metrics_ will be returned when metrics are computed.

    """
    def __init__(self, trainer, metrics, config):
        """Construct a new ValidationMetrics instance.

        All selected metrics in `metrics` _must_ be summary metrics
        (i.e., implement the Metric class), not per-step metrics.
        """
        self.trainer = trainer
        self.metrics = metrics

        # All metrics that must be computed, not just selected
        # ones. When we are finished calling
        # compute_computation_order, this list will contain all of the
        # selected metrics and their dependencies.
        self.used_metrics = [*metrics]

        self.config = config

        self.trainer.log("Solving validation metrics computation order...")

        def compute_computation_order(classes, func_from_class,
                                      built_in_dependencies):
            """Determine the computation order of metrics for a phase.

            :param classes:
                The selected metric class instances to compute
            :param func_from_class:
                A function that, given a metric class, returns the function
                to be called for the current phase that we need to get
                dependencies from
                
                e.g., lambda metric: metric.step
            :param built_in_dependencies:
                A set of string names for dependencies that will be
                pre-computed and available to all metrics during this phase

            :returns A list of metrics in the order they need to be computed
                     for the phase.            
            """

            # We use a directed graph for the dependency graph.
            # Nodes are metrics, and edges show dependencies between them.
            g = nx.DiGraph()
            # A set of metrics that we have or will compute
            # dependencies for.  Used to determine when we need to
            # recursively compute dependencies.
            used_metrics = set(classes)

            def compute_dependencies(metric):
                """Compute the dependencies for a metric."""

                name = metric.__class__.__name__
                # The metric function (e.g., step or compute) for the phase
                # we are considering
                func = func_from_class(metric)

                # Metric dependencies are the function arguments minus
                # the pre-computed dependencies.
                deps = set(
                    get_function_param_names(func)) - built_in_dependencies

                # If we've come across a dependencies that hasn't been
                # selected by the user, we need to add it to the dependencies
                # _and_ compute its dependencies as well.
                uncomputed_deps = deps - used_metrics

                for dep in uncomputed_deps:
                    new_metric = construct_metric(dep, config)
                    used_metrics.add(new_metric)
                    self.used_metrics.append(new_metric)

                    compute_dependencies(new_metric)

                # We explicitly add the metric to the dependency graph
                # as a node in case no other metrics rely on
                # it. Explicitly adding it ensures that it will appear
                # somewhere in the topologically sorted listed of
                # nodes.
                g.add_node(name)

                for dep in deps:
                    # The directedness of the edges is important! We want
                    # dependencies to be computed first, so the edge must flow
                    # from dependency to current metric.
                    g.add_edge(dep, name)

            for metric in classes:
                compute_dependencies(metric)

            # Turn the set of all needed metrics into a map for easily
            # fetching the needed metrics by name.
            used_metrics = {
                metric.__class__.__name__: metric
                for metric in used_metrics
            }

            # Get the order the metrics must be computed in
            solved_order = nx.algorithms.dag.topological_sort(g)

            return [used_metrics[metric] for metric in solved_order]

        built_in_dependencies_per_step = {
            'images', 'labels', 'batch_size', 'logits', 'loss', 'config'
        }
        built_in_dependencies_summary = {'config'}

        # The order here is important! We must solve the computation
        # order of the selected summary metrics _first_, because we
        # will likely end up needing to compute other summary metrics
        # as dependencies of these. These dependency summary metrics
        # may have additional dependencies in the per-step phase,
        # which must be properly accounted for.
        self.summary_metrics = compute_computation_order(
            metrics, lambda metric: metric.compute,
            built_in_dependencies_summary)

        self.per_step_metrics = compute_computation_order(
            self.summary_metrics, lambda metric: metric.step,
            built_in_dependencies_per_step)

        self.trainer.log("Per-Step Metrics: {}".format(", ".join(
            [metric.__class__.__name__ for metric in self.per_step_metrics])))
        self.trainer.log("Summary Metrics: {}".format(", ".join(
            [metric.__class__.__name__ for metric in self.summary_metrics])))
        self.trainer.log("Selected Summary Metrics: {}".format(", ".join(
            [metric.__class__.__name__ for metric in self.metrics])))

    def compute_metrics(self, val_loader):
        """Compute the selected metrics from the given validation set loader."""

        for metric in self.used_metrics:
            metric.reset()

        start_time = time.time()

        with torch.no_grad():
            for step, (_image_ids, images, labels) in enumerate(val_loader):

                # Compute the pre-computed dependencies available to
                # all metrics
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)
                logits = self.trainer.model(images)
                loss = self.trainer.criterion(input=logits, target=labels)

                step_results = {
                    "images": images,
                    "labels": labels,
                    "batch_size": images.shape[0],
                    "logits": logits,
                    "loss": loss,
                    "config": self.config
                }

                # Now compute the per-step phase of all metrics
                for metric in self.per_step_metrics:

                    # This is why all step() methods must accept **kwargs,
                    # because it allows us to be a bit lazy here with the
                    # parameter passing. If we didn't accept **kwargs in
                    # step(), we would have to select only the step results
                    # explicitly asked for in step(), or we would get an
                    # invalid keyword argument exception.
                    step_results[metric.__class__.__name__] = metric.step(
                        **step_results)

                if (self.config.verbose
                        and step % self.config.verbose_step == 0):
                    end_time = time.time()
                    reported_results = [
                        metric.report_step(
                            step_results[metric.__class__.__name__])
                        for metric in self.per_step_metrics
                        if isinstance(metric, PerStepReportableMetric)
                    ]

                    results_str = ", ".join([
                        "Validation Steps: {} / {}".format(
                            step, len(val_loader)), *reported_results,
                        "time: {:.3f}".format(end_time - start_time)
                    ])

                    print(results_str, end="\r")

        # Compute the pre-computed dependencies available to all metrics
        summary_results = {"config": self.config}

        # Now compute the summary phase of all summary metrics
        for metric in self.summary_metrics:
            summary_results[metric.__class__.__name__] = metric.compute(
                **summary_results)

        # Only return the values of the selected summary metrics
        return {
            metric.__class__.__name__:
            summary_results[metric.__class__.__name__]
            for metric in self.metrics
        }
