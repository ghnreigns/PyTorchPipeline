import numpy as np
import sklearn


def multiclass_roc(y_true, y_preds_softmax_array, config):
    # Review Comments:
    # You never use this variable.
    label_dict = dict()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    roc_scores = []
    for label_num in range(len(config.class_list)):

        # get y_true_multilabel binarized version for each loop (end of each epoch)
        y_true_multiclass_array = sklearn.preprocessing.label_binarize(
            y_true, classes=config.class_list)
        y_true_for_curr_class = y_true_multiclass_array[:, label_num]
        y_preds_for_curr_class = y_preds_softmax_array[:, label_num]
        # calculate fpr,tpr and thresholds across various decision thresholds
        # pos_label = 1 because one hot encode guarantees it
        fpr[label_num], tpr[label_num], _ = sklearn.metrics.roc_curve(
            y_true=y_true_for_curr_class,
            y_score=y_preds_for_curr_class,
            pos_label=1)
        roc_auc[label_num] = sklearn.metrics.auc(fpr[label_num],
                                                 tpr[label_num])
        roc_scores.append(roc_auc[label_num])
        # if binary class, the one hot encode will (n_samples,1) and therefore will only need to slice [:,0] ONLY.
        # that is why usually for binary class, we do not need to use this piece of code, just for testing purposes.
        # However, it will now treat our 0 (negative class) as positive, hence returning the roc for 0, in which case
        # to get both 0 and 1, you just need to use 1-roc(0)value
        if config.num_classes == 2:
            roc_auc[config.class_list[1]] = 1 - roc_auc[label_num]
            return roc_auc, roc_scores
            # Review Comments:
            # No need to break after return
            break
    avg_roc_score = np.mean(roc_scores)
    return roc_auc, avg_roc_score


def get_acc_score(config, result_df):
    """Get the accuracy of model predictions."""
    preds = result_df["preds"].values
    labels = result_df[config.class_col_name].values
    score = sklearn.metrics.accuracy_score(y_true=labels, y_pred=preds)
    return score


def get_roc_score(config, result_df):
    """Get the roc score of model predictions. max_label is needed for binary classification."""
    max_label = str(np.max(result_df[config.class_col_name].values))
    preds = result_df[max_label].values
    labels = result_df[config.class_col_name].values
    score = sklearn.metrics.roc_auc_score(y_true=labels, y_score=preds)
    return score
