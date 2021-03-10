import numpy as np
import sklearn

"""
multiclass_roc is macro_multilabel_auc
def macro_multilabel_auc(label, pred):
    aucs = []
    for i in range(len(target_cols)):
        aucs.append(roc_auc_score(label[:, i], pred[:, i]))
    print(np.round(aucs, 4))
    return np.mean(aucs)
"""


def multiclass_roc(y_true, y_preds_softmax_array, config):
    # print("y_true", y_true)
    # print("y_preds", y_preds_softmax_array)
    label_dict, fpr, tpr, roc_auc, roc_scores = dict(), dict(), dict(), dict(), []
    for label_num in range(len(config.class_list)):
        """
        Get y_true_multilabel binarized version for each loop (end of each epoch).
        """
        y_true_multiclass_array = sklearn.preprocessing.label_binarize(
            y_true, classes=config.class_list
        )
        y_true_for_curr_class = y_true_multiclass_array[:, label_num]
        y_preds_for_curr_class = y_preds_softmax_array[:, label_num]
        """
        Calculate fpr,tpr and thresholds across various decision thresholds pos_label = 1 because one hot encode guarantees it.
        """

        fpr[label_num], tpr[label_num], _ = sklearn.metrics.roc_curve(
            y_true=y_true_for_curr_class, y_score=y_preds_for_curr_class, pos_label=1
        )
        roc_auc[label_num] = sklearn.metrics.auc(fpr[label_num], tpr[label_num])
        roc_scores.append(roc_auc[label_num])
        """
        If binary class, the one hot encode will (n_samples,1) and therefore will only need to slice [:,0] ONLY.
        This is why usually for binary class, we do not need to use this piece of code, just for testing purposes.
        However, it will now treat our 0 (negative class) as positive, hence returning the roc for 0,
        in which case to get both 0 and 1, you just need to use 1-roc[0] value.
        """

        if config.num_classes == 2:
            roc_auc[config.class_list[1]] = 1 - roc_auc[label_num]
            return roc_auc, roc_auc[1]
    """
    The point of avg_roc_score is if there are 11 classes (refer to RANZCR competition), then we may also want to 
    average out the 11 roc scores for each image.
    """
    avg_roc_score = np.mean(roc_scores, axis=None)

    def macro_multilabel_auc(y_true, y_preds_softmax_array):
        aucs = []
        for i in range(len(target_cols)):
            aucs.append(
                sklearn.metrics.roc_auc_score(y_true[:, i], y_preds_softmax_array[:, i])
            )
        macro_mean_auc = np.mean(aucs)
        print("macro multi label mean auc", macro_mean_auc)
        return macro_mean_auc

    return roc_auc, avg_roc_score
