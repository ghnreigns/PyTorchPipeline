import numpy as np


"""
With reference to the Cassava Competition, the total number of images is 21397; as we split the dataset into
5-folds, the validation fold will have on average 21397/5=4280 (rounded) images. Thus, to simulate the OOF predictions, we
create 5 separate val_dfs because we want to get the aggregate predictions over all validation folds. Note that the
Union of the 5 val_dfs will be equal to the whole dataset. That is Union(len(val_df_fold_{}.format(fold_num)) = len(df_folds)
"""

"""
The below is an example for multiclass-classification, this may not work for multilabel-classification.
"""


def oof_visualization(df_folds):
    for fold_num in df_folds:
        val_df_fold = df_folds[df_folds["fold"] == 1].reset_index(drop=True)
        # 5 classes here since it is Cassava.
        dummy_preds_random = np.random_rand(len(val_df_fold), 5)
        print(
            "Len of val_df_fold is {} and shape of dummy_preds_random is {}".format(
                len(val_df_fold), dummy_preds_random.shape
            )
        )
        # val_df_fold
        oof_df = pd.concat([oof_df, _oof_df])