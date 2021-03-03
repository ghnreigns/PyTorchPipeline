import pandas as pd
from IPython.display import display
from sklearn.model_selection import GroupKFold, StratifiedKFold

### Problem 4: I am thinking we can add just one flag in config.yaml so that we can choose whether to use skf or gkf. 
### Something like make_folds(train_csv: pd.DataFrame, config) -> pd.DataFrame: if config.cv_schema== 'skf' etc etc

def make_folds(train_csv: pd.DataFrame, config, cv_schema=None, use_skf=True, use_gkf=False) -> pd.DataFrame:
    """Split the given dataframe into training folds."""
    # TODO: add options for cv_scheme as it is cumbersome here.
    if use_skf:
        df_folds = train_csv.copy()
        skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X=df_folds[config.image_col_name], y=df_folds[config.class_col_name])
        ):
            df_folds.loc[val_idx, "fold"] = int(fold + 1)
        df_folds["fold"] = df_folds["fold"].astype(int)
        print(df_folds.groupby(["fold", config.class_col_name]).size())

    elif use_gkf:
        df_folds = train_csv.copy()
        gkf = GroupKFold(n_splits=config.num_folds)
        groups = df_folds[config.group_kfold_split].values
        for fold, (train_index, val_index) in enumerate(
            gkf.split(X=df_folds, y=df_folds[config.class_col_name], groups=groups)
        ):
            df_folds.loc[val_index, "fold"] = int(fold + 1)
        df_folds["fold"] = df_folds["fold"].astype(int)
        try:
            print(df_folds.groupby(["fold", config.class_col_name]).size())
        except:
            display(df_folds)

    return df_folds
