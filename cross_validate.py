def make_folds(train_csv: pd.DataFrame, config, cv_schema=None, use_skf=True, use_gkf=False) -> pd.DataFrame:
    """Split the given dataframe into training folds."""
    # TODO: add options for cv_scheme as it is cumbersome here.
    if use_skf:
        df_folds = train_csv.copy()
        skf = StratifiedKFold(5, shuffle=True, random_state=config.seed)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X=df_folds[config.image_col_name], y=df_folds[config.class_col_name])
        ):
            df_folds.loc[val_idx, "fold"] = int(fold + 1)
        df_folds["fold"] = df_folds["fold"].astype(int)
        print(df_folds.groupby(["fold", config.class_col_name]).size())

    elif use_gkf:
        df_folds = train_csv.copy()
        gkf = GroupKFold(n_splits=CFG.n_fold)
        groups = folds["PatientID"].values
        for n, (train_index, val_index) in enumerate(
            gkf.split(df_folds, df_folds[config.class_col_name], groups=df_folds["PatientID"].values)
        ):
            df_folds.loc[val_index, "fold"] = int(n)
        folds["fold"] = folds["fold"].astype(int)
        display(folds.groupby("fold").size())

    return df_folds