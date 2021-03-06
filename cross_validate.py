import pandas as pd
from IPython.display import display
from sklearn.model_selection import GroupKFold, StratifiedKFold

### Problem 4: I am thinking we can add just one flag in config.yaml so that we can choose whether to use skf or gkf.
### Something like make_folds(train_csv: pd.DataFrame, config) -> pd.DataFrame: if config.cv_schema== 'skf' etc etc


def make_folds(train_csv: pd.DataFrame, config) -> pd.DataFrame:
    """Split the given dataframe into training folds."""
    # TODO: add options for cv_scheme as it is cumbersome here.
    if config.cv_schema == "StratifiedKFold":
        df_folds = train_csv.copy()
        skf = StratifiedKFold(
            n_splits=config.num_folds, shuffle=True, random_state=config.seed
        )

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(
                X=df_folds[config.image_col_name], y=df_folds[config.class_col_name]
            )
        ):
            df_folds.loc[val_idx, "fold"] = int(fold + 1)
        df_folds["fold"] = df_folds["fold"].astype(int)
        print(df_folds.groupby(["fold", config.class_col_name]).size())

    elif config.cv_schema == "GroupKfold":
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

    else:  # No CV Schema used in this file, but custom one
        df_folds = train_csv.copy()
        try:
            print(df_folds.groupby(["fold", config.class_col_name]).size())
        except:
            display(df_folds)

    return df_folds


class RepeatedStratifiedGroupKFold:
    def __init__(self, n_splits=5, n_repeats=1, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        k = self.n_splits

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std(
                    [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
                )
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        rnd = check_random_state(self.random_state)
        for repeat in range(self.n_repeats):
            labels_num = np.max(y) + 1
            y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
            y_distr = Counter()
            for label, g in zip(y, groups):
                y_counts_per_group[g][label] += 1
                y_distr[label] += 1

            y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
            groups_per_fold = defaultdict(set)

            groups_and_y_counts = list(y_counts_per_group.items())
            rnd.shuffle(groups_and_y_counts)

            for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
                best_fold = None
                min_eval = None
                for i in range(k):
                    fold_eval = eval_y_counts_per_fold(y_counts, i)
                    if min_eval is None or fold_eval < min_eval:
                        min_eval = fold_eval
                        best_fold = i
                y_counts_per_fold[best_fold] += y_counts
                groups_per_fold[best_fold].add(g)

            all_groups = set(groups)
            for i in range(k):
                train_groups = all_groups - groups_per_fold[i]
                test_groups = groups_per_fold[i]

                train_indices = [i for i, g in enumerate(groups) if g in train_groups]
                test_indices = [i for i, g in enumerate(groups) if g in test_groups]

                yield train_indices, test_indices