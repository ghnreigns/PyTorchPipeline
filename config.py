class GlobalConfig:
    seed = 1930
    num_classes = 2
    class_list = [0, 1]
    batch_size = 32
    n_epochs = 8

    # unpack the key dict
    scheduler = "StepLR"
    scheduler_params = {
        "StepLR": {"step_size": 2, "gamma": 0.3, "last_epoch": -1, "verbose": True},
        "ReduceLROnPlateau": {
            "mode": "max",
            "factor": 0.5,
            "patience": 1,
            "threshold": 0.0001,
            "threshold_mode": "rel",
            "cooldown": 0,
            "min_lr": 1e-5,
            "eps": 1e-08,
            "verbose": True,
        },
        "CosineAnnealingWarmRestarts": {"T_0": 10, "T_mult": 1, "eta_min": 0, "last_epoch": -1, "verbose": True},
    }

    # do scheduler.step after optimizer.step
    train_step_scheduler = False
    val_step_scheduler = True

    # optimizer
    optimizer = "AdamW"
    optimizer_params = {
        "AdamW": {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": 1e-6, "amsgrad": False},
        "Adam": {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": 1e-6, "amsgrad": False},
    }

    # criterion
    criterion = "CrossEntropyLoss"
    criterion_val = "CrossEntropyLoss"
    criterion_params = {
        "CrossEntropyLoss": {
            "weight": None,
            "size_average": None,
            "ignore_index": -100,
            "reduce": None,
            "reduction": "mean",
        },
        "LabelSmoothingLoss": {"classes": 2, "smoothing": 0.05, "dim": -1},
        "FocalCosineLoss": {"alpha": 1, "gamma": 2, "xent": 0.1},
    }

    image_size = 256
    resize = 256
    crop_size = {128: 110, 256: 200, 384: 320, 512: 400}
    verbose = 1
    verbose_step = 1
    num_folds = 5
    image_col_name = "image_name"
    class_col_name = "target"
    paths = {
        "train_path": "/content/train/",
        "test_path": "../input/siim-isic-melanoma-classification/jpeg/test",
        "csv_path": "/content/drive/My Drive/KAGGLE-MELANOMA/siim-isic-melanoma-classification/train.csv",
        "log_path": "./log.txt",
        "save_path": "/content/drive/My Drive/KAGGLE-MELANOMA/weights/tf_effnet_b2_ns",
        "model_weight_path_folder": "/content/drive/My Drive/pretrained-effnet-weights",
    }

    effnet = "tf_efficientnet_b2_ns"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
