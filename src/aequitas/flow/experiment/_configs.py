# Default hyperparameter search spaces for each method
lightgbm_space = {
    "lightgbm": {
        "classpath": "aequitas.flow.methods.base_estimator.LightGBM",
        "args": {
            "boosting_type": ["dart", "gbdt"],
            "enable_bundle": [False],
            "n_estimators": {"type": "int", "range": [100, 1000]},
            "num_leaves": {"type": "int", "range": [10, 1000]},
            "min_child_samples": {
                "type": "int",
                "range": [1, 500],
                "log": True,
            },
            "learning_rate": {"type": "float", "range": [0.001, 0.1]},
            "n_jobs": [1],
        },
    },
}

undersampling_space = {
    "undersampling": {
        "classpath": "aequitas.flow.methods.preprocessing.PrevalenceSampling",
        "args": {
            "alpha": {"type": "float", "range": [0.5, 1]},
        },
    },
}

oversampling_space = {
    "oversampling": {
        "classpath": "aequitas.flow.methods.preprocessing.PrevalenceSampling",
        "args": {
            "alpha": {"type": "float", "range": [0.5, 1]},
            "strategy": ["oversample"],
        },
    },
}

fairgbm_punitive_space = {
    "fairgbm_punitive": {
        "classpath": "aequitas.flow.methods.inprocessing.fairgbm.FairGBM",
        "args": {
            "boosting_type": ["dart", "gbdt"],
            "enable_bundle": [False],
            "n_estimators": {"type": "int", "range": [100, 1000]},
            "num_leaves": {"type": "int", "range": [10, 1000]},
            "min_child_samples": {
                "type": "int",
                "range": [1, 500],
                "log": True,
            },
            "learning_rate": {"type": "float", "range": [0.001, 0.1]},
            "n_jobs": [1],
            "constraint_stepwise_proxy": ["cross_entropy"],
            "multiplier_learning_rate": {
                "type": "float",
                "range": [0.01, 0.1],
                "log": True,
            },
            "constraint_type": "fpr",
        },
    },
}

fairgbm_assistive_space = {
    "fairgbm_assistive": {
        "classpath": "aequitas.flow.methods.inprocessing.FairGBM",
        "args": {
            "boosting_type": ["dart", "gbdt"],
            "enable_bundle": [False],
            "n_estimators": {"type": "int", "range": [100, 1000]},
            "num_leaves": {"type": "int", "range": [10, 1000]},
            "min_child_samples": {
                "type": "int",
                "range": [1, 500],
                "log": True,
            },
            "learning_rate": {"type": "float", "range": [0.001, 0.1]},
            "n_jobs": [1],
            "constraint_stepwise_proxy": ["cross_entropy"],
            "multiplier_learning_rate": {
                "type": "float",
                "range": [0.01, 0.1],
                "log": True,
            },
            "constraint_type": "fnr",
        },
    },
}

exponentiated_gradient_punitive_space = {
    "exponentiated_gradient_punitive": {
        "classpath": "aequitas.flow.methods.inprocessing.FairlearnClassifier",
        "args": {
            "reduction": ["fairlearn.reductions.ExponentiatedGradient"],
            "estimator": ["lightgbm.LGBMClassifier"],
            "constraint": ["fairlearn.reductions.FalsePositiveRateParity"],
            "eps": {"type": "float", "range": [0.005, 0.5], "log": True},
            "max_iter": [10],
            "model__boosting_type": ["dart"],
            "model__enable_bundle": [False],
            "model__n_estimators": {"type": "int", "range": [100, 1000]},
            "model__num_leaves": {"type": "int", "range": [10, 1000]},
            "model__min_child_samples": {
                "type": "int",
                "range": [1, 500],
                "log": True,
            },
            "model__learning_rate": {"type": "float", "range": [0.001, 0.1]},
            "model__n_jobs": [1],
        },
    },
}

exponentiated_gradient_assistive_space = {
    "exponentiated_gradient_assistive": {
        "classpath": "aequitas.flow.methods.inprocessing.FairlearnClassifier",
        "args": {
            "reduction": ["fairlearn.reductions.ExponentiatedGradient"],
            "estimator": ["lightgbm.LGBMClassifier"],
            "constraint": ["fairlearn.reductions.TruePositiveRateParity"],
            "eps": {"type": "float", "range": [0.005, 0.5], "log": True},
            "max_iter": [10],
            "model__boosting_type": ["dart"],
            "model__enable_bundle": [False],
            "model__n_estimators": {"type": "int", "range": [100, 1000]},
            "model__num_leaves": {"type": "int", "range": [10, 1000]},
            "model__min_child_samples": {
                "type": "int",
                "range": [1, 500],
                "log": True,
            },
            "model__learning_rate": {"type": "float", "range": [0.001, 0.1]},
            "model__n_jobs": [1],
        },
    },
}

grid_search_punitive_space = {
    "grid_search_punitive": {
        "classpath": "aequitas.flow.methods.inprocessing.FairlearnClassifier",
        "args": {
            "reduction": ["fairlearn.reductions.GridSearch"],
            "estimator": ["lightgbm.LGBMClassifier"],
            "constraint": ["fairlearn.reductions.FalsePositiveRateParity"],
            "grid_size": [10],
            "model__boosting_type": ["dart"],
            "model__enable_bundle": [False],
            "model__n_estimators": {"type": "int", "range": [100, 1000]},
            "model__num_leaves": {"type": "int", "range": [10, 1000]},
            "model__min_child_samples": {
                "type": "int",
                "range": [1, 500],
                "log": True,
            },
            "model__learning_rate": {"type": "float", "range": [0.001, 0.1]},
            "model__n_jobs": [1],
        },
    },
}

grid_search_assistive_space = {
    "grid_search_assistive": {
        "classpath": "aequitas.flow.methods.inprocessing.FairlearnClassifier",
        "args": {
            "reduction": ["fairlearn.reductions.GridSearch"],
            "estimator": ["lightgbm.LGBMClassifier"],
            "constraint": ["fairlearn.reductions.TruePositiveRateParity"],
            "grid_size": [10],
            "model__boosting_type": ["dart"],
            "model__enable_bundle": [False],
            "model__n_estimators": {"type": "int", "range": [100, 1000]},
            "model__num_leaves": {"type": "int", "range": [10, 1000]},
            "model__min_child_samples": {
                "type": "int",
                "range": [1, 500],
                "log": True,
            },
            "model__learning_rate": {"type": "float", "range": [0.001, 0.1]},
            "model__n_jobs": [1],
        },
    },
}

# Method configurations
lightgbm_method = {
    "lightgbm_baseline": {
        "inprocessing": lightgbm_space,
    },
}

undersampling_method = {
    "undersampling": {
        "preprocessing": undersampling_space,
        "inprocessing": lightgbm_space,
    },
}

oversampling_method = {
    "oversampling": {
        "preprocessing": oversampling_space,
        "inprocessing": lightgbm_space,
    },
}

fairgbm_punitive_method = {
    "fairgbm_punitive": {
        "inprocessing": fairgbm_punitive_space,
    },
}

fairgbm_assistive_method = {
    "fairgbm_assistive": {
        "inprocessing": fairgbm_assistive_space,
    },
}

exponentiated_gradient_punitive_method = {
    "exponentiated_gradient_punitive": {
        "inprocessing": exponentiated_gradient_punitive_space,
    },
}

exponentiated_gradient_assistive_method = {
    "exponentiated_gradient_assistive": {
        "inprocessing": exponentiated_gradient_assistive_space,
    },
}

grid_search_punitive_method = {
    "grid_search_punitive": {
        "inprocessing": grid_search_punitive_space,
    },
}

grid_search_assistive_method = {
    "grid_search_assistive": {
        "inprocessing": grid_search_assistive_space,
    },
}

preprocessing_methods = [
    lightgbm_method,
    undersampling_method,
    oversampling_method,
]

inprocessing_methods = [
    fairgbm_punitive_method,
    fairgbm_assistive_method,
    exponentiated_gradient_punitive_method,
    exponentiated_gradient_assistive_method,
    grid_search_punitive_method,
    grid_search_assistive_method,
]

baseline_methods = [
    lightgbm_method,
]

# Experiment configurations
test_experiment = {
    "n_trials": 1,
    "n_jobs": 1,
    "sampler": "RandomSampler",
    "sampler_args": {
        "seed": 42,
    },
}

small_experiment = {
    "n_trials": 10,
    "n_jobs": 1,
    "sampler": "RandomSampler",
    "sampler_args": {
        "seed": 42,
    },
}

medium_experiment = {
    "n_trials": 50,
    "n_jobs": 1,
    "sampler": "RandomSampler",
    "sampler_args": {
        "seed": 42,
    },
}

large_experiment = {
    "n_trials": 100,
    "n_jobs": 1,
    "sampler": "RandomSampler",
    "sampler_args": {
        "seed": 42,
    },
}
