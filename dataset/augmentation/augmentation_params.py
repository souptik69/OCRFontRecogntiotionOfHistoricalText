DA_CONFIG = {"dpi": {
    "proba": 0.0,  # default: prob=0.2
    "min_factor": 0.75,
    "max_factor": 1.25,
},
    "perspective": {
        "proba": 0.6,
        "min_factor": 0,
        "max_factor": 0.3,
    },
    "elastic_distortion": {
        "proba": 0.6,
        "max_magnitude": 20,
        "max_kernel": 3,
    },
    "random_transform": {
        "proba": 0.6,
        "max_val": 16,
    },
    "dilation_erosion": {
        "proba": 0.6,
        "min_kernel": 1,
        "max_kernel": 3,
        "iterations": 1,
    },
    "brightness": {
        "proba": 0.0,  # default: proba=0.2
        "min_factor": 0.01,
        "max_factor": 1,
    },
    "contrast": {
        "proba": 0.6,
        "min_factor": 0.8,
        "max_factor": 1,
    },
    "sign_flipping": {
        "proba": 0.0,
    }, }
