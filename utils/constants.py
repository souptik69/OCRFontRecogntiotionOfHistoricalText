START_OF_SEQUENCE = "<SOS>"
END_OF_SEQUENCE = "<EOS>"
END_OF_SEQUENCE_RECIPIENT = "<EOSR>"
END_OF_SEQUENCE_BODY = "<EOSB>"
SPACE = " "
PAD = "<PAD>"
BLANK = "<BLANK>"

IMAGE = "image"
LINE_IMAGE = "line_image"
ORIGINAL_IMAGE = "original_image"
ORIGINAL_IMAGE_SIZE = "original_image_size"
TEXT = "text"
MASK = "mask"
CONTENT_INFOS = "content_infos"
RECIPIENT = "recipient"
UNPADDED_IMAGE_WIDTH = "unpadded_image_width"
UNPADDED_TEXT_LEN = "unpadded_text_len"
S2S_TEXT = "s2s_text"
TGT_KEY_PADDING_MASK = "tgt_key_padding_mask"
TGT_MASK = "tgt_mask"
MEMORY_KEY_PADDING_MASK = "memory_key_padding_mask"

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




