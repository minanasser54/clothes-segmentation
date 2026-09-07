import os


class BaseConfig:
    DATASET_URL = "rajkumarl/people-clothing-segmentation"
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "Data")

    IMG_SIZE = 256
    VAL_SIZE = 0.1
    TEST_SIZE = 0.1
    RANDOM_SEED = 42

    NUM_AUGMENTED_COPIES = 2

    NUM_CLASSES = 9                       # oldly  59 = 58 clothing classes + background/null
    MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")

    UNET_FEATURES = [32, 64, 128, 256] 
    UNET_PP_FEATURES  = [32, 64, 128, 256]
    
    DICE_WEIGHT = 0.7

    DEFAULT_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_LR = 1e-3

config = BaseConfig()