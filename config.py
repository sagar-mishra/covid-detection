from numpy import Inf

import os

TRAIN_DEVICE = "cuda"
INFERENCE_DEVICE = "cpu"
BATCH_SIZE = 32
EPOCHS = 20
BASE_PATH = "D:\data\covid_detection"
DATA_PATH = os.path.join(BASE_PATH, "dataset")
TRAIN_PATH = os.path.join(DATA_PATH, "train")
VAL_PATH = os.path.join(DATA_PATH, "val")
TEST_PATH = os.path.join(DATA_PATH, "test")
MODEL_PATH = "D:\projects\CovidDetection\\trained_model_weights"