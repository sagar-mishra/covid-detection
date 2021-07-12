import config
import os
import torch
import cv2
from models.base_conv_net_model import BaseConvNet
from util.inference_util import load_model, predict
from flask import Flask
from flask import request

app = Flask(__name__)

Model = None
IDX_TO_CLASS = None
CLASS_TO_IDX = None
Device = config.INFERENCE_DEVICE

@app.route("/")
def welcome():
    return "hello"

@app.route("/predict")
def do_prediction():
    image_path = request.args.get("image_path")
    image = cv2.imread(image_path)
    prediction = predict(MODEL, image, IDX_TO_CLASS, height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1])
    return f"Prediction is : {prediction}"

if __name__ == "__main__":

    current_model_weight_path = os.path.join(config.MODEL_PATH, config.MODEL_NAME)
    check_point_path = os.path.join(current_model_weight_path, config.CHECKPOINT_NAME)
    check_point = torch.load(check_point_path, map_location = torch.device(config.INFERENCE_DEVICE))
    model = BaseConvNet(num_classes=config.NUM_OF_CLASSES)
    MODEL = load_model(model, check_point["state_dict"])
    CLASS_TO_IDX = check_point["class_to_idx"]
    IDX_TO_CLASS = check_point['idx_to_class']

    app.run()