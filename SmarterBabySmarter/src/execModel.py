import os

from ultralytics import YOLO

if __name__ == "__main__":
    IMAGE_PATH = "./images"
    VIDEO_PATH = "./videos"
    MODEL_PATH = "./runs/detect/tune/weights/best.pt"

    model = YOLO(MODEL_PATH)
    results = model(IMAGE_PATH,show= True)

