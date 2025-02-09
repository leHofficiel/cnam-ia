import os
import sys
import yaml
from picsellia import Client
from ultralytics import YOLO

with open("config.yaml","r") as file:
    yaml_config = yaml.safe_load(file)["config"]

CONFIG = {
    "ORGA_NAME": yaml_config["ORGA_NAME"],
    "API_TOKEN": yaml_config["API_TOKEN"],
    "DATASET_ID": yaml_config["DATASET_ID"],
    "PROJECT_NAME": yaml_config["PROJECT_NAME"],
    "EXPERIMENT_NAME": yaml_config["EXPERIMENT_NAME"],}

if __name__ == "__main__":
    IMAGE_PATH = "./images"
    VIDEO_PATH = "./videos"
    MODEL_PATH = "./runs/detect/tune/weights/best.pt"

    client = Client(
        organization_name=CONFIG["ORGA_NAME"],
        api_token=CONFIG["API_TOKEN"]
    )

    project = client.get_project(CONFIG["PROJECT_NAME"])
    experiment = project.get_experiment(CONFIG["EXPERIMENT_NAME"])

    base_model = experiment.get_base_model_version()
    base_model_pt = base_model.get_file('model-latest')
    base_model_pt.download("./",force_replace=True)

    model = YOLO("./best.pt")
    if len(sys.argv) >= 2 and sys.argv[1] == "-webcam":
        results = model(0,show= True)
    if len(sys.argv) >= 3 and sys.argv[1] == "-image":
        results = model(sys.argv[2],show= True)
    if len(sys.argv) >= 3 and sys.argv[1] == "-video":
        results = model(sys.argv[2],show= True)