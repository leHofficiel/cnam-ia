import sys

import yaml
from picsellia import Client
from ultralytics import YOLO

with open("config.yaml", "r", encoding="utf-8") as file:
    yaml_config = yaml.safe_load(file)["config"]

config = {
    "ORGA_NAME": yaml_config["ORGA_NAME"],
    "API_TOKEN": yaml_config["API_TOKEN"],
    "DATASET_ID": yaml_config["DATASET_ID"],
    "PROJECT_NAME": yaml_config["PROJECT_NAME"],
    "EXPERIMENT_NAME": yaml_config["EXPERIMENT_NAME"],
}

if __name__ == "__main__":

    client = Client(organization_name=config["ORGA_NAME"], api_token=config["API_TOKEN"])

    project = client.get_project(config["PROJECT_NAME"])
    experiment = project.get_experiment(config["EXPERIMENT_NAME"])

    base_model = experiment.get_base_model_version()
    base_model_pt = base_model.get_file("model-best")
    base_model_pt.download("./", force_replace=True)

    model = YOLO("./best.pt")
    if len(sys.argv) >= 2 and sys.argv[1] == "-webcam":
        results = model(0, show=True)
    elif len(sys.argv) >= 3 and sys.argv[1] == "-image":
        results = model(sys.argv[2], save=True, project="./result_after_exec/images")
    elif len(sys.argv) >= 3 and sys.argv[1] == "-video":
        results = model(sys.argv[2], save=True, project="./result_after_exec/videos")
