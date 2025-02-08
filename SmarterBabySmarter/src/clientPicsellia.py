import os
import sys
import yaml
import shutil
import pandas as pd
from picsellia import Client
from picsellia.types.enums import AnnotationFileType, LogType, InferenceType, Framework
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

# Configuration globale
CONFIG = {
    "ORGA_NAME": "Picsalex-MLOps",
    "API_TOKEN": "",
    "DATASET_ID": "0193688e-aa8f-7cbe-9396-bec740a262d0",
    "PROJECT_NAME": "Groupe_4",
    "EXPERIMENT_NAME": "Test_rework9",
    "DATASET_PATH": "./dataset",
    "ANNOTATION_PATH": "./annotations",
    "YOLO_CONFIG_PATH": "./dataset/yolo_config.yaml",
    "YOLO_MODEL": "yolo11n.pt",
    "TRAIN_CONFIG": {
        "data": "./dataset/yolo_config.yaml",
        "epochs": 10,
        "batch": 32,
        "imgsz": 640,
        "device": "cuda",
        "workers": 8,
        "optimizer": "AdamW",
        "lr0": 0.01,
        "patience": 40,
        "seed": 42,
        "mosaic": 0,
        "close_mosaic": 0,
    },
}


def on_train_end(trainer: DetectionTrainer):
    """Callback à la fin de l'entraînement pour enregistrer les métriques."""
    try:
        metrics_csv = pd.read_csv(trainer.csv)
        for col in metrics_csv.columns:
            experiment.log(name=col, data=list(metrics_csv[col]), type=LogType.LINE)
    except Exception as e:
        print(f'Error logging metrics: {e}')

    export = experiment.export_in_existing_model(client.get_model(CONFIG["PROJECT_NAME"]))
    experiment.attach_model_version(export)
    experiment.attach_dataset(CONFIG["EXPERIMENT_NAME"], dataset)
    export.update(type=InferenceType.OBJECT_DETECTION)
    export.update(framework=Framework.PYTORCH)
    export.store(name="model-latest", path=trainer.best)
    experiment.log_parameters(CONFIG["TRAIN_CONFIG"])

    # Log additional information
    experiment.log(name="box_loss", data={'train': [float(trainer.loss_items[0].item())]}, type=LogType.LINE)
    experiment.log(name="cls_loss", data={'train': [float(trainer.loss_items[1].item())]}, type=LogType.LINE)
    experiment.log(name="dfl_loss", data={'train': [float(trainer.loss_items[2].item())]}, type=LogType.LINE)


if __name__ == "__main__":
    client = Client(
        organization_name=CONFIG["ORGA_NAME"],
        api_token=CONFIG["API_TOKEN"]
    )

    project = client.get_project(CONFIG["PROJECT_NAME"])
    dataset = client.get_dataset_version_by_id(CONFIG["DATASET_ID"])

    try:
        experiment = project.get_experiment(CONFIG["EXPERIMENT_NAME"])
    except Exception:
        experiment = project.create_experiment(CONFIG["EXPERIMENT_NAME"])

    if len(sys.argv) >= 2 and sys.argv[1] == "-clear":
        shutil.rmtree(CONFIG["DATASET_PATH"], ignore_errors=True)
        train_assets, val_assets, test_assets, _, _, _, labels = dataset.train_test_val_split([0.6, 0.2, 0.2],
                                                                                              random_seed=42)

        train_assets.download(f"{CONFIG['DATASET_PATH']}/images/train", use_id=True)
        val_assets.download(f"{CONFIG['DATASET_PATH']}/images/val", use_id=True)
        test_assets.download(f"{CONFIG['DATASET_PATH']}/images/test", use_id=True)

        dataset.export_annotation_file(AnnotationFileType.YOLO, CONFIG["ANNOTATION_PATH"], use_id=True)

        annotation_zip = f'{CONFIG["ANNOTATION_PATH"]}/{dataset.id}_annotations.zip'
        if os.path.exists(annotation_zip):
            shutil.unpack_archive(annotation_zip, CONFIG["ANNOTATION_PATH"])

        for split in ['train', 'val', 'test']:
            os.makedirs(f'{CONFIG["DATASET_PATH"]}/labels/{split}', exist_ok=True)
            for image in os.listdir(f"{CONFIG['DATASET_PATH']}/images/{split}"):
                image_name = os.path.splitext(image)[0]
                annotation_file = f'{CONFIG["ANNOTATION_PATH"]}/{image_name}.txt'
                if os.path.exists(annotation_file):
                    shutil.move(annotation_file, f'{CONFIG["DATASET_PATH"]}/labels/{split}/{image_name}.txt')

        yaml_config = {
            "path": CONFIG["DATASET_PATH"],
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {index: label.name for index, label in enumerate(labels)}
        }

        with open(CONFIG["YOLO_CONFIG_PATH"], 'w') as file:
            yaml.dump(yaml_config, file)

        print("Dataset prepared")

    # Initialisation du modèle YOLO
    model = YOLO(CONFIG["YOLO_MODEL"])
    model.add_callback("on_train_end", on_train_end)

    print(f'Training {CONFIG["YOLO_MODEL"]} on {CONFIG["TRAIN_CONFIG"]["data"]}')
    model.train(**CONFIG["TRAIN_CONFIG"])
