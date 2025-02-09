import os
import sys
import yaml
import shutil
import argparse
import logging
from pathlib import Path
from picsellia import Client
from picsellia.types.enums import AnnotationFileType, LogType, InferenceType, Framework
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator


def load_config(config_file: str) -> dict:
    """Charge la configuration depuis un fichier YAML."""
    with open(config_file, "r") as file:
        config_yaml = yaml.safe_load(file)
    return config_yaml["config"]

def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def prepare_dataset(dataset, config: dict) -> None:
    """Prépare le dataset en téléchargeant les images, en exportant et réorganisant les annotations."""
    dataset_path = Path(config["DATASET_PATH"])
    annotation_path = Path(config["ANNOTATION_PATH"])

    # Supprime le dossier existant
    if dataset_path.exists():
        shutil.rmtree(dataset_path, ignore_errors=True)

    # Effectue le split train/val/test
    train_assets, val_assets, test_assets, _, _, _, labels = dataset.train_test_val_split([0.6, 0.2, 0.2],
                                                                                          random_seed=42)
    # Crée les dossiers nécessaires
    for split in ['train', 'val', 'test']:
        (dataset_path / "images" / split).mkdir(parents=True, exist_ok=True)

    # Télécharge les images
    train_assets.download(str(dataset_path / "images" / "train"), use_id=True)
    val_assets.download(str(dataset_path / "images" / "val"), use_id=True)
    test_assets.download(str(dataset_path / "images" / "test"), use_id=True)

    # Exporte les annotations au format YOLO
    annotation_path.mkdir(parents=True, exist_ok=True)
    dataset.export_annotation_file(AnnotationFileType.YOLO, str(annotation_path), use_id=True)
    annotation_zip = annotation_path / f'{dataset.id}_annotations.zip'
    if annotation_zip.exists():
        shutil.unpack_archive(str(annotation_zip), str(annotation_path))

    # Déplace les fichiers d’annotation dans les dossiers correspondants
    for split in ['train', 'val', 'test']:
        labels_dir = dataset_path / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir = dataset_path / "images" / split
        for image_file in images_dir.iterdir():
            if image_file.is_file():
                image_name = image_file.stem
                annotation_file = annotation_path / f'{image_name}.txt'
                if annotation_file.exists():
                    shutil.move(str(annotation_file), str(labels_dir / f'{image_name}.txt'))

    # Crée le fichier de configuration YOLO
    yolo_config = {
        "path": config["DATASET_PATH"],
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: label.name for index, label in enumerate(labels)}
    }
    yolo_config_path = Path(config["YOLO_CONFIG_PATH"])
    with open(yolo_config_path, 'w') as file:
        yaml.dump(yolo_config, file)

    logging.info("Dataset prepared.")


def on_train_end(trainer: DetectionTrainer) -> None:
    """Callback appelé à la fin de l'entraînement."""
    export = experiment.export_in_existing_model(client.get_model(CONFIG["PROJECT_NAME"]))
    experiment.attach_model_version(export)
    experiment.attach_dataset(CONFIG["EXPERIMENT_NAME"], dataset)
    export.update(type=InferenceType.OBJECT_DETECTION)
    export.update(framework=Framework.PYTORCH)
    export.store(name="model-best", path=trainer.best)
    experiment.log_parameters(CONFIG["TRAIN_CONFIG"])

    # Stocke plusieurs fichiers de métriques
    metrics_files = {
        "confusion_matrix": "confusion_matrix.png",
        "confusion_matrix_normalized": "confusion_matrix_normalized.png",
        "F1_curve": "F1_curve.png",
        "P_curve": "P_curve.png",
        "PR_curve": "PR_curve.png",
        "R_curve": "R_curve.png",
        "labels": "labels.jpg"
    }
    for key, filename in metrics_files.items():
        experiment.store(key, str(Path(trainer.save_dir) / filename))


def on_train_epoch_end(trainer: DetectionTrainer) -> None:
    """Callback appelé à la fin de chaque epoch pour enregistrer les métriques d'entraînement."""
    log_data = {
        "box_loss": float(trainer.loss_items[0].item()),
        "cls_loss": float(trainer.loss_items[1].item()),
        "dfl_loss": float(trainer.loss_items[2].item()),
        "epoch": float(trainer.epoch),
        "fitness": trainer.fitness,
        "precision(B)": float(trainer.metrics["metrics/precision(B)"]),
        "recall(B)": float(trainer.metrics["metrics/recall(B)"]),
        "mAP50(B)": float(trainer.metrics["metrics/mAP50(B)"]),
        "mAP50-95(B)": float(trainer.metrics["metrics/mAP50-95(B)"]),
        "pg0": float(trainer.lr["lr/pg0"]),
        "pg1": float(trainer.lr["lr/pg1"]),
        "pg2": float(trainer.lr["lr/pg2"]),
    }
    for name, value in log_data.items():
        experiment.log(name=name, data={'train': [value]}, type=LogType.LINE)


def on_val_end(trainer: DetectionValidator) -> None:
    """Callback appelé à la fin de la validation."""
    experiment.log(name="box_loss", data={'val': [float(trainer.loss[0].item())]}, type=LogType.LINE)
    experiment.log(name="cls_loss", data={'val': [float(trainer.loss[1].item())]}, type=LogType.LINE)
    experiment.log(name="dfl_loss", data={'val': [float(trainer.loss[2].item())]}, type=LogType.LINE)


def train_and_evaluate(client, project, dataset, config: dict) -> None:
    """Lance l'entraînement et l'évaluation du modèle YOLO."""
    global experiment  # Pour que les callbacks puissent y accéder
    try:
        experiment = project.get_experiment(config["EXPERIMENT_NAME"])
    except Exception:
        experiment = project.create_experiment(config["EXPERIMENT_NAME"])

    # Initialisation du modèle YOLO
    model = YOLO(config["YOLO_MODEL"])
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end", on_train_end)
    model.add_callback("on_val_end", on_val_end)

    logging.info(f'Training {config["YOLO_MODEL"]} on {config["TRAIN_CONFIG"]["data"]}')
    model.train(**config["TRAIN_CONFIG"])

    # Évaluation sur le jeu de test
    results = model.predict(f"{config['DATASET_PATH']}/images/test", device="cuda")
    for item in results:
        img_id = Path(item.path).stem
        asset = dataset.find_asset(id=img_id)
        boxes = [
            (
                int(item.boxes.xywh[i][0] - item.boxes.xywh[i][2] // 2),  # x_min
                int(item.boxes.xywh[i][1] - item.boxes.xywh[i][3] // 2),  # y_min
                int(item.boxes.xywh[i][2]),  # width
                int(item.boxes.xywh[i][3]),  # height
                dataset.get_label(item.names[int(item.boxes.cls[i])]),  # label
                float(item.boxes.conf[i])  # confidence
            )
            for i in range(item.boxes.cls.shape[0])
        ]
        experiment.add_evaluation(asset, rectangles=boxes)
        logging.info(f'Asset {img_id} evaluation uploaded')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Gestion des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Script de formation YOLO avec picsellia")
    parser.add_argument("-clear", action="store_true", help="Prépare le dataset en le nettoyant d'abord")
    args = parser.parse_args()

    # Chargement de la configuration
    CONFIG = load_yaml("config.yaml")["config"]
    CONFIG_TRAIN = load_yaml("config_train.yaml")["config_train"]
    CONFIG.update({
        "DATASET_PATH": "./dataset",
        "ANNOTATION_PATH": "./annotations",
        "YOLO_CONFIG_PATH": "./dataset/yolo_config.yaml",
        "YOLO_MODEL": "yolo11n.pt",
        "TRAIN_CONFIG": CONFIG_TRAIN,
    })

    # Initialisation du client et récupération du projet/dataset
    client = Client(
        organization_name=CONFIG["ORGA_NAME"],
        api_token=CONFIG["API_TOKEN"]
    )
    project = client.get_project(CONFIG["PROJECT_NAME"])
    dataset = client.get_dataset_version_by_id(CONFIG["DATASET_ID"])

    # Préparation du dataset si l'argument -clear est présent
    if args.clear:
        prepare_dataset(dataset, CONFIG)

    # Entraînement et évaluation
    train_and_evaluate(client, project, dataset, CONFIG)
