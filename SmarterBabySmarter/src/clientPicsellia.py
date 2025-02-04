import os
import sys

import yaml
import pandas
from networkx import config
from picsellia import Client
from picsellia.types.enums import AnnotationFileType, LogType, InferenceType, Framework
import shutil

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


def on_train_end(trainer:DetectionTrainer):
    metrics_csv = pandas.read_csv(trainer.csv)
    for col in metrics_csv.columns:
        try:
            experiment.log(name=col, data=list(metrics_csv[col]), type=LogType.LINE)
        except Exception as e:
            print(f'Error: {e}')

    export = experiment.export_in_existing_model(client.get_model(PROJECT_NAME))

    experiment.attach_model_version(export)
    export.update(type=InferenceType.OBJECT_DETECTION)
    export.update(framework=Framework.PYTORCH)
    export.store(name="model-latest", path=trainer.best)

    experiment.log_parameters(config)

if __name__ == "__main__":
    ORGA_NAME = "Picsalex-MLOps"
    API_TOKEN = "a89eb6bbf402bd5cb538415ebf46b2709c16d4ed"
    DATASET_ID = "0193688e-aa8f-7cbe-9396-bec740a262d0"
    PROJECT_NAME = "Groupe_4"
    EXPERIMENT_NAME = "Babymonster"
    DATASET_PATH = "./dataset"
    ANNOTATION_PATH = "./annotations"

    client = Client(
        organization_name=ORGA_NAME,
        api_token=API_TOKEN
    )

    project = client.get_project(PROJECT_NAME)

    dataset = client.get_dataset_version_by_id(DATASET_ID)

    try:
        experiment = project.get_experiment(EXPERIMENT_NAME)
    except Exception as e:
            experiment = project.create_experiment(EXPERIMENT_NAME)


    # experiment.attach_dataset(dataset)

    if len(sys.argv) >= 2 and sys.argv[1] == "-clear":
        shutil.rmtree(DATASET_PATH, ignore_errors=True)

        train_assets, test_assets, val_assets, count_train, count_test, count_val, labels = (
            dataset.train_test_val_split(ratios=[0.6,0.2,0.2], random_seed=42))

        train_assets.download(f"{DATASET_PATH}/images/train", use_id=True)
        val_assets.download(f"{DATASET_PATH}/images/val", use_id=True)
        test_assets.download(f"{DATASET_PATH}/images/test", use_id=True)

        dataset.export_annotation_file(AnnotationFileType.YOLO, ANNOTATION_PATH, use_id=True)
        zip_path = f'{ANNOTATION_PATH}/0192f6db-86b6-784c-80e6-163debb242d5/annotations/{dataset.id}_annotations.zip'
        shutil.unpack_archive(zip_path, ANNOTATION_PATH)

        for split in ['train', 'val', 'test']:
            os.makedirs(f'{DATASET_PATH}/labels/{split}', exist_ok=True)
            for image in os.listdir(f"{DATASET_PATH}/images/{split}"):
                image_name = os.path.splitext(image)[0]
                annotation_path = f'{ANNOTATION_PATH}/{image_name}.txt'
                if os.path.exists(annotation_path):
                    shutil.move(annotation_path, f'{DATASET_PATH}/labels/{split}/{image_name}.txt')

        yaml_config = {
            "path": DATASET_PATH,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {}
        }

        index = 0
        for label in labels:
            yaml_config["names"][index] = label.name
            index += 1

        yaml_config_path = f'{DATASET_PATH}/yolo_config.yaml'
        with open(yaml_config_path, 'w') as file:
            yaml.dump(yaml_config, file)

        print("Dataset prepared")

    config = {
        "data": './dataset/yolo_config.yaml',
        "batch_size": 32,
        "imgsz": 640,
        "device": "cuda",
        "workers": 8,
        "optimizer": "AdamW",
        "lr0": 0.01,
        "patience": 40,
        "epochs": 150,
       # "iterations": 250,
    }

    print(f'Device type: cuda')
    # Load a model
    #model = YOLO("yolo11n.pt")
    model = YOLO("yolo11m.pt")
    model.add_callback("on_train_end", on_train_end)

    print(model.task)
    print(model.overrides)
    results = model.train(
    data = config["data"],
    epochs = 500,
#    iterations = 175,
    patience = 50,
    imgsz = 640,
    device = "cuda",
    batch = 16,
    workers = 8,
    mosaic = 0,
    close_mosaic = 0,
    seed = 42,
    optimizer = "auto",

    )


    '''
    cos_lr = True,
    lr0=0.00179,
    lrf = 0.01518,
    momentum = 0.86333,
    weight_decay = 0.00051,
    warmup_epochs = 3.09569,
    warmup_momentum = 0.44584,
    box = 5.1931,
    cls = 0.74142,
    dfl = 1.16798,
    hsv_h = 0.01322,
    hsv_s = 0.41755,
    hsv_v = 0.20555,
    degrees = 0.0,
    translate = 0.0979,
    scale = 0.4822,
    shear = 0.0,
    perspective = 0.0,
    flipud = 0.0,
    fliplr = 0.16048,
    bgr = 0.0,
    mixup = 0.0,
    copy_paste = 0.0,
    '''

    '''
    results = model.train(
        data=config["data"],
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        device=config["device"],
        batch=config["batch_size"],
        workers=config["workers"],
        mosaic=0,
        close_mosaic=0,
        seed=42,
        optimizer=config["optimizer"],
        patience=75,
        lr0=0.00179,
        lrf=0.01518,
        momentum=0.86333,
        weight_decay=0.00051,
        warmup_epochs=3.09569,
        warmup_momentum=0.44584,
        box=5.1931,
        cls=0.74142,
        dfl=1.16798,
        hsv_h=0.01322,
        hsv_s=0.41755,
        hsv_v=0.20555,
        degrees=0.0,
        translate=0.0979,
        scale=0.4822,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.16048,
        bgr=0.0,
        mixup=0.0,
        copy_paste=0.0,
    )
'''
'''
    # Train the model
    results = model.train(
        data=config["data"],
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        close_mosaic=0,
        device=config["device"],
        batch=config["batch_size"],
        workers=config["workers"],
        mosaic=0,
        seed=42,

    )
'''

'''    custom_overrides = {
        'lr0': 0.00923,
        'lrf': 0.01177,
        'momentum': 0.98,
        'weight_decay': 0.00044,
        'warmup_epochs': 3.06095,
        'warmup_momentum': 0.82572,
        'box': 6.68439,
        'cls': 0.42109,
        'dfl': 1.20487,
        'hsv_h': 0.01034,
        'hsv_s': 0.63122,
        'hsv_v': 0.39781,
        'degrees': 0.0,
        'translate': 0.12352,
        'scale': 0.76731,
        'shear': 0.0,
        'fliplr': 0.48081,
        'mosaic': 0,
        'optimizer': "AdamW"
        
        

    }'''





