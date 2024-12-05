# import os
# import zipfile
#
# from picsellia import Client
# from picsellia.types.enums import AnnotationFileType
# from sklearn.model_selection import train_test_split
# import shutil
#
# # Your organization
# ORGANIZATION_NAME = "Picsalex-MLOps"
# OUTPUT_DIR = "./datasets"
# YOLO_DIR = "./yolo_dataset"
# DATASET_ID = "0193688e-aa8f-7cbe-9396-bec740a262d0"
#
# client = Client(
#     api_token="a89eb6bbf402bd5cb538415ebf46b2709c16d4ed",
#     organization_name=ORGANIZATION_NAME,
# )
#
# # # Récupération du dataset version
# # dataset = client.get_dataset_version_by_id(DATASET_ID)
# #
# # # Export des annotations au format YOLO
# # export_path = os.path.join(
# #     OUTPUT_DIR,
# #     f"yolo_annotations.zip/{DATASET_ID}/annotations/{DATASET_ID}_annotations.zip",
# # )
# # os.makedirs(OUTPUT_DIR, exist_ok=True)
# #
# # print("Export des annotations au format YOLO...")
# # dataset.export_annotation_file(
# #     annotation_file_type=AnnotationFileType.YOLO,
# #     # target_path=export_path,  # Format YOLO
# # )
# # print(f"Export terminé. Fichier exporté : {export_path}")
# #
# # print(f"Export terminé. Fichier exporté : {export_path}")
# #
# # # Décompression du fichier exporté
# # print("Décompression des annotations YOLO...")
# # with zipfile.ZipFile(export_path, "r") as zip_ref:
# #     zip_ref.extractall(OUTPUT_DIR)
# #
# # print(f"Décompression terminée dans : {OUTPUT_DIR}")
#
# # Récupérer la version du dataset
# dataset = client.get_dataset_version_by_id(DATASET_ID)
#
# output_dir_dataset = "./datasets"
# os.makedirs(output_dir_dataset, exist_ok=True)
# dataset.list_assets().download(output_dir_dataset)
# print("Datasets importés")
#
# project = client.get_project(project_name="Groupe_4")
#
# experiment = project.get_experiment(name="experiment-0")
# """
# experiment = project.create_experiment(
#     name="experiment-0",
#     description="base experiment",
# )
#
# experiment.attach_dataset(
#     name="⭐️ cnam_product_2024",
#     dataset_version=dataset,
# )
# """
# datasets = experiment.list_attached_dataset_versions()
# print(f"Datasets attachés à l'expérience : {datasets}")
#
# path_dir = "./datasets/annotations"
# for dataset in datasets:
#     dataset.export_annotation_file(
#         AnnotationFileType.YOLO, path_dir + "/annotations.zip"
#     )
#
#     print(f"Exportation terminée. Fichier exporté : {path_dir}")
#     annotations_dir = os.path.join(path_dir, "annotations")
#
#     # Trouver la première archive ZIP dans le dossier ou sous-dossier
#     zip_file = next(
#         (
#             os.path.join(root, file)
#             for root, _, files in os.walk(path_dir)
#             for file in files
#             if file.endswith(".zip")
#         ),
#         None,
#     )
#
#     if zip_file:
#         # Créer le dossier "annotations" s'il n'existe pas
#         os.makedirs(annotations_dir, exist_ok=True)
#
#         # Décompresser l'archive ZIP
#         with zipfile.ZipFile(zip_file, "r") as zip_ref:
#             zip_ref.extractall(annotations_dir)
#         print(f"Archive décompressée dans : {annotations_dir}")
#
#         # Supprimer l'archive ZIP
#         os.remove(zip_file)
#         print(f"Archive {zip_file} supprimée.")
#     else:
#         print(
#             "Aucune archive ZIP trouvée dans le dossier 'datasets' ou ses sous-dossiers."
#         )
#
#
# extracted_files = os.listdir(annotations_dir)
# print(f"Fichiers extraits : {extracted_files}")
#
# file_count = len(extracted_files)
# print(f"Nombre total de fichiers extraits : {file_count}")
#
#
# # Dossiers d'entrée (à adapter si nécessaire)
# images_dir = os.path.join(YOLO_DIR, "images")
# labels_dir = os.path.join(YOLO_DIR, "labels")
#
# # Dossiers de sortie
# train_images_dir = os.path.join(YOLO_DIR, "images/train")
# val_images_dir = os.path.join(YOLO_DIR, "images/val")
# train_labels_dir = os.path.join(YOLO_DIR, "labels/train")
# val_labels_dir = os.path.join(YOLO_DIR, "labels/val")
#
# # Création des répertoires de sortie
# os.makedirs(train_images_dir, exist_ok=True)
# os.makedirs(val_images_dir, exist_ok=True)
# os.makedirs(train_labels_dir, exist_ok=True)
# os.makedirs(val_labels_dir, exist_ok=True)

import os
import shutil
import zipfile
import random
from picsellia import Client
from picsellia.types.enums import AnnotationFileType

# Configuration de l'organisation et des chemins
ORGANIZATION_NAME = "Picsalex-MLOps"
OUTPUT_DIR = "./datasets"
YOLO_DIR = "./yolo_dataset"
DATASET_ID = "0193688e-aa8f-7cbe-9396-bec740a262d0"

client = Client(
    #TODO mettre son token
    api_token="token",
    organization_name=ORGANIZATION_NAME,
)

# Récupérer la version du dataset
dataset = client.get_dataset_version_by_id(DATASET_ID)

# Télécharger le dataset
os.makedirs(OUTPUT_DIR, exist_ok=True)
dataset.list_assets().download(OUTPUT_DIR)
print("Datasets importés")

# Exporter les annotations au format YOLO
annotations_zip = os.path.join(OUTPUT_DIR, "annotations.zip")
dataset.export_annotation_file(AnnotationFileType.YOLO, annotations_zip)
print(f"Export des annotations terminé. Fichier exporté : {annotations_zip}")

# Décompresser les annotations YOLO
annotations_dir = os.path.join(OUTPUT_DIR, "annotations")
os.makedirs(annotations_dir, exist_ok=True)

with zipfile.ZipFile(annotations_zip, "r") as zip_ref:
    zip_ref.extractall(annotations_dir)
print(f"Annotations décompressées dans : {annotations_dir}")

# Supprimer le fichier ZIP après extraction
os.remove(annotations_zip)

# Déplacer les fichiers d'annotations et d'images dans le répertoire YOLO
images_dir = os.path.join(YOLO_DIR, "images")
labels_dir = os.path.join(YOLO_DIR, "labels")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

for root, _, files in os.walk(annotations_dir):
    for file in files:
        if file.endswith((".jpg", ".png")):
            shutil.move(os.path.join(root, file), images_dir)
        elif file.endswith(".txt"):
            shutil.move(os.path.join(root, file), labels_dir)

# Générer les splits train, val, test
image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]
label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

image_to_label = {os.path.splitext(img)[0]: img for img in image_files}
label_to_image = {os.path.splitext(lbl)[0]: lbl for lbl in label_files}

common_files = list(set(image_to_label.keys()).intersection(set(label_to_image.keys())))

random.seed(42)
random.shuffle(common_files)

total = len(common_files)
train_split = int(0.6 * total)
val_split = int(0.2 * total) + train_split

train_files = common_files[:train_split]
val_files = common_files[train_split:val_split]
test_files = common_files[val_split:]

# Dossiers de sortie pour les splits
train_images_dir = os.path.join(YOLO_DIR, "images/train")
val_images_dir = os.path.join(YOLO_DIR, "images/val")
test_images_dir = os.path.join(YOLO_DIR, "images/test")
train_labels_dir = os.path.join(YOLO_DIR, "labels/train")
val_labels_dir = os.path.join(YOLO_DIR, "labels/val")
test_labels_dir = os.path.join(YOLO_DIR, "labels/test")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)


def copy_split(files, images_target, labels_target):
    for file in files:
        shutil.copy(os.path.join(images_dir, image_to_label[file]), images_target)
        shutil.copy(os.path.join(labels_dir, label_to_image[file]), labels_target)


copy_split(train_files, train_images_dir, train_labels_dir)
copy_split(val_files, val_images_dir, val_labels_dir)
copy_split(test_files, test_images_dir, test_labels_dir)

# Générer le fichier coco.yaml
coco_yaml_path = os.path.join(YOLO_DIR, "coco.yaml")
num_classes = 80  # Remplacez par le nombre réel de classes
class_names = [
    f"class_{i}" for i in range(num_classes)
]  # Modifiez avec vos classes réelles

coco_yaml_content = f"""
train: {os.path.join(YOLO_DIR, 'images/train')}
val: {os.path.join(YOLO_DIR, 'images/val')}
test: {os.path.join(YOLO_DIR, 'images/test')}

nc: {num_classes}
names: {class_names}
"""

with open(coco_yaml_path, "w") as f:
    f.write(coco_yaml_content.strip())

print(f"Fichier coco.yaml généré : {coco_yaml_path}")

# Résumé final
print("Split terminé avec succès.")
print(
    f"Train : {len(train_files)} exemples, Val : {len(val_files)} exemples, Test : {len(test_files)} exemples."
)
