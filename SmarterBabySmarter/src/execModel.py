import os

from ultralytics import YOLO

if __name__ == "__main__":
    IMAGE_PATH = "./images"
    MODEL_PATH = "./runs/detect/train3/weights/best.pt"

    model = YOLO(MODEL_PATH)

    # Run batched inference on a list of images
    images = []
    for image in os.listdir(IMAGE_PATH):
        images.append(f'{IMAGE_PATH}/{image}')

    results = model(images)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen                to disk
