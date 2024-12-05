from src.image_processor import ImageProcessor

if __name__ == "__main__":
    chemin_images = "input_images"
    taille_cible = 640
    processeur = ImageProcessor(chemin_images)
    processeur.process_folder(taille_cible)
