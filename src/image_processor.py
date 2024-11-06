import os
from datetime import datetime

from PIL import Image, ImageOps


class ImageProcessor:

    def __init__(self, chemin):
        self.chemin = chemin

    def process_folder(self, taille_souhaite: int):
        """
        Traite un dossier d'images
        Args:
            taille_souhaite (int): taille cible des images
        """
        output_folder = os.path.join(
            "datasets", datetime.now().strftime("%Y%m%d%H%M%S")
        )
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(self.chemin):
            self.process_image(
                os.path.join(self.chemin, filename), taille_souhaite, output_folder
            )

    def process_image(self, filepath: str, size: int, output_folder: str):
        """
        Traite une image
        Args:
            filepath (str): chemin de l'image
            size (int): taille ciible de l'image
            output_folder (str): Dossier de sortie
        """
        try:
            img = Image.open(filepath)
            img = self.resize_and_pad(img, size)
            img.save(os.path.join(output_folder, os.path.basename(filepath)))
            img.close()

        except Exception as e:
            print(f"Erreur lors du traitement {filepath}: {e}")

    def resize_and_pad(self, img, size):
        """
        Redimensionne et ajoute un padding à l'image
        Args:
            img: Image
            size: taille cible de l'image
        """
        original_width = img.size[0]
        original_height = img.size[1]
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            new_width = size
            new_height = int(size / aspect_ratio)
        else:
            new_height = size
            new_width = int(size * aspect_ratio)

        img = img.resize((new_width, new_height))

        return self.add_padding(img, size, new_width, new_height)

    def add_padding(self, img, size, new_width, new_height):
        """
        Ajoute un padding à l'image
        Args:
            img: Image
            size: taille cible de l'image
            new_width: nouvelle largeur
            new_height: nouvelle hauteur
        """
        width = size - new_width
        height = size - new_height
        padding = (0, 0, width, height)

        fill_color = (114, 114, 144) if img.mode == "RGB" else 114
        return ImageOps.expand(img, padding, fill=fill_color)
