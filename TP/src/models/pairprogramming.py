# from abc import ABC, abstractmethod
#
#
# class Image(ABC):
#     def __init__(self, data):
#         self._data= data
#
#     @property
#     @abstractmethod
#     def width(self):
#         pass
#
#     @property
#     @abstractmethod
#     def height(self):
#         pass
#
#     @property
#     @abstractmethod
#     def get_pixel(self,x,y):
#         pass
#
#     @property
#     @abstractmethod
#     def set_pixel(self, x, y, color):
#         pass
#
#
#
#
# class Grayscale(Image):
#
#     @property
#     def set_pixel(self, x, y, color):
#         self.data[y][x]= color
#
#     def __init__(self,data):
#         super().__init__(data)
#
#     @property
#     def height(self):
#         return len(self._data)
#
#     @property
#     def get_pixel(self,x,y):
#         return self._data[y][x]
#
#     @property
#     def width(self):
#         return len(self._data[0])
#
# data = [[]]
# grayscale_image = Grayscale(data)
# print(grayscale_image.width)

# from PIL import Image
#
# image = Image.open("../images/Kim.Chaewon.jpg")
#
# resized_image = image.resize((300,300))
#
# rotated_image = image.rotate(45)
#
# rotated_image.save("../images/rotated.jpg")
#
# print("toto")

import numpy as np
from PIL import Image

image = Image.open("../../input_images/Kim.Chaewon.jpg")
image_data = np.array(image)

print(image_data.shape)

new_array = np.array([[1, 2], [3, 4]])
print(new_array * 2)

sequence_data = np.arange(2, 14)
print(sequence_data)

sequence_data = sequence_data.reshape(6, 2)

# put blue channel to 0
image_data[:, :, 2] = 0
print(image_data)

np.mean(image_data)
np.min(image_data)
np.max(image_data)

processed_image = Image.fromarray(image_data)
processed_image.save("../images/processed.jpg")
