import math


class Dog:
    def __init__(self, name, age):
        self.name = name
        self.__age = age

    def bark(self):
        return f"{self.name} says Woof !"

my_dog = Dog("Doggy", 3)

class Circle:
    pi = math.pi

    def __init__(self, radius):
        self._radius = radius

    def area(self):
        return Circle.pi * self._radius ** 2

circle = Circle(5)
print(circle.area())

