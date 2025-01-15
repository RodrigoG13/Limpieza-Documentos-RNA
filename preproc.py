import cv2
import numpy as np
from matplotlib import pyplot as plt

# Leer la imagen
imagen = cv2.imread('1.jpeg')

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Redimensionar la imagen
tamaño_deseado = (256, 256)
gris_redimensionada = cv2.resize(gris, tamaño_deseado)

# Aplicar un filtro para reducir el ruido
gris_suavizada = cv2.GaussianBlur(gris_redimensionada, (5, 5), 0)

# Normalización de los píxeles
gris_normalizada = gris_suavizada / 255.0

# Mostrar la imagen procesada
plt.imshow(gris_normalizada, cmap='gray')
plt.show()
