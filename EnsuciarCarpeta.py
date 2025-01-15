import os
import random
from PIL import Image
from Dibujantes import *

def aplicar_dibujante_aleatorio(ruta_imagen, dibujantes, probabilidades):
    """
    Aplica un Dibujante aleatorio, basado en las probabilidades dadas, a la imagen especificada.

    :param ruta_imagen: Ruta a la imagen a procesar.
    :param dibujantes: Lista de clases Dibujante disponibles.
    :param probabilidades: Lista de probabilidades para cada Dibujante.
    :return: Una imagen procesada por un Dibujante aleatorio.
    """
    dibujante = random.choices(dibujantes, weights=probabilidades, k=1)[0]()  
    return dibujante.rayar(ruta_imagen)

def procesar_imagenes(carpeta_origen, carpeta_destino, dibujantes, probabilidades):
    """
    Procesa todas las imágenes en una carpeta con Dibujantes aleatorios basados en probabilidades.

    :param carpeta_origen: Ruta a la carpeta con las imágenes originales.
    :param carpeta_destino: Ruta a la carpeta donde se guardarán las imágenes procesadas.
    :param dibujantes: Lista de clases Dibujante disponibles.
    :param probabilidades: Lista de probabilidades para cada Dibujante.
    """
    os.makedirs(carpeta_destino, exist_ok=True)
    for archivo in os.listdir(carpeta_origen):
        ruta_completa = os.path.join(carpeta_origen, archivo)
        if os.path.isfile(ruta_completa) and archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            imagen_procesada = aplicar_dibujante_aleatorio(ruta_completa, dibujantes, probabilidades)
            imagen_procesada.save(os.path.join(carpeta_destino, archivo))

# Define tus dibujantes y sus probabilidades correspondientes
dibujantes_disponibles = [DibujanteGeometrico, DibujanteEscritor, DibujanteSazon, DibujanteGenesis, DibujanteEscritorDeLinea]
probabilidades = [0, 0.35, 0, 0.5, 0.15]  # Asegúrate de que sumen 1

carpeta_origen = 'Test'
carpeta_destino = 'TestBuenas'
procesar_imagenes(carpeta_origen, carpeta_destino, dibujantes_disponibles, probabilidades)
