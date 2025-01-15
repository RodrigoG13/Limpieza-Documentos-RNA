import os

def renombrar_imagenes(directorio, inicio, fin):
    # Obtener la lista de archivos en el directorio
    archivos = os.listdir(directorio)
    archivos = sorted(archivos)  # Opcional, para ordenar los archivos

    """# Comprobar si el rango es adecuado
    if fin - inicio + 1 != len(archivos):
        print("El rango no coincide con el número de archivos.")
        return"""

    # Renombrar cada archivo
    for i, archivo in enumerate(archivos):
        nombre_nuevo = f"{inicio + i}.jpg"  # Asumiendo que son imágenes JPG
        os.rename(os.path.join(directorio, archivo), os.path.join(directorio, nombre_nuevo))

# Usar la función
directorio = 'Uri'  # Cambia esto por tu directorio
inicio = 301  # Comenzar a numerar desde 100
fin = 355    # Finalizar en 199 (por ejemplo)

renombrar_imagenes(directorio, inicio, fin)
