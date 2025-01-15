import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.models as models
from pytorch_ssim import SSIM
import torch
import torch.nn as nn
import torch.optim as optim



class Discriminador(nn.Module):
    def __init__(self):
        super(Discriminador, self).__init__()

        # Capa 1: 500x500 -> 250x250
        self.hidden1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Capa 2: 250x250 -> 125x125
        self.hidden2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Capa 3: 125x125 -> 63x63
        self.hidden3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Capa 4: 63x63 -> 32x32
        self.hidden4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Capa 5: 32x32 -> 16x16
        self.hidden5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Capa final: 16x16 -> 13x13 (salida)
        self.out = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.hidden1(img)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.out(x)
        x = torch.mean(x, dim=[2, 3])  # Realiza un promedio global
        return x



class ImageDataset(Dataset):
    def __init__(self, dir_buenas, dir_rayadas, transformacion=None):
        self.dir_buenas = dir_buenas
        self.dir_rayadas = dir_rayadas
        self.transformacion = transformacion

        # Asumimos que cada imagen 'buena' tiene su correspondiente imagen 'rayada' con el mismo nombre
        self.nombres_archivos = os.listdir(dir_buenas)

    def __len__(self):
        return len(self.nombres_archivos)

    def __getitem__(self, idx):
        ruta_img_buena = os.path.join(self.dir_buenas, self.nombres_archivos[idx])
        ruta_img_rayada = os.path.join(self.dir_rayadas, self.nombres_archivos[idx])

        img_buena = Image.open(ruta_img_buena).convert('L')  # Convertir a escala de grises
        img_rayada = Image.open(ruta_img_rayada).convert('L')

        if self.transformacion:
            img_buena = self.transformacion(img_buena)
            img_rayada = self.transformacion(img_rayada)

        return img_rayada, img_buena   
    
    
def guardar_modelo(modelo, ruta, epoch):
    # Crear directorio si no existe
    directorio = os.path.dirname(ruta)
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    # Ruta del archivo con información de la época
    ruta_completa = f'{ruta}_epoch_{epoch}.pth'
    torch.save(modelo.state_dict(), ruta_completa)
    print(f'Modelo guardado en: {ruta_completa}')


# Hiperparametros
tam_batch = 16
tam_workers = 4
# Definición de los parámetros del entrenamiento
num_epochs = 160
visualization_interval = 5  # Visualizar cada 5 épocas, por ejemplo
dims = 500


# Definir transformaciones
transformacion = transforms.Compose([
    transforms.Resize((dims, dims)),  # Redimensionar a 500x500
    transforms.ToTensor(),  # Convertir a tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizar a un rango [-1, 1]
])

# Crear conjuntos de datos de entrenamiento y validación
conjunto_entrenamiento = ImageDataset(
    dir_buenas='BuenasTrain',
    dir_rayadas='RayadasTrain',
    transformacion=transformacion
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Crear el DataLoader para el entrenamiento
cargador_entrenamiento = DataLoader(conjunto_entrenamiento, batch_size=tam_batch, shuffle=True, num_workers=tam_workers)

import os

# Rutas a los directorios de imágenes buenas y rayadas
directorio_buenas = '/home/rodrigo/Documentos/5to semestre/machin/BuenasTrain'
directorio_rayadas = '/home/rodrigo/Documentos/5to semestre/machin/RayadasTrain'

# Contar archivos en cada directorio
num_buenas = len(os.listdir(directorio_buenas))
num_rayadas = len(os.listdir(directorio_rayadas))

print(f'Número de imágenes buenas: {num_buenas}')
print(f'Número de imágenes rayadas: {num_rayadas}')

input()

modelo_discriminador = Discriminador().to(device)

# Función de pérdida y optimizador
criterio = nn.BCELoss()
optimizador = torch.optim.Adam(modelo_discriminador.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Función para obtener etiquetas reales y falsas
def etiquetas_reales_y_falsas(num):
    return torch.ones(num, 1).to(device), torch.zeros(num, 1).to(device)

intervalo_guardado = 5  # Guardar el modelo cada 5 épocas, por ejemplo


# Entrenamiento del discriminador
num_epochs = 50  # Número de épocas de entrenamiento


for epoch in range(num_epochs):
    correctas_total = 0
    imagenes_total = 0

    for i, (img_rayadas, img_buenas) in enumerate(cargador_entrenamiento):
        img_rayadas, img_buenas = img_rayadas.to(device), img_buenas.to(device)
        lote_tamaño = img_buenas.size(0)
        imagenes_total += lote_tamaño * 2  # Cuenta tanto las buenas como las rayadas

        # Etiquetas reales y falsas
        etiquetas_reales, etiquetas_falsas = etiquetas_reales_y_falsas(lote_tamaño)
        etiquetas_reales = etiquetas_reales.view(-1)  # Cambiar la forma de las etiquetas para que coincidan con la salida
        etiquetas_falsas = etiquetas_falsas.view(-1)

        # Entrenar con imágenes buenas (reales)
        optimizador.zero_grad()
        salida_real = modelo_discriminador(img_buenas).view(-1)
        perdida_real = criterio(salida_real, etiquetas_reales)
        perdida_real.backward()

        # Contar clasificaciones correctas para imágenes buenas
        predicciones_correctas = (salida_real > 0.5).float() == etiquetas_reales
        correctas_total += predicciones_correctas.sum().item()

        # Entrenar con imágenes rayadas (falsas)
        salida_falsa = modelo_discriminador(img_rayadas).view(-1)
        perdida_falsa = criterio(salida_falsa, etiquetas_falsas)
        perdida_falsa.backward()

        # Contar clasificaciones correctas para imágenes rayadas
        predicciones_correctas = (salida_falsa < 0.5).float() == etiquetas_falsas
        correctas_total += predicciones_correctas.sum().item()

        # Actualizar los pesos del discriminador
        optimizador.step()

    
    print(f'Epoch [{epoch+1}/{num_epochs}], Paso [{i+1}/{len(cargador_entrenamiento)}], '
                f'Clasificaciones Correctas: {correctas_total}/{imagenes_total}')
            
    # Guardar el modelo cada 'intervalo_guardado' épocas
    if (epoch + 1) % intervalo_guardado == 0:
        guardar_modelo(modelo_discriminador, '/home/rodrigo/Documentos/5to semestre/machin/modelosDISCR/', epoch + 1)

