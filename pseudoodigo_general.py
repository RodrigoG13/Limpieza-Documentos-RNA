import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Función de Carga de Imágenes
def cargar_imagenes(directorio):
    imagenes = []
    for archivo in os.listdir(directorio):
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            ruta = os.path.join(directorio, archivo)
            imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            imagenes.append(imagen)
    return imagenes

def preprocesar_imagenes(lista_imagenes, tamaño_deseado=(500, 500)):
    imagenes_procesadas = []
    for imagen in lista_imagenes:
        imagen = cv2.resize(imagen, tamaño_deseado)
        imagen = imagen / 255.0  # Normalización
        imagen = imagen.astype(np.float32)  # Cambia el tipo de datos a float32
        imagenes_procesadas.append(imagen)
    return imagenes_procesadas


class ImageDataset(Dataset):
    def __init__(self, img_buenas, img_rayadas, transform=None):
        self.imagenes_buenas = img_buenas
        self.imagenes_rayadas = img_rayadas
        self.transform = transform

    def __len__(self):
        return min(len(self.imagenes_buenas), len(self.imagenes_rayadas))

    def __getitem__(self, idx):
        imagen_buena = self.imagenes_buenas[idx]
        imagen_rayada = self.imagenes_rayadas[idx]

        if self.transform:
            imagen_buena = self.transform(imagen_buena)
            imagen_rayada = self.transform(imagen_rayada)

        return imagen_rayada, imagen_buena

# En cargar_datos_gan
"""def cargar_datos_gan(img_buenas, img_rayadas, batch_size=40):"""
def cargar_datos_gan(img_buenas, img_rayadas, batch_size=16):
    transformaciones = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((500, 500)),  # Asegúrate de que todas las imágenes tengan el mismo tamaño
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    dataset = ImageDataset(img_buenas, img_rayadas, transform=transformaciones)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Clases de Generador y Discriminador, funciones de entrenamiento, evaluación, guardado y carga del modelo, etc.

class Generador(nn.Module):
    def __init__(self):
        super(Generador, self).__init__()
        self.main = nn.Sequential(
            # Inicia con un canal de entrada (imagen en escala de grises) y expande gradualmente
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # Aquí, el tamaño de la imagen se ha reducido significativamente
            # Comienza el proceso de upscaling

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # La última capa reduce a un canal de salida
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # La salida aquí debería ser una imagen de 720x720
        )

    def forward(self, x):
        return self.main(x)



class Discriminador(nn.Module):
    def __init__(self):
        super(Discriminador, self).__init__()
        self.main = nn.Sequential(
            # Capa convolucional 1
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa convolucional 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa convolucional 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa convolucional 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa convolucional 5
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # Agregar más capas si es necesario para reducir aún más las dimensiones
            # ...

            # Pooling adaptativo para llevar las características a 1x1
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.final = nn.Sequential(
            # La última capa convolucional que lleva a una salida de un solo valor (decisión real o falso)
            nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        x = self.final(x)
        # Aplanar el resultado para obtener un vector de salida con un valor por imagen
        return x.view(-1)



def crear_gan():
    generador = Generador()
    discriminador = Discriminador()
    return generador, discriminador


def desnormalizar(imagen):
    # Desnormaliza usando los mismos valores que usaste para la normalización
    imagen = imagen * 0.5 + 0.5
    imagen = imagen.clamp(0, 1)  # Asegúrate de que los valores estén entre 0 y 1
    return imagen



def entrenar_gan(generador, discriminador, dataloader, epochs, device, intervalo_visualizacion=10000):
    # Definir optimizadores y criterio de pérdida
    optimizador_g = optim.Adam(generador.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizador_d = optim.Adam(discriminador.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterio = nn.BCELoss()

    generador.to(device)
    discriminador.to(device)

    for epoch in range(epochs):
        imagenes_por_epoca = 0

        for i, (imagenes_rayadas, imagenes_buenas) in enumerate(dataloader):
            # Mover los datos al dispositivo (GPU o CPU)
            imagenes_rayadas, imagenes_buenas = imagenes_rayadas.to(device), imagenes_buenas.to(device)
            
            # Actualiza el contador de imágenes por época
            """print(imagenes_por_epoca)
            print(imagenes_rayadas.size(0))"""
            imagenes_por_epoca += imagenes_rayadas.size(0)

            # ====================
            # Actualizar Discriminador
            # ====================
            optimizador_d.zero_grad()

            # Entrenar con imágenes reales
            etiquetas_reales = torch.ones(imagenes_buenas.size(0), 1).to(device).view(-1)
            salida_real = discriminador(imagenes_buenas).view(-1)
            perdida_d_real = criterio(salida_real, etiquetas_reales)

            # Entrenar con imágenes falsas generadas
            z = torch.randn(imagenes_rayadas.size(0), 100, 1, 1).to(device)  # Vector de ruido
            imagenes_falsas = generador(imagenes_rayadas)
            etiquetas_falsas = torch.zeros(imagenes_rayadas.size(0), 1).to(device).view(-1)
            salida_falsa = discriminador(imagenes_falsas.detach()).view(-1)
            perdida_d_falsa = criterio(salida_falsa, etiquetas_falsas)

            # Calcular la pérdida total y actualizar
            perdida_d = (perdida_d_real + perdida_d_falsa) / 2
            perdida_d.backward()
            optimizador_d.step()

            # ====================
            # Actualizar Generador
            # ====================
            optimizador_g.zero_grad()

            # Generar imágenes y calcular la pérdida con las etiquetas "reales"
            salida_falsa = discriminador(imagenes_falsas).view(-1)
            perdida_g = criterio(salida_falsa, etiquetas_reales)

            # Retropropagar y actualizar
            perdida_g.backward()
            optimizador_g.step()
            #print(i)

            if i % 50 == 0:  # Ajustar según la frecuencia deseada para la impresión
                print(f"Epoch [{epoch+1}/{epochs}], Paso [{i}/{len(dataloader)}], "
                    f"Pérdida D: {perdida_d.item()}, Pérdida G: {perdida_g.item()}")

            # Visualización periódica de las imágenes generadas
            if i % intervalo_visualizacion == 0:
                generador.eval()  # Modo de evaluación para la generación de imágenes
                with torch.no_grad():
                    # Usa 'imagenes_rayadas' en lugar de 'z' para generar las imágenes falsas
                    imagenes_falsas = generador(imagenes_rayadas)
                    if epoch % 20 == 0:
                        # Desnormalizar las imágenes
                        imagen_rayada = desnormalizar(imagenes_rayadas[0]).cpu().squeeze()
                        imagen_falsa = desnormalizar(imagenes_falsas[0]).cpu().squeeze()
                        imagen_buena = desnormalizar(imagenes_buenas[0]).cpu().squeeze()
                        
                        # Imprimir el tamaño de las imágenes
                        print(f"Tamaño de la imagen rayada: {imagen_rayada.shape}")
                        print(f"Tamaño de la imagen generada: {imagen_falsa.shape}")
                        print(f"Tamaño de la imagen buena: {imagen_buena.shape}")
                        
                        # Mostrar imágenes
                        plt.figure(figsize=(10, 4))
                        plt.subplot(1, 3, 1)
                        plt.imshow(imagen_rayada, cmap='gray')
                        plt.title("Rayada")
                        plt.axis('off')

                        plt.subplot(1, 3, 2)
                        plt.imshow(imagen_falsa, cmap='gray')
                        plt.title("Generada")
                        plt.axis('off')

                        plt.subplot(1, 3, 3)
                        plt.imshow(imagen_buena, cmap='gray')
                        plt.title("Buena")
                        plt.axis('off')

                        plt.show()
                generador.train()  # Volver al modo de entrenamiento

        # Al final de cada época, imprime el número total de imágenes utilizadas
        #print(f"Época {epoch + 1}/{epochs} completada. Número total de imágenes procesadas: {imagenes_por_epoca}")
        #input()
    print("Entrenamiento completado.")



def evaluar_modelo(generador, dataloader, device):
    generador.eval()
    mse_total = 0.0
    ssim_total = 0.0
    total_imagenes = 0

    with torch.no_grad():
        for i, (imagenes_rayadas, imagenes_buenas) in enumerate(dataloader):
            imagenes_rayadas = imagenes_rayadas.to(device)
            imagenes_buenas = imagenes_buenas.to(device)
            imagenes_generadas = generador(imagenes_rayadas)

            mse = torch.nn.functional.mse_loss(imagenes_generadas, imagenes_buenas)
            mse_total += mse.item()

            # Calculando SSIM
            # Convertir las imágenes a formato numpy para calcular SSIM
            img_gen_np = imagenes_generadas.cpu().numpy().transpose(0, 2, 3, 1)
            img_real_np = imagenes_buenas.cpu().numpy().transpose(0, 2, 3, 1)
            ssim_val = np.mean([ssim(img_real, img_gen, multichannel=True) 
                                for img_real, img_gen in zip(img_real_np, img_gen_np)])
            ssim_total += ssim_val

            total_imagenes += 1

            # Visualización de las primeras N imágenes
            if i < 5:  # Ajustar este número según sea necesario
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(imagenes_rayadas[0].cpu().permute(1, 2, 0))
                plt.title("Rayada")
                plt.subplot(1, 3, 2)
                plt.imshow(imagenes_generadas[0].cpu().permute(1, 2, 0))
                plt.title("Generada")
                plt.subplot(1, 3, 3)
                plt.imshow(imagenes_buenas[0].cpu().permute(1, 2, 0))
                plt.title("Buena")
                plt.show()

    mse_promedio = mse_total / total_imagenes
    ssim_promedio = ssim_total / total_imagenes
    print(f"MSE Promedio: {mse_promedio}")
    print(f"SSIM Promedio: {ssim_promedio}")

def guardar_modelo(modelo, ruta):
    torch.save(modelo.state_dict(), ruta)

def cargar_modelo(modelo, ruta):
    modelo.load_state_dict(torch.load(ruta))
    return modelo


if __name__ == "__main__":
    # Define las rutas a tus directorios de imágenes
    directorio_buenas = "Buenas"
    directorio_rayadas = "Rayadas"

    # Preprocesar las imágenes
    imagenes_buenas = preprocesar_imagenes(cargar_imagenes(directorio_buenas))
    imagenes_rayadas = preprocesar_imagenes(cargar_imagenes(directorio_rayadas))

    # Define la ruta donde quieres guardar los modelos
    ruta_generador = "generador.pth"
    ruta_discriminador = "discriminador.pth"

    # Configura el dispositivo para usar la GPU si está disponible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Imprime un mensaje sobre el dispositivo que se está utilizando
    if device.type == 'cuda':
        print(f"Usando GPU para el entrenamiento: {torch.cuda.get_device_name(device)}")
    else:
        print("Usando CPU para el entrenamiento")

    # Crea instancias del generador y del discriminador
    generador, discriminador = crear_gan()

    # Carga los datos usando la función cargar_datos_gan
    dataloader = cargar_datos_gan(imagenes_buenas, imagenes_rayadas)

    # Entrena la GAN
    entrenar_gan(generador, discriminador, dataloader, 400, device)  # Ajusta el número de épocas según sea necesario

    # Evaluar el modelo
    print("Evaluando el modelo...")
    evaluar_modelo(generador, dataloader, device)

    # Guarda los modelos entrenados
    guardar_modelo(generador, ruta_generador)
    guardar_modelo(discriminador, ruta_discriminador)

    print("Modelos entrenados y evaluados guardados exitosamente.")