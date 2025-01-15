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

import torch
import torch.nn as nn
import torch.optim as optim


class GANGame:
    def __init__(self, generador, discriminador, device):
        self.generador = generador
        self.discriminador = discriminador
        self.device = device
        self.historial_perdida_g = []
        self.historial_perdida_d = []


        # Inicializar optimizadores para ambos modelos
        self.optimizador_g = optim.Adam(self.generador.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizador_d = optim.Adam(self.discriminador.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterio_perceptual = CombinedLoss(lambda_mse=1.0, lambda_perceptual=0.01, device=self.device)

        # Criterio para calcular la pérdida
        self.criterio = nn.BCELoss()

    def entrenar_lote(self, imagenes_rayadas, imagenes_buenas):
        # Entrenar Discriminador
        self.discriminador.zero_grad()

        # Etiquetas reales y falsas
        etiquetas_reales = torch.ones((imagenes_buenas.size(0), 1), device=self.device)
        etiquetas_falsas = torch.zeros((imagenes_rayadas.size(0), 1), device=self.device)
        
        # Redimensionar las etiquetas objetivo para que coincidan
        etiquetas_reales = etiquetas_reales.view(-1)
        etiquetas_falsas = etiquetas_falsas.view(-1)

        # Entrenar con imágenes buenas (reales)
        salida_reales = self.discriminador(imagenes_buenas)
        salida_reales = salida_reales.view(salida_reales.size(0), -1).mean(1)  # Promedio para obtener [batch_size]

        perdida_d_real = self.criterio(salida_reales, etiquetas_reales)

        # Generar imágenes "limpias" a partir de las rayadas
        imagenes_generadas = self.generador(imagenes_rayadas)

        # Entrenar con imágenes generadas (falsas)
        salida_falsas = self.discriminador(imagenes_generadas.detach())
        salida_falsas = salida_falsas.view(salida_falsas.size(0), -1).mean(1)  # Promedio para obtener [batch_size]
                
        perdida_d_falsa = self.criterio(salida_falsas, etiquetas_falsas)

        # Backpropagation y optimización para el discriminador
        perdida_d = perdida_d_real + perdida_d_falsa
        perdida_d.backward()
        self.optimizador_d.step()

        # Entrenar Generador
        self.generador.zero_grad()

        # Generar salida para el discriminador con las imágenes generadas actualizadas
        salida_generador = self.discriminador(imagenes_generadas)
        salida_generador = salida_generador.view(salida_generador.size(0), -1).mean(1)

        # Calcular la pérdida del generador
        perdida_g = self.criterio_perceptual(imagenes_generadas, imagenes_buenas, salida_generador)  # Aquí usamos la CombinedLoss
        
        # Backpropagation y optimización para el generador
        perdida_g.backward()
        self.optimizador_g.step()

        return perdida_g.item(), perdida_d.item()

    def entrenar(self, cargador_datos, num_epochs, intervalo_visualizacion=5, intervalo_guardado=10, ruta_guardado="/home/rodrigo/Documentos/5to semestre/machin/modelosGAN/gan"):
        for epoch in range(num_epochs):
            self.criterio_perceptual.ajustar_lambdas(epoch, num_epochs)  # Ajuste dinámico de lambdas

            perdida_g_epoch = 0
            perdida_d_epoch = 0
            num_lotes = 0

            for i, (imagenes_rayadas, imagenes_buenas) in enumerate(cargador_datos):
                imagenes_rayadas, imagenes_buenas = imagenes_rayadas.to(self.device), imagenes_buenas.to(self.device)

                # Entrenar un lote
                perdida_g, perdida_d = self.entrenar_lote(imagenes_rayadas, imagenes_buenas)
                perdida_g_epoch += perdida_g
                perdida_d_epoch += perdida_d
                num_lotes += 1

                if i % 50 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Lote {i}, Pérdida D: {perdida_d}, Pérdida G: {perdida_g}')

            # Visualización y guardado de modelos
            if (epoch + 1) % intervalo_visualizacion == 0:
                with torch.no_grad():
                    self.visualizar_imagenes(imagenes_rayadas, self.generador(imagenes_rayadas), epoch + 1)

            if (epoch + 1) % intervalo_guardado == 0:
                self.guardar_modelos(ruta_guardado, epoch + 1)

            self.historial_perdida_g.append(perdida_g_epoch / num_lotes)
            self.historial_perdida_d.append(perdida_d_epoch / num_lotes)

        # Graficar las pérdidas después de cada época
        self.graficar_perdidas()
                
                
                    
    def guardar_modelos(self, ruta, epoch):
        torch.save(self.generador.state_dict(), f'{ruta}_generador_epoch_{epoch}.pth')
        torch.save(self.discriminador.state_dict(), f'{ruta}_discriminador_epoch_{epoch}.pth')
        print(f'Modelos guardados en epoch {epoch}')

    def visualizar_imagenes(self, img_rayada, img_generada, epoch):
        img_rayada = img_rayada[0].cpu().squeeze(0)
        img_generada = img_generada[0].cpu().squeeze(0)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img_rayada, cmap='gray')
        plt.title('Imagen Rayada')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img_generada, cmap='gray')
        plt.title('Imagen Limpiada')
        plt.axis('off')

        plt.suptitle(f'Epoch {epoch}')
        plt.show()


    def graficar_perdidas(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.historial_perdida_g, label='Pérdida Generador')
        plt.plot(self.historial_perdida_d, label='Pérdida Discriminador')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title('Pérdida durante el Entrenamiento')
        plt.legend()
        plt.show()



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



class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generador(nn.Module):
    def __init__(self, n_residual_blocks=16, ruta_pesos = None, tam_kernel=4):
        super(Generador, self).__init__()

        # Encoder: Capas de convolución para reducir la dimensión
        self.hidden_enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=tam_kernel, stride=2, padding=1),  # 500x500 -> 250x250
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.hidden_enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=tam_kernel, stride=2, padding=1),  # 250x250 -> 125x125
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.hidden_enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=tam_kernel, stride=2, padding=1),  # 125x125 -> 63x63
            nn.LeakyReLU(0.01, inplace=True)
        )

        # Bloques residuales en el 'cuello de botella'
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(n_residual_blocks)]
        )

        # Decoder: Capas transpuestas de convolución para aumentar la dimensión
        self.hidden_dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=tam_kernel, stride=2, padding=1, output_padding=1),  # 63x63 -> 128x128
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.hidden_dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=tam_kernel, stride=2, padding=1, output_padding=1),  # 128x128 -> 256x256
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.hidden_dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=tam_kernel, stride=2, padding=1, output_padding=1),  # 256x256 -> 512x512
            nn.Tanh()
        )
        
        if ruta_pesos is not None:
            self.cargar_pesos(ruta_pesos)
            
    
    def cargar_pesos(self, ruta_peso):
        # Cargar los pesos guardados
        self.load_state_dict(torch.load(ruta_peso, map_location=lambda storage, loc: storage))
        print(f'Pesos cargados desde: {ruta_peso}')
    

    def forward(self, x):
        # Envuelve las operaciones de cada capa en una función llamable para usar con checkpoint
        def run_hidden_enc1(x):
            return self.hidden_enc1(x)
        def run_hidden_enc2(x):
            return self.hidden_enc2(x)
        def run_hidden_enc3(x):
            return self.hidden_enc3(x)
        def run_hidden_dec1(x):
            return self.hidden_dec1(x)
        def run_hidden_dec2(x):
            return self.hidden_dec2(x)
        def run_res_block(block, x):
            return block(x)

        # Aplicar checkpoint a cada uno de los bloques definidos
        x = checkpoint(run_hidden_enc1, x)
        x = checkpoint(run_hidden_enc2, x)
        x = checkpoint(run_hidden_enc3, x)
        
        # Para los bloques residuales, aplicamos checkpoint a cada bloque individualmente
        for i, block in enumerate(self.res_blocks.children()):
            x = checkpoint(lambda x: block(x), x)
        
        x = checkpoint(run_hidden_dec1, x)
        x = checkpoint(run_hidden_dec2, x)
        x = self.hidden_dec3(x)  # No necesitamos checkpoint aquí si la memoria no es un problema

        # Redimensionamiento para garantizar las dimensiones correctas de la salida
        x = F.interpolate(x, size=(500, 500), mode='bilinear', align_corners=False)
        
        return x
    

def visualizar_imagenes(imagen_buena, imagen_generada):
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(imagen_buena[0].cpu().squeeze(0), cmap='gray')
    plt.title('Imagen Buena')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(imagen_generada[0].cpu().squeeze(0), cmap='gray')
    plt.title('Imagen Generada')
    plt.axis('off')

    plt.show()
    

def guardar_modelo(modelo, ruta, epoch):
    # Crear directorio si no existe
    directorio = os.path.dirname(ruta)
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    # Ruta del archivo con información de la época
    ruta_completa = f'{ruta}_epoch_{epoch}.pth'
    torch.save(modelo.state_dict(), ruta_completa)
    print(f'Modelo guardado en: {ruta_completa}')
    



class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:23]
        self.vgg_submodel = vgg.to(device)  # Asegúrate de mover VGG al dispositivo adecuado
        self.device = device
        for param in self.vgg_submodel.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        # Asegúrate de que la entrada y el objetivo estén en el mismo dispositivo
        input = input.to(self.device)
        target = target.to(self.device)
        # Replicar los canales si es necesario
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        perception_loss = F.mse_loss(self.vgg_submodel(input), self.vgg_submodel(target))
        return perception_loss



class CombinedLoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_perceptual=0.01, lambda_adv=0.001, device=None):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss(device)
        self.lambda_mse = lambda_mse
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adv = lambda_adv

    def forward(self, output, target, discriminator_output):
        loss_mse = self.mse_loss(output, target)
        loss_perceptual = self.perceptual_loss(output, target)
        loss_adv = -torch.log(discriminator_output).mean()  # Pérdida adversaria
        return self.lambda_mse * loss_mse + self.lambda_perceptual * loss_perceptual + self.lambda_adv * loss_adv

    def ajustar_lambdas(self, epoch, num_epochs):
        # Ajustar los pesos de las pérdidas a lo largo del entrenamiento
        if epoch < num_epochs // 2:
            # Mayor énfasis en la calidad de la imagen al principio
            self.lambda_mse = 1.0
            self.lambda_perceptual = 0.02  # Aumentar la importancia de la percepción
            self.lambda_adv = 0.001  # Menor énfasis en la pérdida adversaria
        else:
            # Equilibrar con la pérdida adversaria hacia la segunda mitad del entrenamiento
            self.lambda_adv = 0.005  # Aumentar gradualmente la importancia de la pérdida adversaria

    


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
        return x

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

# Instanciar el Generador y el Discriminador
generador = Generador(ruta_pesos="combinada_epoch_160.pth").to(device)
discriminador = Discriminador().to(device)

# Instanciar la clase GANGame con los modelos y el dispositivo
gan_game = GANGame(generador, discriminador, device)

# Iniciar el proceso de entrenamiento
gan_game.entrenar(cargador_entrenamiento, num_epochs)