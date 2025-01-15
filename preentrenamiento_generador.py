import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.models as models


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
        # Encoder
        x = self.hidden_enc1(x)
        x = self.hidden_enc2(x)
        x = self.hidden_enc3(x)

        # Bloques residuales
        x = self.res_blocks(x)

        # Decoder
        x = self.hidden_dec1(x)
        x = self.hidden_dec2(x)
        x = self.hidden_dec3(x)

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
        for param in self.vgg_submodel.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        # Asegúrate de que la entrada y el objetivo estén en el mismo dispositivo
        input = input.to(device)
        target = target.to(device)
        # Replicar los canales si es necesario
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        perception_loss = F.mse_loss(self.vgg_submodel(input), self.vgg_submodel(target))
        return perception_loss



class CombinedLoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_perceptual=0.01, device=None):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss(device)
        self.lambda_mse = lambda_mse
        self.lambda_perceptual = lambda_perceptual

    def forward(self, output, target):
        loss_mse = self.mse_loss(output, target)
        loss_perceptual = self.perceptual_loss(output, target)
        return self.lambda_mse * loss_mse + self.lambda_perceptual * loss_perceptual
    
    
def preentrenar_generador(modelo_generador, conjunto_datos, dispositivo, num_epochs=100, learning_rate=0.001, intervalo_visualizacion=10, intervalo_guardado=20, ruta_guardado="/home/rodrigo/Documentos/5to semestre/machin/modelosGEN/"):
    # Criterio y optimizador
    criterio = CombinedLoss(lambda_mse=1.0, lambda_perceptual=0.01, device=device)

    optimizador = optim.Adam(modelo_generador.parameters(), lr=learning_rate)
    
    # Cargador de datos
    cargador_datos = DataLoader(conjunto_datos, batch_size=16, shuffle=True)

    # Ciclo de entrenamiento
    modelo_generador.train()
    for epoch in range(num_epochs):
        perdida_total = 0

        for data in cargador_datos:
            # Solo necesitamos las imágenes buenas porque estamos replicándolas
            imagenes_rayadas, imagenes_buenas = data
            imagenes_buenas = imagenes_buenas.to(dispositivo)
            imagenes_rayadas = imagenes_rayadas.to(dispositivo)

            # Forward pass
            outputs = modelo_generador(imagenes_rayadas)
            perdida = criterio(outputs, imagenes_buenas)

            # Backward y optimización
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()

            perdida_total += perdida.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Pérdida: {perdida_total/len(cargador_datos)}')
        
        # Guardar el modelo cada 'intervalo_guardado' épocas
        if (epoch + 1) % intervalo_guardado == 0:
            guardar_modelo(modelo_generador, ruta_guardado, epoch+1)

        # Visualización de las imágenes
        if (epoch + 1) % intervalo_visualizacion == 0:
            with torch.no_grad():
                # Seleccionar una imagen aleatoria del conjunto de datos
                imagenes_rayada, imagen_buena = next(iter(cargador_datos))
                imagen_buena = imagen_buena.to(dispositivo)
                imagen_generada = modelo_generador(imagen_buena)

                visualizar_imagenes(imagenes_rayada, imagen_generada)

    return modelo_generador



# Hiperparametros
tam_batch = 32
tam_workers = 4
# Definición de los parámetros del entrenamiento
num_epochs = 160
visualization_interval = 10  # Visualizar cada 5 épocas, por ejemplo
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

conjunto_validacion = ImageDataset(
    dir_buenas='BuenasVal',
    dir_rayadas='RayadasVal',
    transformacion=transformacion
)

# Crear los cargadores de datos para entrenamiento y validación
cargador_entrenamiento = DataLoader(conjunto_entrenamiento, batch_size=tam_batch, shuffle=True, num_workers=tam_workers)
cargador_validacion = DataLoader(conjunto_validacion, batch_size=tam_batch, shuffle=False, num_workers=tam_workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

modelo_generador = Generador(ruta_pesos="combinada_epoch_160.pth").to(device)
modelo_generador_preentrenado = preentrenar_generador(modelo_generador, conjunto_entrenamiento, device, num_epochs=num_epochs, intervalo_visualizacion=visualization_interval)