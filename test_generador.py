import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    


def visualizar_imagenes(imagen_rayada, imagen_generada_1, imagen_generada_2, titulo1, titulo2):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Mostrar la imagen rayada
    axs[0].imshow(imagen_rayada.cpu().squeeze(0), cmap='gray')
    axs[0].set_title('Imagen Rayada')
    axs[0].axis('off')

    # Mostrar la imagen generada por el modelo 1
    axs[1].imshow(imagen_generada_1.cpu().squeeze(0), cmap='gray')
    axs[1].set_title(titulo1)
    axs[1].axis('off')

    # Mostrar la imagen generada por el modelo 2
    axs[2].imshow(imagen_generada_2.cpu().squeeze(0), cmap='gray')
    axs[2].set_title(titulo2)
    axs[2].axis('off')

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
    

def procesar_bloques(imagen, modelo, dispositivo):
    # Redimensionar la imagen a 1000x1000
    imagen_grande = F.interpolate(imagen.unsqueeze(0), size=(1000, 1000), mode='bilinear', align_corners=False)

    # Dividir en 4 bloques de 500x500
    bloques = [imagen_grande[..., :500, :500], imagen_grande[..., :500, 500:], 
            imagen_grande[..., 500:, :500], imagen_grande[..., 500:, 500:]]

    # Procesar cada bloque
    bloques_procesados = [modelo(bloque.to(dispositivo)).cpu() for bloque in bloques]

    # Reconstruir la imagen
    parte_superior = torch.cat(bloques_procesados[:2], dim=3)
    parte_inferior = torch.cat(bloques_procesados[2:], dim=3)
    imagen_completa = torch.cat([parte_superior, parte_inferior], dim=2).squeeze(0)

    # Redimensionar de vuelta a 500x500
    imagen_final = F.interpolate(imagen_completa.unsqueeze(0), size=(500, 500), mode='bilinear', align_corners=False)

    return imagen_final
    




if __name__ == "__main__":
    
    
    # Hiperparametros
    tam_batch = 32
    tam_workers = 4
    # Definición de los parámetros del entrenamiento
    num_epochs = 160
    visualization_interval = 10  # Visualizar cada 5 épocas, por ejemplo
    dims = 500
    
    titulo1 = "Modelo de época 100"
    
    titulo2 = "Modelo de época 160"
    
    # Definir transformaciones
    transformacion = transforms.Compose([
        transforms.Resize((dims, dims)),  # Redimensionar a 500x500
        transforms.ToTensor(),  # Convertir a tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizar a un rango [-1, 1]
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Cargar ambos modelos
    ruta_modelo_1 = '/home/rodrigo/Documentos/5toSemestre/Machine Learning/proyecto/ProyectoMachin/modelos/modelo_epoch_100.pth'
    ruta_modelo_2 = '/home/rodrigo/Documentos/5toSemestre/Machine Learning/proyecto/ProyectoMachin/modelos/modelo_epoch_160.pth'

    modelo_1 = Generador().to(device)
    modelo_1.load_state_dict(torch.load(ruta_modelo_1))
    modelo_1.eval()

    modelo_2 = Generador().to(device)
    modelo_2.load_state_dict(torch.load(ruta_modelo_2))
    modelo_2.eval()

    # Preparar los datos (usando el mismo conjunto de datos para ambos modelos)
    conjunto_prueba = ImageDataset(
        dir_buenas='TestBuenas',
        dir_rayadas='TestBuenas',
        transformacion=transformacion
    )
    cargador_prueba = DataLoader(conjunto_prueba, batch_size=tam_batch, shuffle=False, num_workers=tam_workers)

    # Ejecutar inferencias y comparar
    with torch.no_grad():
        for data in cargador_prueba:
            imagenes_rayadas, _ = data
            imagenes_rayadas = imagenes_rayadas.to(device)

            imagenes_generadas_1 = modelo_1(imagenes_rayadas)
            imagenes_generadas_2 = modelo_2(imagenes_rayadas)

            num_muestras = len(imagenes_rayadas)
            for j in range(num_muestras):
                """plt.figure()
                plt.imshow(imagenes_rayadas[j].cpu().squeeze(0), cmap='gray')
                plt.title('Imagen Rayada')
                plt.axis('off')
                plt.show()"""
                visualizar_imagenes(imagenes_rayadas[j] ,imagenes_generadas_1[j], imagenes_generadas_2[j], titulo1, titulo2)



