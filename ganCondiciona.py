import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
import os




class Generator(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(Generator, self).__init__()
        
        # Encoder: Capas de convolución para reducir la dimensión
        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        # Añadir una capa adicional para mayor profundidad
        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        
        # Decoder: Capas transpuestas de convolución para aumentar la dimensión
        self.decoder_layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_rate)
        )
        self.decoder_layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )
        # Añadir una capa adicional para completar la reconstrucción
        self.decoder_layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # La salida debe estar entre -1 y 1 si se usa tanh en la última capa
        )

    def forward(self, x):
        e1 = self.encoder_layer1(x)
        e2 = self.encoder_layer2(e1)
        e3 = self.encoder_layer3(e2)
        
        d1 = self.decoder_layer1(e3)
        
        # Asegurarse de que d1 tiene las mismas dimensiones que e2 antes de la suma
        d1 = F.interpolate(d1, size=e2.size()[2:], mode='nearest') + e2
        
        d2 = self.decoder_layer2(d1)
        
        # Asegurarse de que d2 tiene las mismas dimensiones que e1 antes de la suma
        d2 = F.interpolate(d2, size=e1.size()[2:], mode='nearest') + e1
        
        d3 = self.decoder_layer3(d2)
        
        return d3



class Discriminator(nn.Module):
    def __init__(self,  dropout_rate=0.5):
        super(Discriminator, self).__init__()
        # Definición de las capas del discriminador con grupos de capas individuales
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # entrada: [1, 500, 500], salida: [64, 250, 250]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # salida: [128, 125, 125]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate)
        )
        # Continúa definiendo más capas si es necesario
        self.out_layer = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=125),  # Reducir a un solo valor
            nn.Sigmoid()  # Sigmoid para la probabilidad de salida
        )

    def forward(self, x):
        # Aplicar checkpointing a cada grupo de capas
        x = checkpoint(self.layer1, x, use_reentrant=True)
        x = checkpoint(self.layer2, x, use_reentrant=True)
        # Si hay más capas, continuar aplicando checkpointing aquí
        x = self.out_layer(x)
        return x.view(-1, 1).squeeze(1)  # Asegúrate de que la salida sea un tensor 1D para cada imagen en el batch


# Función de pérdida para el 'inpainting'
class InpaintingLoss(nn.Module):
    def __init__(self):
        super(InpaintingLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, fake_images, discriminator_outputs, real_images):
        # Pérdida L1 para la fidelidad de contenido
        content_loss = self.l1_loss(fake_images, real_images)

        # Pérdida adversaria basada en la salida del discriminador
        adversarial_loss = self.bce_loss(discriminator_outputs, torch.ones_like(discriminator_outputs))

        # Combinación de las pérdidas
        return content_loss + adversarial_loss

    

class CustomImageDataset(Dataset):
    def __init__(self, good_dir, corrupted_dir, transform=None):
        self.good_dir = good_dir
        self.corrupted_dir = corrupted_dir
        self.transform = transform

        # Asumimos que cada imagen 'buena' tiene su correspondiente imagen 'rayada' con el mismo nombre
        self.filenames = os.listdir(good_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        good_img_path = os.path.join(self.good_dir, self.filenames[idx])
        corrupted_img_path = os.path.join(self.corrupted_dir, self.filenames[idx])

        good_image = Image.open(good_img_path).convert('L')  # Convertir a escala de grises
        corrupted_image = Image.open(corrupted_img_path).convert('L')

        if self.transform:
            good_image = self.transform(good_image)
            corrupted_image = self.transform(corrupted_image)

        return corrupted_image, good_image
    
    
def validate_and_visualize(generator, val_loader, device, epoch, visualization_interval):
    if epoch % visualization_interval == 0:
        with torch.no_grad():
            generator.eval()

            for corrupted_images, real_images in val_loader:
                corrupted_images, real_images = corrupted_images.to(device), real_images.to(device)
                generated_images = generator(corrupted_images)

                # Visualizar las primeras imágenes del batch
                visualize_images(real_images, corrupted_images, generated_images, epoch)
                break  # Solo visualizamos un batch

def visualize_images(real_images, corrupted_images, generated_images, epoch):
    plt.figure(figsize=(12, 4))

    # Imagen Limpia
    plt.subplot(1, 3, 1)
    plt.imshow(real_images[0].cpu().squeeze(), cmap='gray')
    plt.title('Imagen Limpia')
    plt.axis('off')

    # Imagen Sucia
    plt.subplot(1, 3, 2)
    plt.imshow(corrupted_images[0].cpu().squeeze(), cmap='gray')
    plt.title('Imagen Sucia')
    plt.axis('off')

    # Imagen Generada
    plt.subplot(1, 3, 3)
    plt.imshow(generated_images[0].cpu().squeeze(), cmap='gray')
    plt.title('Imagen Generada')
    plt.axis('off')

    plt.show()
    
    
def save_model(model, model_name, epoch, save_interval, save_directory="./models"):
    if epoch % save_interval == 0:
        # Crear el directorio si no existe
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        file_path = f"{save_directory}/{model_name}_epoch_{epoch}.pth"
        torch.save(model.state_dict(), file_path)



def train_model(generator, discriminator, criterion, optimizer_G, optimizer_D, train_loader, val_loader, num_epochs, device, visualization_interval, save_interval):
    # Listas para almacenar las pérdidas a lo largo del entrenamiento
    losses_G = []
    losses_D = []

    for epoch in range(num_epochs):
        # Fase de entrenamiento
        generator.train()
        discriminator.train()

        running_loss_G = 0.0
        running_loss_D = 0.0

        for corrupted_images, real_images in train_loader:
            corrupted_images, real_images = corrupted_images.to(device), real_images.to(device)
            
            """print("Dimensiones de 'corrupted_images':", corrupted_images.shape)
            print("Dimensiones de 'real_images':", real_images.shape)"""

            # Entrenamiento del Discriminador
            optimizer_D.zero_grad()
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)


            fake_images = generator(corrupted_images)
            #print("Dimensiones de 'fake_images':", fake_images.shape)
            outputs_fake = discriminator(fake_images.detach())
            #print("Dimensiones de 'outputs_fake':", outputs_fake.shape)
            loss_fake = criterion(outputs_fake, fake_labels, real_images)
            
            outputs_real = discriminator(real_images)
            #print("Dimensiones de 'outputs_real':", outputs_real.shape)
            loss_real = criterion(fake_images, outputs_real, real_images)

            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # Entrenamiento del Generador
            optimizer_G.zero_grad()

            outputs_fake_for_G = discriminator(fake_images)
            loss_G = criterion(fake_images, outputs_fake_for_G, real_images)

            loss_G.backward()

            optimizer_G.step()

            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()

        # Calcular la pérdida promedio para esta época
        epoch_loss_G = running_loss_G / len(train_loader)
        epoch_loss_D = running_loss_D / len(train_loader)

        # Almacenar las pérdidas para su posterior visualización
        losses_G.append(epoch_loss_G)
        losses_D.append(epoch_loss_D)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss G: {epoch_loss_G:.4f}, Loss D: {epoch_loss_D:.4f}")

        # Validación y Visualización
        validate_and_visualize(generator, val_loader, device, epoch, visualization_interval)
        save_model(generator, "generator", epoch, save_interval, "./models")
        save_model(discriminator, "discriminator", epoch, save_interval, "./models")

    # Al final del entrenamiento, graficar las pérdidas
    plt.figure(figsize=(10, 5))
    plt.title("Pérdidas del Generador y Discriminador Durante el Entrenamiento")
    plt.plot(losses_G, label="Generador")
    plt.plot(losses_D, label="Discriminador")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.show()

    print('Entrenamiento finalizado.')


    

if __name__ == "__main__":
    
    # Hiperparametros
    tam_batch = 16
    tam_workers = 4
    # Definición de los parámetros del entrenamiento
    num_epochs = 50
    save_interval = 5 # Guardar el modelo cada 5 épocas, por ejemplo
    # Definición del intervalo para visualización
    visualization_interval = 5  # Visualizar cada 5 épocas, por ejemplo
    save_interval = 5
    
        
    # Definir transformaciones, podrías incluir transformaciones adicionales según sea necesario
    transform = transforms.Compose([
        transforms.Resize((500, 500)),  # Redimensionar a 500x500
        transforms.ToTensor(),  # Convertir a tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizar a un rango [-1, 1]
        ])


    # Crear conjuntos de datos de entrenamiento y validación
    train_dataset = CustomImageDataset(
        good_dir='BuenasTrain',
        corrupted_dir='RayadasTrain',
        transform=transform
    )

    val_dataset = CustomImageDataset(
        good_dir='BuenasVal',
        corrupted_dir='RayadasVal',
        transform=transform
    )

    # Crear los DataLoaders para entrenamiento y validación
    train_loader = DataLoader(train_dataset, batch_size=tam_batch, shuffle=True, num_workers=tam_workers)
    val_loader = DataLoader(val_dataset, batch_size=tam_batch, shuffle=False, num_workers=tam_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Inicialización de la red y optimizadores
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = InpaintingLoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.002)
    
    
    train_model(generator, discriminator, criterion, optimizer_G, optimizer_D, train_loader, val_loader, num_epochs, device, visualization_interval, save_interval)
    