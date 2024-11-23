import torch
import torch.nn as nn
import yaml
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from osgeo import gdal
import os
from prepare_data_multi import prepare_multispectral_dataset
from resnet34 import train_resnet
from densenet121 import train_densenet
import csv

os.environ['CPL_LOG'] = 'OFF'
# Preparar dataset multiespectral
prepare_multispectral_dataset()

class MultispectralImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        
        # Get all classes (subdirectories)
        self.classes = [d for d in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, d))]
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image paths and their corresponding labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.tif') or img_name.endswith('.tiff'):  # Add more extensions if needed
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Read the multispectral image
        image = read_multispectral_image(img_path)

        # Convert image to PyTorch tensor and change dimensions order (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label
    
# Función para leer imágenes multiespectrales con GDAL
def read_multispectral_image(image_path):
    # Usamos GDAL para leer la imagen TIFF (supongamos que tiene 5 bandas)
    dataset = gdal.Open(image_path)
    bands = dataset.RasterCount  # Obtiene la cantidad de bandas

    # Asegúrate de que el número de bandas sea 5 (ajústalo si tienes más o menos)
    if bands != 5:
        raise ValueError(f"Se esperaba una imagen con 5 bandas, pero la imagen tiene {bands} bandas.")

    # Leer todas las bandas y convertirlas a un array de NumPy
    image_array = np.stack([dataset.GetRasterBand(i+1).ReadAsArray() for i in range(bands)], axis=-1)
    return image_array

# Cargar parámetros desde el archivo YAML
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

lr = params["learning_rate"]
num_epochs = params["epochs"]
batch_size = params["batch_size"]
save_weights_every = params["save_weights_every"]
model_params = params["model"]

architecture = model_params["architecture"]
pretrained = model_params["pretrained"]
transfer_learning = model_params["transfer_learning"]

# Crear modelo
if architecture == "ResNet34":
    model = models.resnet34(pretrained=pretrained)
elif architecture == "DenseNet121":
    model = models.densenet121(pretrained=pretrained)
else:
    raise ValueError(f"Modelo {architecture} no soportado")

# Modificar la primera capa de convolución para aceptar 5 canales en lugar de 3
model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
# Eliminar última capa y congelar pesos
edited_model = nn.Sequential(*list(model.children())[:-1])
for parameter in edited_model.parameters():
    parameter.requires_grad = False

# Transformaciones y datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionamos la imagen
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456], 
                         std=[0.229, 0.224, 0.225, 0.229, 0.224])  # Normalizamos para 5 bandas
])

train_dataset = MultispectralImageDataset("data/train", transform=transform)
valid_dataset = MultispectralImageDataset("data/validation", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

print(f"Número de tiles usados para entrenamiento: {len(train_dataset)}")
print(f"Número de tiles usados para validación: {len(valid_dataset)}")

# Guardar métricas en CSV
os.makedirs("metrics", exist_ok=True)
csv_file = r'metrics/training_metrics.csv'
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train_Loss", "Train_Accuracy", "Validation_Loss", "Validation_Accuracy"])

# Configurar dispositivo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Se utilizará {device} para el modelo\n")

# Entrenar el modelo
def train_model(model):
    if isinstance(model, models.resnet.ResNet):
        train_resnet(
            device, edited_model, len(train_dataset.classes), train_loader, valid_loader,
            train_dataset.classes, lr, num_epochs, csv_file, save_weights_every
        )
    elif isinstance(model, models.densenet.DenseNet):
        train_densenet(model)
    else:
        print("El modelo seleccionado no está configurado.")
        return

train_model(model)

from performance_graphs import grafic
grafic(csv_file)