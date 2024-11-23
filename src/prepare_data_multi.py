import os
import random
import numpy as np
from osgeo import gdal
import torch
from collections import defaultdict
from torchvision import transforms
from shutil import copy2, rmtree
from torch.utils.data import Dataset, DataLoader

class MultispectralImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Cargar y preprocesar imágenes multiespectrales
        
        Args:
            root_dir (str): Directorio raíz con subdirectorios de clases
            transform (callable, optional): Transformaciones a aplicar
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Recopilar rutas de imágenes y etiquetas
        self.images = []
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                self.images.append((img_path, self.class_to_idx[cls]))

    def read_multispectral_image(self, image_path):
        """
        Leer imagen multiespectral usando GDAL
        
        Args:
            image_path (str): Ruta de la imagen
        
        Returns:
            numpy.ndarray: Array de imagen multiespectral
        """
        dataset = gdal.Open(image_path)
        bands = dataset.RasterCount

        # Leer todas las bandas
        image_array = np.stack([dataset.GetRasterBand(i+1).ReadAsArray() for i in range(bands)], axis=-1)
        
        # Convertir a tensor de PyTorch
        image_tensor = torch.from_numpy(image_array).float()
        
        # Normalización personalizada para imágenes multiespectrales
        image_tensor = (image_tensor - image_tensor.mean()) / image_tensor.std()
        
        return image_tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = self.read_multispectral_image(img_path)
        
        # Transformaciones adicionales si están definidas
        if self.transform:
            image = self.transform(image)
        
        return image, label

def balance_multispectral_dataset(input_dir, output_dir):
    """
    Balancear conjunto de datos multiespectrales
    
    Args:
        input_dir (str): Directorio de entrada con datos originales
        output_dir (str): Directorio de salida para datos balanceados
    """
    # Eliminar y crear directorio de salida
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.makedirs(output_dir)

    # Contar imágenes por clase
    class_counts = defaultdict(list)
    for cls in os.listdir(input_dir):
        cls_path = os.path.join(input_dir, cls)
        for img_name in os.listdir(cls_path):
            class_counts[cls].append(os.path.join(cls_path, img_name))

    # Encontrar número mínimo de imágenes
    min_images = min(len(imgs) for imgs in class_counts.values())

    # Balancear clases
    for cls, images in class_counts.items():
        # Crear directorio de clase en salida
        output_cls_dir = os.path.join(output_dir, cls)
        os.makedirs(output_cls_dir)

        # Seleccionar imágenes al azar
        selected_images = random.sample(images, min_images)
        
        # Copiar imágenes balanceadas
        for img_path in selected_images:
            copy2(img_path, output_cls_dir)

def prepare_multispectral_dataset(
    raw_data_dir="data/raw2", 
    train_dir="data/train", 
    validation_dir="data/validation"
):
    """
    Preprocesar conjunto de datos multiespectrales
    
    Args:
        raw_data_dir (str): Directorio con datos originales
        train_dir (str): Directorio de salida para entrenamiento
        validation_dir (str): Directorio de salida para validación
    """
    # Balancear conjuntos de datos
    balance_multispectral_dataset(
        os.path.join(raw_data_dir, "Train"), 
        train_dir
    )
    balance_multispectral_dataset(
        os.path.join(raw_data_dir, "Validation"), 
        validation_dir
    )

    print("Preprocesamiento de datos multiespectrales completado.")
