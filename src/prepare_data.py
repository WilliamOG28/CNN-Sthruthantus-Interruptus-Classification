# Importaciones de librerías necesarias
import os  # Para operaciones con directorios y rutas
import random  # Para selección aleatoria de imágenes
from collections import defaultdict  # Para conteo de imágenes por clase
from torchvision.datasets import ImageFolder  # Para cargar conjuntos de datos de imágenes
from torchvision import transforms  # Para transformaciones de imágenes
from shutil import copy2  # Para copiar archivos

# Definición de rutas de directorios para datos
RAW_DATA_DIR = "data/raw"  # Directorio de datos originales
TRAIN_DIR = "data/train"  # Directorio para datos de entrenamiento
VALIDATION_DIR = "data/validation"  # Directorio para datos de validación

# Configuración de transformaciones para preprocesar imágenes
transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensiona todas las imágenes a 224x224 píxeles
    transforms.ToTensor(),  # Convierte imágenes a tensores de PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización estándar
])

def balance_dataset(dataset: ImageFolder, output_dir: str):
    """
    Función para balancear un conjunto de datos reduciendo el número de imágenes en clases grandes.

    Pasos:
    1. Contar imágenes por clase
    2. Encontrar número mínimo de imágenes en una clase
    3. Seleccionar muestras aleatorias para igualar el número mínimo
    4. Guardar imágenes balanceadas en nuevo directorio
    """
    print(f"Procesando datos para: {output_dir}")
    
    # Contar imágenes por clase
    class_counts = defaultdict(int)
    for _, label in dataset.samples:
        class_counts[dataset.classes[label]] += 1

    min_images = min(class_counts.values())  # Tamaño de la clase más pequeña

    # Balancear las clases
    balanced_samples = []
    for folder, num_images in class_counts.items():
        # Obtener rutas de imágenes para cada clase
        folder_paths = [path for path, label in dataset.samples if dataset.classes[label] == folder]
        
        # Seleccionar imágenes al azar para igualar la clase más pequeña
        selected_images = random.sample(folder_paths, min_images) if num_images > min_images else folder_paths
        
        # Agregar imágenes seleccionadas con sus etiquetas
        balanced_samples += [(img, dataset.class_to_idx[folder]) for img in selected_images]

    # Guardar imágenes balanceadas
    os.makedirs(output_dir, exist_ok=True)
    for img_path, label in balanced_samples:
        # Crear subdirectorios para cada clase
        class_folder = os.path.join(output_dir, dataset.classes[label])
        os.makedirs(class_folder, exist_ok=True)
        
        # Copiar imagen al nuevo directorio
        copy2(img_path, class_folder)

    print(f"Datos balanceados guardados en: {output_dir}")

def prepare_dataset():
    """
    Función principal para preprocesar datos:
    1. Cargar datos originales
    2. Balancear conjunto de entrenamiento
    3. Balancear conjunto de validación
    """
    # Cargar datos sin procesar
    raw_dataset = ImageFolder(RAW_DATA_DIR, transform=transforms)
    
    # Balancear datos de entrenamiento
    train_dataset = ImageFolder(os.path.join(RAW_DATA_DIR, "Train"), transform=transforms)
    balance_dataset(train_dataset, TRAIN_DIR)

    # Balancear datos de validación
    validation_dataset = ImageFolder(os.path.join(RAW_DATA_DIR, "Validation"), transform=transforms)
    balance_dataset(validation_dataset, VALIDATION_DIR)

    print("Preprocesamiento completado.")
