import os
import cv2
import numpy as np
from PIL import Image

# Definir directorios de entrada y salida
input_original_image = r"C:\Users\messi\Desktop\Nuevo Dataset\Originales\Entrenamiento"  # Directorio con imágenes originales en RGB
input_segmented_image = r"C:\Users\messi\Desktop\Nuevo Dataset\Mascaras\Entrenamiento"  # Directorio con máscaras de segmentación
output_directory = r"C:\Users\messi\Desktop\Nuevo Dataset\Dataset\Entrenamiento"  # Directorio de salida para imágenes multiplicadas

# Iterar a través de los archivos en el directorio de imágenes originales
for file_name in os.listdir(input_original_image):
    # Construir rutas completas para las imágenes originales y segmentadas
    original_path = os.path.join(input_original_image, file_name)
    segmented_path = os.path.join(input_segmented_image, file_name.replace('.JPG', '.png'))
    
    # Leer imágenes y convertir de BGR a RGB
    image = cv2.cvtColor(cv2.imread(original_path), cv2.COLOR_BGR2RGB)
    imageseg = cv2.cvtColor(cv2.imread(segmented_path), cv2.COLOR_BGR2RGB)
    
    # Normalizar la imagen de segmentación (convertir valores de 0-255 a 0-1)
    imagenseg_norm = imageseg.astype(float) / 255.0
    
    # Convertir imágenes a arreglos numpy
    matriz = np.asarray(image)
    matrizseg = np.asanyarray(imagenseg_norm)
    
    # Multiplicar la imagen original por la máscara de segmentación 
    # Esto hace que los píxeles fuera de la región de interés sean negros
    multi = np.multiply(matrizseg, matriz)
    
    # Convertir el resultado de vuelta a un formato de imagen que se pueda guardar
    image_multi = Image.fromarray(multi.astype('uint8'))
    
    # Generar nombre de archivo de salida
    output_name = os.path.join(output_directory, '{}_seg.png'.format(os.path.splitext(file_name)[0]))
    
    # Guardar la imagen multiplicada
    image_multi.save(output_name)