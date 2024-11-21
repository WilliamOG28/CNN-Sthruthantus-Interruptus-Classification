# Importar bibliotecas necesarias para procesamiento de imágenes, operaciones de archivos, manejo de JSON y marcas de tiempo
from PIL import Image
import os
import json
from datetime import datetime

# Definir directorios de entrada, salida y salida de JSON para el conjunto de datos de vegetación
input_directory = r"C:/Users/messi/Desktop/Nuevo_Dataset/Elaboracion_Dataset/Multiplicadas/Vegetacion/Validacion"
output_directory = r"C:/Users/messi/Desktop/Nuevo_Dataset/Elaboracion_Dataset/Dataset/Vegetacion/Validation"
json_output_directory = r"C:/Users/messi/Desktop/Nuevo_Dataset/Elaboracion_Dataset/Json/Vegetacion/Validacion"

# Crear directorio de salida de JSON si no existe
if not os.path.exists(json_output_directory):
    os.makedirs(json_output_directory)

def get_black_and_info_percentage(imagen, umbral=10):
    """
    Calcula el porcentaje de píxeles negros y no negros en una imagen.
    
    Argumentos:
        imagen (str): Ruta del archivo de imagen
        umbral (int, opcional): Umbral para determinar píxel negro. Por defecto 10.
    
    Retorna:
        tuple: Porcentaje de píxeles negros y no negros
    """
    with Image.open(imagen) as img:
        width, height = img.size
        black_pixels = 0
        total_pixels = width * height

        # Contar píxeles por debajo del umbral en todos los canales de color como negros
        for x in range(width):
            for y in range(height):
                pixel = img.getpixel((x, y))
                if all(componente <= umbral for componente in pixel):
                    black_pixels += 1

        # Calcular porcentajes
        black_percentage = (black_pixels / total_pixels) * 100
        info_percentage = 100 - black_percentage  # Porcentaje de píxeles no negros
        return black_percentage, info_percentage

def get_perspective(filename):
    """
    Determinar la perspectiva de la imagen basándose en el nombre del archivo.
    
    Argumentos:
        filename (str): Nombre del archivo de imagen
    
    Retorna:
        str: Tipo de perspectiva (Ortogonal, Oblicua o desconocida)
    """
    if "ot1" in filename:
        return "Ortogonal"
    elif "ob3" in filename:
        return "Oblicua"
    else:
        return "desconocida"

# Tamaño del tile para segmentación de imagen
tile_size = 16

# Contadores para rastrear imágenes procesadas
read_images_counter = 0
tiles_images_counter = 0
file_names = []

# Contar imágenes de entrada
for filename in os.listdir(input_directory):
    if filename.endswith(".png"):
        read_images_counter += 1
        file_names.append(filename)

print(f"Antes del recorte, hay {read_images_counter} imágenes en la carpeta.")

# Bucle principal de procesamiento de imágenes
for filename in file_names:
    # Abrir la imagen original
    image = Image.open(os.path.join(input_directory, filename))
    width, height = image.size

    # Crear tiles (segmentos de 16x16 píxeles) de la imagen
    for i in range(0, width, tile_size):
        for j in range(0, height, tile_size):
            # Recortar una región de 16x16 píxeles
            box = (i, j, i + tile_size, j + tile_size)
            region = image.crop(box)

            # Generar nombre de archivo único para el tile
            output_filename = filename.replace('.png', f'{i}_{j}.png')
            output_path = os.path.join(output_directory, output_filename)
            region.save(output_path)

            # Analizar porcentaje de información de píxeles
            black_percentage, info_percentage = get_black_and_info_percentage(output_path)

            # Procesar tiles con información significativa (>90% píxeles no negros)
            if info_percentage > 90:
                # Preparar metadatos para JSON
                contiene_muerdago = "No"
                especie_muerdago = "Struthanthus_Interruptus" if contiene_muerdago == "Si" else None
                fecha_hora_elaboracion = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                perspectiva = get_perspective(filename)
                coordenadas = f"({i}, {j})"
                latitud = ""
                longitud = ""
                fecha_hora_captura = ""
                grado_infestacion = ""
                clase = "Vegetacion que no pertenece a muerdago"

                # Crear diccionario de metadatos para JSON
                metadata = {
                    "ID de la imagen origen": filename,
                    "ID del recorte": output_filename,
                    "Clase": clase,
                    "Contiene Muerdago": contiene_muerdago,
                    "Fecha y hora de captura": fecha_hora_captura,
                    "Latitud": latitud,
                    "Longitud": longitud,
                    "Fecha y hora de elaboracion del recorte": fecha_hora_elaboracion,
                    "Perspectiva": perspectiva,
                    "Coordenada en la imagen": coordenadas,
                }

                # Agregar metadatos específicos de muérdago si está presente
                if contiene_muerdago == "Si":
                     metadata["Especie de Muerdago"] = "Struthanthus_Interruptus"
                     metadata["Porcentaje de Muerdago"] = info_percentage
                     metadata["Grado de infestacion"] = grado_infestacion

                # Guardar metadatos como JSON
                json_filename = output_filename.replace('.png', '.json')
                json_path = os.path.join(json_output_directory, json_filename)
                with open(json_path, 'w') as json_file:
                    json.dump(metadata, json_file, indent=4)

                print(f"Recorte guardado: {output_filename}, JSON creado: {json_filename}")
                tiles_images_counter += 1
            else:
                # Eliminar tiles con información insuficiente
                os.remove(output_path)
                print(f"Imagen eliminada: {output_filename} (Porcentaje de informacion: {info_percentage}%)")

# Resumen final de imágenes procesadas
print(f"Se recortaron y guardaron {tiles_images_counter} imágenes.")