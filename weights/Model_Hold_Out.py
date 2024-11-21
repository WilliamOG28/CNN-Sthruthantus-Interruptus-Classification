#%% Importar las bibliotecas necesarias
import matplotlib.pyplot as plt  # Importa la biblioteca para visualización de datos.
import torch  # Importa PyTorch, una biblioteca popular para aprendizaje profundo.
import torch.nn as nn  # Importa el módulo de redes neuronales de PyTorch.
from torchvision.datasets import ImageFolder  # Importa el conjunto de datos ImageFolder de torchvision.
from torchvision import transforms  # Importa funciones para realizar transformaciones en imágenes.
from torch.utils.data import DataLoader  # Importa DataLoader para cargar datos de manera eficiente.
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score  # Importa métricas comunes para evaluar modelos.
import time  # Importa el módulo time para medir el tiempo de ejecución.
import torchvision.models as models  # Importa modelos preentrenados de torchvision.
import seaborn as sns  # Importa seaborn para visualización de datos estadísticos.
import random  # Importa la biblioteca random para generar números aleatorios.
from collections import defaultdict # Importa la clase defaultdict del módulo collections
import json
import csv
from src.validation import validate_model
#%%
csv_file = r'C:/Users/messi/Desktop/Proyecto_Muerdago/CSV/training_metrics.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'])

# Ingresar el modelo a entrenar, anexo la liga para consultar los modelos que tiene la librar��a pytorch"https://pytorch.org/vision/stable/models.html se debe reemplazar donde dice resnet34"
model = models.resnet34(pretrained=True)# La opci�n pretrained, si es True sirve para descargar el modelo con los pesos entrenados por defecto y si es False para descargar unicamente el modelo sin alg�n entrenamiento previo.
#%% Definir la transformaci�n de datos para los modelos
transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensiona las im�genes a 224x224 p��xeles
    transforms.ToTensor(),  # Convierte las im�genes a tensores para poder ser utilizadas en la librer��a Pytorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) # Normaliza los valores de los p��xeles
#%%Cargar el conjunto de datos para entrenamientos 
#%%Cargar el conjunto de datos para entrenamientos 
train_dataset = ImageFolder(r"C:/Users/messi/Desktop/Proyecto_Muerdago/data/Train", transform=transforms)
valid_dataset = ImageFolder(r"C:/Users/messi/Desktop/Proyecto_Muerdago/data/Validation", transform=transforms)

# Balanceo para dataset de entrenamiento
train_classes = train_dataset.classes
train_folder_image_count = defaultdict(int)
classes = train_dataset.classes
# Cuenta el n�mero de clases en el conjunto de datos
num_classes = len(classes)

# Contar imágenes en cada clase
for _, label in train_dataset.samples:
    train_folder_image_count[train_classes[label]] += 1
min_train_images = min(train_folder_image_count.values())

# Balancear las clases en entrenamiento
train_balanced_samples = []
for folder, num_images in train_folder_image_count.items():
    folder_paths = [path for path, label in train_dataset.samples if train_classes[label] == folder]
    if num_images > min_train_images:
        selected_images = random.sample(folder_paths, min_train_images)
    else:
        selected_images = folder_paths
    train_balanced_samples += [(img, train_classes.index(folder)) for img in selected_images]

train_balanced_dataset = ImageFolder(root=train_dataset.root, transform=transforms)
train_balanced_dataset.samples = train_balanced_samples

# Verificación de balanceo de entrenamiento
train_labels_count = defaultdict(int)
for _, label in train_balanced_dataset.samples:
    train_labels_count[train_classes[label]] += 1
print("Cantidad de imágenes por categoría en el conjunto de datos de entrenamiento balanceado:")
print(train_labels_count)

# Balanceo para dataset de validación
valid_classes = valid_dataset.classes
valid_folder_image_count = defaultdict(int)

# Contar imágenes en cada clase
for _, label in valid_dataset.samples:
    valid_folder_image_count[valid_classes[label]] += 1
min_valid_images = min(valid_folder_image_count.values())

# Balancear las clases en validación
valid_balanced_samples = []
for folder, num_images in valid_folder_image_count.items():
    folder_paths = [path for path, label in valid_dataset.samples if valid_classes[label] == folder]
    if num_images > min_valid_images:
        selected_images = random.sample(folder_paths, min_valid_images)
    else:
        selected_images = folder_paths
    valid_balanced_samples += [(img, valid_classes.index(folder)) for img in selected_images]

valid_balanced_dataset = ImageFolder(root=valid_dataset.root, transform=transforms)
valid_balanced_dataset.samples = valid_balanced_samples

# Verificación de balanceo de validación
valid_labels_count = defaultdict(int)
for _, label in valid_balanced_dataset.samples:
    valid_labels_count[valid_classes[label]] += 1
print("Cantidad de imagenes por categoría en el conjunto de datos de validación balanceado:")
print(valid_labels_count)

print(f"Numero de imagenes para entrenamiento: {len(train_balanced_dataset)}")
print(f"Numero de imagenes para validacion: {len(valid_balanced_dataset)}")

#%%Modelo
# Eliminamos la �ltima capa de salida del modelo para m�s adelante poderla editar de acuerdo a nuestros objetivos.
edited_model = nn.Sequential(*list(model.children())[:-1])
# Desactivar el entrenamiento para cada par�metro del modelo 
for i, parameter in enumerate(edited_model.parameters()):
    parameter.requires_grad = False # Esta funci�n se encarga de conservar los pesos por defecto y no seguir entrenandolos, para evitar un coste computacional alto y centrarse a entrenar la �ltima capa Softmax que editaremos
#%%Método para la validación del modelo
def validate_model(Train_Model, valid_loader, criterion, device):
    # Inicializa las variables para llevar el conteo de las predicciones correctas y el total de predicciones
    correct_val = 0
    total_val = 0
    # Inicializa la variable para llevar el seguimiento del loss promedio durante la validación
    running_loss = 0
    # Deshabilita el cálculo de gradientes durante la validación
    with torch.no_grad():
        # Itera sobre el conjunto de datos de validación
        for i, data in enumerate(valid_loader):
            # Obtiene las entradas y los objetivos del batch actual
            inputs, targets = data
            # Envía las entradas y los objetivos al dispositivo (CPU o GPU)
            inputs, targets = inputs.to(device), targets.to(device)
            # Propaga las entradas a través de la red neuronal (forward pass)
            outputs = Train_Model(inputs)
            # Calcula el loss entre las salidas predichas y los objetivos reales
            loss = criterion(outputs, targets)
            # Suma el loss del batch actual al loss acumulado
            running_loss += loss.item()
            # Calcula las predicciones maximizando las salidas de la red
            _, predicted = torch.max(outputs, 1)
            # Actualiza el contador del total de predicciones
            total_val += targets.size(0)
            # Actualiza el contador de predicciones correctas sumando los casos donde las predicciones coinciden con los objetivos
            correct_val += (predicted == targets).sum().item()
        # Calcula la precisión de la validación y la agrega a la lista de precisión de validación
        valid_accuracy.append(100 * correct_val / total_val)
        # Calcula el loss promedio de la validación y lo agrega a la lista de loss de validación
        valid_loss.append(running_loss / len(valid_loader))
        # Devuelve los valores de loss y precisión de la validación
        return valid_loss, valid_accuracy
#%%Función para calcular métricas de la matriz de confusión
def metrics(Train_Model, valid_loader, classes):
    # Inicializa contadores
    correct = 0
    total = 0
    # Listas para almacenar etiquetas reales y predicciones
    all_labels = []
    all_predictions = []
    # Desactiva el cálculo del gradiente para la validación
    with torch.no_grad():
        # Itera sobre los datos de validación
        for data in valid_loader:
            # Obtiene las imágenes y etiquetas
            images, labels = data
            # Mueve las imágenes y etiquetas al dispositivo (GPU si está disponible)
            images = images.to(device)
            labels = labels.to(device)
            # Realiza las predicciones con la red neuronal
            outputs = Train_Model(images)
            # Obtiene el índice de la clase con mayor probabilidad
            _, predicted = torch.max(outputs.data, 1)
            # Actualiza los contadores de precisión
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Guarda las etiquetas reales y predicciones
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    # Calcula la matriz de confusión y métricas de evaluación
    cm = confusion_matrix(all_labels, all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    # Grafica la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Class')
    plt.ylabel('Real Class')
    plt.title('Confusion Matrix')
    plt.savefig(r'C:/Users/messi/Desktop/Proyecto_Muerdago/Plots/ResNet_CM.png')
    plt.show()
    # Imprime las métricas de evaluación
    print(f"Accuracy: {accuracy}\n")
    print(f"Precision: {precision}\n")
    print(f"Recall: {recall}\n")
    print(f"F1 Score: {f1}\n")
    # Devuelve las métricas de evaluación
    return accuracy, precision, recall, f1
#%%Función para graficar métricas
def grafic(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list):
    # Configura el tamaño de la figura y los subgráficos
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    # Encuentra la longitud mínima de las listas de pérdida
    min_len_loss = min(len(train_loss_list), len(valid_loss_list))
    # Grafica las curvas de pérdida de entrenamiento y validación hasta la longitud mínima
    plt.plot(range(1, min_len_loss + 1), train_loss_list[:min_len_loss], label='Training Loss')
    plt.plot(range(1, min_len_loss + 1), valid_loss_list[:min_len_loss], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Configura el segundo subgráfico
    plt.subplot(1, 2, 2)
    # Encuentra la longitud mínima de las listas de precisión
    min_len_acc = min(len(train_accuracy_list), len(valid_accuracy_list))
    # Grafica las curvas de precisión de entrenamiento y validación hasta la longitud mínima
    plt.plot(range(1, min_len_acc + 1), train_accuracy_list[:min_len_acc], label='Training accuracy')
    plt.plot(range(1, min_len_acc + 1), valid_accuracy_list[:min_len_acc], label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(r'C:/Users/messi/Desktop/Proyecto_Muerdago/Plots/ResNet_Plot.png')
    # Muestra la figura con los subgráficos
    plt.show()
#%% Función para calcular la precisión por clase
def accuracy_for_class(Train_Model, valid_loader, classes, device, output_file=r'C:/Users/messi/Desktop/Proyecto_Muerdago/Plots/accuracy_class.json'):
    # Inicializa listas para contar clasificaciones correctas y totales por clase
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    
    # Mueve el modelo a la GPU si est� disponible
    Train_Model.to(device)
    
    # Desactiva el c�lculo del gradiente para la validaci�n
    with torch.no_grad():
        # Itera sobre los datos de validaci�n
        for data in valid_loader:
            # Obtiene las im�genes y etiquetas
            images, labels = data
            # Mueve las im�genes y etiquetas al dispositivo (GPU si est� disponible)
            images = images.to(device)
            labels = labels.to(device)
            # Realiza las predicciones con el modelo
            outputs = Train_Model(images)
            _, predicted = torch.max(outputs, 1)
            # Calcula las clasificaciones correctas por clase
            correct = (predicted == labels)
            # Actualiza las listas de clasificaciones correctas y totales por clase
            for i in range(len(classes)):
                class_correct[i] += (correct * (labels == i)).sum().item()
                class_total[i] += (labels == i).sum().item()
    
    # Crea un diccionario con las precisiones por clase
    accuracy_dict = {}
    for i in range(len(classes)):
        accuracy = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        accuracy_dict[classes[i]] = {
            "Correct": class_correct[i],
            "Total": class_total[i],
            "Accuracy": accuracy
        }
    
    # Guarda el diccionario en un archivo JSON
    with open(output_file, 'w') as json_file:
        json.dump(accuracy_dict, json_file, indent=4)
    
    print(f"Precisi�n por clase guardada en {output_file}")
#%%Inicialización de parámetros
train_loss = []  # Lista para almacenar la pérdida de entrenamiento por época
valid_loss = []  # Lista para almacenar la pérdida de validación por época
train_accuracy = []  # Lista para almacenar la precisión de entrenamiento por época
valid_accuracy = []  # Lista para almacenar la precisión de validación por época
all_accuracies = []  # Lista para almacenar todas las precisiónes calculadas
all_precisions = []  # Lista para almacenar todas las precisiones calculadas
all_recalls = []  # Lista para almacenar todas las recalls calculadas
all_f1_scores = []  # Lista para almacenar todos los F1 scores calculados
lr = 5e-4  # Tasa de aprendizaje
num_epochs = 30  # Número de épocas
batch_size = 64  # Tamaño del lote
save_weights_every = 10 # Número de épocas entre cada guardado de pesos
#%% Verificar si la GPU está disponible
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Se utilizara cuda para el modelo\n")
else:
    device = torch.device('cpu')
    print("Se utilizara cpu para el modelo\n")
#%%Crear el método de validación del modelo
# Crear data loaders
train_loader = DataLoader(train_balanced_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_balanced_dataset, batch_size=batch_size, shuffle=True)
print(f"Numero de tiles usados para entrenamiento: {len(train_balanced_dataset)}\n")
print(f"Numero de tiles usados para validacion: {len(valid_balanced_dataset)}\n")
#%%
def train_resnet(model):
    Train_Model = nn.Sequential(edited_model,
                                  nn.Flatten(), # Creamos una capa de aplanamiento para la salida del modelo
                                  nn.Linear(in_features=512, out_features=num_classes, bias=True), # Aplicamos un entrada de 512 características. out_features tiene asignado por defecto el número de clases que se obtiene en la entrada de datos esta puede ser editada de acuerdo a nuestras clases de manera manual
                                  nn.Softmax(dim=1))# Creamos una capa Softmax para la salida del modelo.
    #%%Definir optimizador y función de pérdida
    optimizer = torch.optim.Adam(Train_Model.parameters(), lr=lr, betas=(0.9,0.999))
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()  # Marca de tiempo inicial
    Train_Model.to(device)
    #%%Entrenamiento del modelo
    for epoch in range(num_epochs):
        # Inicialización de variables para seguimiento de métricas
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_time = time.time() - start_time  # Tiempo transcurrido desde el inicio de la época
        start_time = time.time()  # Actualización del tiempo de inicio de la época
        # Iteración sobre los datos de entrenamiento
        for i, data in enumerate(train_loader, 0):
            # Obtención de datos de entrada y etiquetas
            inputs, targets = data
            # Transferencia de datos a dispositivo de cómputo (GPU si está disponible)
            inputs, targets = inputs.to(device), targets.to(device)
            # Reinicio de los gradientes
            optimizer.zero_grad()
            # Paso hacia adelante: cálculo de las salidas del modelo
            outputs = Train_Model(inputs)
            # Cálculo de la pérdida
            loss = criterion(outputs, targets)
            # Retropropagación del error
            loss.backward()
            # Actualización de los pesos
            optimizer.step()
            # Seguimiento de la pérdida acumulada
            running_loss += loss.item()
            # Cálculo de la cantidad de predicciones correctas
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == targets).sum().item()
            # Actualización del contador de ejemplos totales
            total_train += targets.size(0)
            # Impresión de la pérdida cada 500 mini-lotes
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, running_loss / 500))
        # Métricas de entrenamiento y validación
        train_accuracy.append(100 * correct_train / total_train)
        train_loss.append(running_loss / len(train_loader))
        valid_loss, valid_accuracy = validate_model(Train_Model, valid_loader, criterion, device)
        # Impresión de métricas de entrenamiento y validación
        print(f'Train Accuracy: {train_accuracy[-1]:.2f}%')
        print(f'Train Loss: {train_loss[-1]:.4f}')
        print(f'Validation Accuracy: {valid_accuracy[-1]:.2f}%')
        print(f'Validation Loss: {valid_loss[-1]:.4f}')
        print(f"Epoch {epoch + 1}: {epoch_time:.2f} seconds\n")
        
        # Guardar m�tricas en el archivo CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, train_accuracy, valid_loss, valid_accuracy, epoch_time])
        
        # Guardar pesos del modelo cada 10 épocas
        if (epoch + 1) % save_weights_every == 0:
            # Construcción de la ruta de guardado utilizando el número de época
            ruta_guardado_pesos = f'C:/Users/messi/Desktop/Proyecto_Muerdago/Weights/Model_ResNet_epoch_{epoch + 1}.pth'
            # Guardar los pesos del modelo en la ruta especificada
            torch.save(Train_Model.state_dict(), ruta_guardado_pesos)
    #%%Calcular precisión por clase y graficar métricas
    accuracy_for_class(Train_Model, valid_loader, classes, device)  # Calcula la precisión por clase
    grafic(train_loss, valid_loss, train_accuracy, valid_accuracy)  # Grafica las métricas de entrenamiento y validación
    accuracy, precision, recall, f1 = metrics(Train_Model, valid_loader, classes)  # Calcula las métricas globales
    # Agrega las métricas globales a las listas para su posterior análisis
    all_accuracies.append(accuracy)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1_scores.append(f1)
    #%%Guardar el modelo
    # Guardar el estado del modelo (pesos) en un archivo .pth
    torch.save(Train_Model.state_dict(), r"C:/Users/messi/Desktop/Proyecto_Muerdago/Weights\Pesos_Model_ResNet.pth")
    # Guardar el modelo completo en un archivo .pth
    torch.save(Train_Model, r"C:/Users/messi/Desktop/Proyecto_Muerdago/Weights\Model_ResNet.pth")
#%%
def train_densenet(model):
    Train_Model = nn.Sequential(edited_model,
                                  nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Flatten(), # Creamos una capa de aplanamiento para la salida del modelo
                                  nn.Linear(in_features=1024, out_features=num_classes, bias=True), # Aplicamos un entrada de 512 características. out_features tiene asignado por defecto el número de clases que se obtiene en la entrada de datos esta puede ser editada de acuerdo a nuestras clases de manera manual
                                  nn.Softmax(dim=1))# Creamos una capa Softmax para la salida del modelo.
    #%%Definir optimizador y función de pérdida
    optimizer = torch.optim.Adam(Train_Model.parameters(), lr=lr, betas=(0.9,0.999))
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()  # Marca de tiempo inicial
    Train_Model.to(device)
    #%%Entrenamiento del modelo
    for epoch in range(num_epochs):
        # Inicialización de variables para seguimiento de métricas
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_time = time.time() - start_time  # Tiempo transcurrido desde el inicio de la época
        start_time = time.time()  # Actualización del tiempo de inicio de la época
        # Iteración sobre los datos de entrenamiento
        for i, data in enumerate(train_loader, 0):
            # Obtención de datos de entrada y etiquetas
            inputs, targets = data
            # Transferencia de datos a dispositivo de cómputo (GPU si está disponible)
            inputs, targets = inputs.to(device), targets.to(device)
            # Reinicio de los gradientes
            optimizer.zero_grad()
            # Paso hacia adelante: cálculo de las salidas del modelo
            outputs = Train_Model(inputs)
            # Cálculo de la pérdida
            loss = criterion(outputs, targets)
            # Retropropagación del error
            loss.backward()
            # Actualización de los pesos
            optimizer.step()
            # Seguimiento de la pérdida acumulada
            running_loss += loss.item()
            # Cálculo de la cantidad de predicciones correctas
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == targets).sum().item()
            # Actualización del contador de ejemplos totales
            total_train += targets.size(0)
            # Impresión de la pérdida cada 500 mini-lotes
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, running_loss / 500))
        # Métricas de entrenamiento y validación
        train_accuracy.append(100 * correct_train / total_train)
        train_loss.append(running_loss / len(train_loader))
        valid_loss, valid_accuracy = validate_model(Train_Model, valid_loader, criterion, device)
        # Impresión de métricas de entrenamiento y validación
        print(f'Train Accuracy: {train_accuracy[-1]:.2f}%')
        print(f'Train Loss: {train_loss[-1]:.4f}')
        print(f'Validation Accuracy: {valid_accuracy[-1]:.2f}%')
        print(f'Validation Loss: {valid_loss[-1]:.4f}')
        print(f"Época {epoch + 1}: {epoch_time:.2f} seconds\n")
        # Guardar pesos del modelo cada 10 épocas
        if (epoch + 1) % save_weights_every == 0:
            # Construcción de la ruta de guardado utilizando el número de época
            ruta_guardado_pesos = f'C:/Users/maorc/Documents/Tesis/Pesos/DenseNet-121/Pesos/Model_DenseNet_epoch_{epoch + 1}.pth'
            # Guardar los pesos del modelo en la ruta especificada
            torch.save(Train_Model.state_dict(), ruta_guardado_pesos)
    #%%Calcular precisión por clase y graficar métricas
    accuracy_for_class(Train_Model, valid_loader, classes, device)  # Calcula la precisión por clase
    grafic(train_loss, valid_loss, train_accuracy, valid_accuracy)  # Grafica las métricas de entrenamiento y validación
    accuracy, precision, recall, f1 = metrics(Train_Model, valid_loader, classes)  # Calcula las métricas globales
    # Agrega las métricas globales a las listas para su posterior análisis
    all_accuracies.append(accuracy)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1_scores.append(f1)
    #%%Guardar el modelo
    # Guardar el estado del modelo (pesos) en un archivo .pth
    torch.save(Train_Model.state_dict(), r"C:\Users\maorc\Documents\Tesis\Pesos\DenseNet-121\Pesos_Model_DenseNet.pth")
    # Guardar el modelo completo en un archivo .pth
    torch.save(Train_Model, r"C:\Users\maorc\Documents\Tesis\Pesos\DenseNet-121\Model_DenseNet.pth")
#%%
def train_model(model):
    if isinstance(model, models.resnet.ResNet):
        train_resnet(model)
    elif isinstance(model, models.densenet.DenseNet):
        train_densenet(model)
    else:
        print("El modelo seleccionado no est� configurado.")
        return
train_model(model)