import torch  # Importa PyTorch, una biblioteca popular para aprendizaje profundo.
import torch.nn as nn  # Importa el módulo de redes neuronales de PyTorch.
import time  # Importa el módulo time para medir el tiempo de ejecución.
from validation_model import validate_model
from evaluation_metrics import metrics, accuracy_per_class
import csv

def train_resnet(device, edited_model,num_classes,train_loader, valid_loader, classes, lr,num_epochs, csv_file,save_weights_every):
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
        train_accuracy = (100 * correct_train / total_train)
        train_loss = (running_loss / len(train_loader))
        valid_loss, valid_accuracy = validate_model(Train_Model, valid_loader, criterion, device)
        # Impresión de métricas de entrenamiento y validación
        print(f'Train Accuracy: {train_accuracy:.2f}%')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Accuracy: {valid_accuracy:.2f}%')
        print(f'Validation Loss: {valid_loss:.4f}')
        print(f"Epoch {epoch + 1}: {epoch_time:.2f} seconds\n")
        
        # Guardar m�tricas en el archivo CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, train_accuracy, valid_loss, valid_accuracy])
        
        # Guardar pesos del modelo cada 10 épocas
        if (epoch + 1) % save_weights_every == 0:
            # Construcción de la ruta de guardado utilizando el número de época
            ruta_guardado_pesos = f'weights/Model_ResNet_epoch_{epoch + 1}.pth'
            # Guardar los pesos del modelo en la ruta especificada
            torch.save(Train_Model.state_dict(), ruta_guardado_pesos)


    #%%Calcular precisión por clase y graficar métricas
    accuracy_per_class(Train_Model, valid_loader, classes, device)  # Calcula la precisión por clase
    metrics(Train_Model, valid_loader, classes, device)  # Calcula las métricas globales
    # Agrega las métricas globales a las listas para su posterior análisis
    #%%Guardar el modelo
    # Guardar el estado del modelo (pesos) en un archivo .pth
    torch.save(Train_Model.state_dict(), r"weights\Pesos_Model_ResNet.pth")
    # Guardar el modelo completo en un archivo .pth
    torch.save(Train_Model, r"weights\Model_ResNet.pth")
