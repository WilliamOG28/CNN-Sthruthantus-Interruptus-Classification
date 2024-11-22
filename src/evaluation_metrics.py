# Importaciones de librerías necesarias
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score 
import matplotlib.pyplot as plt  # Biblioteca para visualización de datos
import torch  # Biblioteca para aprendizaje profundo
import seaborn as sns  # Biblioteca para visualización de datos estadísticos
import json  # Biblioteca para manejar archivos JSON

# Función para calcular métricas de rendimiento del modelo
def metrics(Train_Model, valid_loader, classes, device, output_file=r'metrics/evaluation_metrics.json'):
    """
    Función para calcular y visualizar métricas de rendimiento de un modelo de clasificación.
    
    Pasos:
    1. Preparar variables de seguimiento
    2. Desactivar cálculo de gradientes
    3. Iterar sobre conjunto de validación
    4. Realizar predicciones
    5. Contabilizar predicciones correctas
    6. Calcular métricas (matriz confusión, precisión, recall, F1)
    7. Guardar métricas en archivo JSON
    8. Visualizar matriz de confusión
    9. Imprimir resultados
    """
    # Variables para almacenar predicciones y etiquetas
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    # Desactiva cálculo de gradientes para validación
    with torch.no_grad():
        # Itera sobre los datos de validación
        for data in valid_loader:
            # Obtiene imágenes y etiquetas
            images, labels = data
            # Mueve datos al dispositivo (GPU/CPU)
            images = images.to(device)
            labels = labels.to(device)
            
            # Realiza predicciones con el modelo
            outputs = Train_Model(images)
            # Obtiene la clase con mayor probabilidad
            _, predicted = torch.max(outputs.data, 1)
            
            # Actualiza contadores de precisión
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Guarda etiquetas y predicciones
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calcula métricas de evaluación
    cm = confusion_matrix(all_labels, all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    # Guarda métricas en diccionario JSON
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    with open(output_file, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    # Grafica matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Class')
    plt.ylabel('Real Class')
    plt.title('Confusion Matrix')
    plt.savefig(r'plots/ResNet_CM.png')
    # Imprime métricas de evaluación
    print(f"Accuracy: {accuracy}\n")
    print(f"Precision: {precision}\n")
    print(f"Recall: {recall}\n")
    print(f"F1 Score: {f1}\n")


# Función para calcular precisión por cada clase
def accuracy_per_class(Train_Model, valid_loader, classes, device, output_file=r'metrics/accuracy_class.json'):
    """
    Función para calcular la precisión del modelo para cada clase individual.

    Pasos:
    1. Inicializar contadores por clase
    2. Mover modelo al dispositivo
    3. Desactivar cálculo de gradientes
    4. Iterar sobre conjunto de validación
    5. Realizar predicciones
    6. Contabilizar predicciones correctas por clase
    7. Calcular precisión individual
    8. Crear diccionario de resultados
    9. Guardar resultados en archivo JSON
    """
    # Inicializa contadores para cada clase
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    
    # Mueve el modelo al dispositivo (GPU/CPU)
    Train_Model.to(device)
    
    # Desactiva cálculo de gradientes
    with torch.no_grad():
        # Itera sobre datos de validación
        for data in valid_loader:
            # Obtiene imágenes y etiquetas
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            # Realiza predicciones
            outputs = Train_Model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Calcula clasificaciones correctas por clase
            correct = (predicted == labels)
            for i in range(len(classes)):
                class_correct[i] += (correct * (labels == i)).sum().item()
                class_total[i] += (labels == i).sum().item()
    
    # Crea diccionario con precisión por clase
    accuracy_dict = {}
    for i in range(len(classes)):
        accuracy = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        accuracy_dict[classes[i]] = {
            "Correct": class_correct[i],
            "Total": class_total[i],
            "Accuracy": accuracy
        }
    
    # Guarda precisión por clase en archivo JSON
    with open(output_file, 'w') as json_file:
        json.dump(accuracy_dict, json_file, indent=4)
    
    print(f"Precisión por clase guardada en {output_file}")