import pandas as pd
import matplotlib.pyplot as plt
#%%Función para graficar métricas

def grafic(csv_file):
    # Cargar los datos desde el archivo CSV
     
    data = pd.read_csv(csv_file)

    # Extraer las columnas relevantes
    train_loss_list = data['Train_Loss']
    valid_loss_list = data['Validation_Loss']
    train_accuracy_list = data['Train_Accuracy']
    valid_accuracy_list = data['Validation_Accuracy']

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
    plt.savefig(r'plots/ResNet_Plot.png')
    # Muestra la figura con los subgráfico