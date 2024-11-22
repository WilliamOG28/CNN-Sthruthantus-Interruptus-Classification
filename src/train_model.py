import torch
import torch.nn as nn
import yaml
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torch.utils.data import DataLoader
from resnet34 import train_resnet
from densenet121 import train_densenet
import csv

# Cargar parámetros desde el archivo YAML
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Extraer parámetros
lr = params["learning_rate"]
num_epochs = params["epochs"]
batch_size = params["batch_size"]
save_weights_every = params["save_weights_every"]
model_params = params["model"]

# Configuración del modelo
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

# Eliminar última capa y congelar pesos
edited_model = nn.Sequential(*list(model.children())[:-1])
for parameter in edited_model.parameters():
    parameter.requires_grad = False

# Configurar transformaciones y cargar datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder("data/train", transform=transform)
valid_dataset = ImageFolder("data/validation", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
print(f"Numero de tiles usados para entrenamiento: {len(train_dataset)}\n")
print(f"Numero de tiles usados para validacion: {len(valid_dataset)}\n")

# Guardar métricas en CSV
csv_file = r'metrics/training_metrics.csv'
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train_Loss", "Train_Accuracy", "Validation_Loss", "Validation_Accuracy"])

# Verificar dispositivo
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Se utilizara cuda para el modelo\n")
else:
    device = torch.device('cpu')
    print("Se utilizara cpu para el modelo\n")

# Entrenar el modelo
def train_model(model):
    if isinstance(model, models.resnet.ResNet):
        train_resnet(
        device, edited_model, len(train_dataset.classes), train_loader, valid_loader,
        train_dataset.classes, lr, num_epochs, csv_file, save_weights_every)
    elif isinstance(model, models.densenet.DenseNet):
        train_densenet(model)
    else:
        print("El modelo seleccionado no esta configurado.")
        return
train_model(model)
