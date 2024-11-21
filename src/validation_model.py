import torch 

#Método para la validación del modelo
def validate_model(Train_Model, valid_loader, criterion, device):
    correct_val = 0
    total_val = 0
    running_loss = 0

    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = Train_Model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()
    
    # Calcula la precisión de la validación y la agrega a la lista de precisión de validación
    valid_accuracy = (100 * correct_val / total_val)
    # Calcula el loss promedio de la validación y lo agrega a la lista de loss de validación
    valid_loss = (running_loss / len(valid_loader))

    return valid_loss, valid_accuracy