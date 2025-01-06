import torch
print(torch.cuda.is_available())  # Debe devolver True si todo está correcto
print(torch.cuda.device_count())  # Número de GPUs detectadas
print(torch.cuda.get_device_name(0))  # Nombre de la primera GPU
