import os
import torchvision
from torchvision import datasets
from PIL import Image
from tqdm import tqdm

# Imposta il percorso di salvataggio
root_dir = "cifar10_png"
train_dir = os.path.join(root_dir, "train")
test_dir = os.path.join(root_dir, "test")

# CIFAR-10 ha 10 classi
class_names = datasets.CIFAR10(root=".", download=True).classes  # Lista dei nomi delle classi

# Crea le directory per train e test
for directory in [train_dir, test_dir]:
    for class_name in class_names:
        os.makedirs(os.path.join(directory, class_name), exist_ok=True)

# Scarica CIFAR-10
train_dataset = datasets.CIFAR10(root=".", train=True, download=True)
test_dataset = datasets.CIFAR10(root=".", train=False, download=True)

# Funzione per salvare le immagini
def save_images(dataset, save_dir):
    for i in tqdm(range(len(dataset))):
        img, label = dataset[i]
        class_name = class_names[label]  # Converte indice numerico in nome classe
        img_path = os.path.join(save_dir, class_name, f"{i}.png")
        img.save(img_path)

# Salva immagini
save_images(train_dataset, train_dir)
save_images(test_dataset, test_dir)

print("CIFAR-10 salvato in formato PNG!")
