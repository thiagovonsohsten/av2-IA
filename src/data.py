"""Carregamento do PneumoniaMNIST com transformações ImageNet."""
from torch.utils.data import DataLoader
from torchvision import transforms

from medmnist import PneumoniaMNIST

from config import BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def get_loaders(seed: int):
    """
    Retorna train_loader, val_loader, test_loader e rótulos do conjunto de teste.
    A seed afeta apenas o shuffle do DataLoader de treino.
    """
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = PneumoniaMNIST(split="train", transform=train_tf, download=True)
    val_ds = PneumoniaMNIST(split="val", transform=eval_tf, download=True)
    test_ds = PneumoniaMNIST(split="test", transform=eval_tf, download=True)

    g = __import__("torch").Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    import numpy as np

    y_test = np.asarray(test_ds.labels).squeeze().astype(int)

    return train_loader, val_loader, test_loader, y_test
