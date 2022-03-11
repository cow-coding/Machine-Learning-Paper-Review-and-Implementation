import argparse
import os
import argparse

import torch
import torch.optim as optim
from torch.optim import SGD
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import AlexNet

def arg_getter():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epoch", required=False, help="Num EPOCHS", default=10, type=int)

    args = parser.parse_args()
    return args

def main(num_epochs=10):
    seed = torch.initial_seed()

    # Hyperparameter Setting
    batch_size = 128
    momentum = 0.9
    weight_decay = 0.0005
    lr_init = 0.01
    image_dim = 227
    num_classes = 1000
    num_epochs = num_epochs
    device = 'cpu'

    # GPU setting
    if torch.cuda.is_available():
        device = 'cuda'

    # dataset, dataloader
    train_dataset = datasets.ImageNet(
        root='./data/train',
        split='train',
        transform=transforms.Compose([
            transforms.CenterCrop(image_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        download=True
    )

    val_dataset = datasets.ImageNet(
        root='./data/valid',
        split='val',
        download=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=batch_size
    )

    valid_dataloader = DataLoader(
        val_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=batch_size
    )

    # Optimizer, Scheduler, Model, Loss setting
    model = AlexNet(num_classes=num_classes)
    optimizer = SGD(
        params=model.parameters(),
        lr=lr_init,
        momentum=momentum,
        weight_decay=weight_decay
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # training
    step = 1
    for epoch in range(1, num_epochs+1):
        lr_scheduler.step()

        for imgs, classes in train_dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            pred = model(imgs)
            loss = F.cross_entropy(pred, classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+0:03}: | Loss: {loss.item()}")


if __name__ == '__main__':
    args = arg_getter()
    main(num_epochs=args.epoch)
