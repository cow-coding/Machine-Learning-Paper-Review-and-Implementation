import torch
import wandb
from model import AlexNet


def main():
    model = AlexNet(output_dim=10)
    data = torch.rand(2, 3, 224, 224)
    print(data.shape)
    print(model(data))


if __name__ == '__main__':
    main()
