import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from simclr import SimCLR, SimCLR_SNN
from simclr.modules import LogisticRegression, get_resnet, get_resnet_spiking
from simclr.modules.transformations import TransformsSimCLR

from utils import yaml_config_hook


def train(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        if args.spiking:
            output, _ = model(x)
        else:
            output = model(x)

        loss = criterion(output, y)
        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def test(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        if args.spiking:
            output, _ = model(x)
        else:
            output = model(x)

        loss = criterion(output, y)
        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(vars(args))

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    if args.spiking:
        model = get_resnet_spiking(args.resnet, args.timestep)
    else:
        model = get_resnet(args.resnet)
    model = model.to(args.device)

    # Load weights from pre-trained file
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))

    # Map the keys from the source model to the target model
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_fp, map_location=args.device.type)
    new_dict = {}
    for k, v in pretrained_dict.items():
        if "encoder." in k:
            # Remove the 'encoder.' prefix and use the rest as the key for the target model
            key = k.replace("encoder.", "")
            new_dict[key] = v

    model.load_state_dict(new_dict, strict=False)

    for param in model.parameters():
        param.requires_grad = False
    # Only the fc layer has requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()


    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(
            args, train_loader, model, criterion, optimizer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}"
        )

    # final testing
    loss_epoch, accuracy_epoch = test(
        args, test_loader, model, criterion, optimizer
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}"
    )
