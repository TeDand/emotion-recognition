#!/usr/bin/env python
# coding: utf-8
import torch
import os
from ghostnet import ghostnet
from torchvision import transforms
import torch.nn as nn
from timm.data import create_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    assert torch.cuda.is_available()
    device_name = torch.cuda.get_device_name()
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda")


    model = ghostnet(num_classes=1000, width=1.0, dropout=0.2)
    input = torch.randn(32,3,224,224)
    y = model(input)

    model.load_state_dict(torch.load('emotion-recognition/state_dict_73.98.pth'))
    model = torch.nn.DataParallel(model, device_ids = list(range(n_gpu))).cuda()
    model.eval()

    imsize = 224
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

    num_classes = 8
    num_features = model.module.classifier.in_features
    new_last_layer = nn.Linear(num_features, num_classes).cuda()
    model.module.classifier = new_last_layer

    dataset = create_dataset(name='', root='emotion-recognition/AffectNetData', transform=loader)

    batch_size = 64

    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = int(0.1 * len(dataset))   # 10% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining 10% for test

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    num_epochs = 450
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), eps=0.001, lr=0.001, weight_decay=0.973)

    model.train()
    for epoch in range(num_epochs):
        total_samples = 0
        correct_predictions = 0
        total_loss = 0

        loop = tqdm(train_loader)
        for images, labels in loop:
            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels.cuda())

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels.cuda()).sum().item()
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(), acc=(predicted == labels.cuda()).sum().item()/labels.size(0))

        avg_loss = total_loss / len(train_loader)
        avg_acc = correct_predictions / total_samples
        print("loss=", avg_loss, ", accuracy=", avg_acc)

        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'emotion-recognition/ghostnet_checkpoints/checkpoint_{epoch}.pth')
        
if __name__ == "__main__":
    main()