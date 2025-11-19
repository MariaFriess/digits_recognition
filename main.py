import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as tfs
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from tqdm import tqdm

from Digits_dataset_class import DigitsDataset
from Linear_model_class import LinearModel


epochs = 2
batch_size = 32
lr = 0.01
transforms = tfs.Compose([tfs.ToImage(),
                          tfs.Grayscale(),
                          tfs.ToDtype(torch.float32, scale=True),
                          tfs.Lambda(lambda _img: _img.ravel())
                          ])

model = LinearModel()

# d_train = DigitsDataset('C:\ML\ml\pytorch\digits_rec\dataset',
#                         train=True, transform_func=to_tensor)  # My dataset class


# ImageFolder dataset class
d_train = ImageFolder('digits_rec/dataset/train', transform=transforms)


train_data = data.DataLoader(d_train, batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

model.train()

for _ in range(epochs):
    loss_mean = 0
    lm_count = 0
    train_tqdm = tqdm(train_data, leave=True)

    for x, y in train_tqdm:
        predict = model(x)
        loss = loss_func(predict, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean

        train_tqdm.set_description(
            f'Epoch [{_ + 1}/{epochs}], loss_mean={loss_mean:.3f}')
    print(_)


model.eval()


# If dataset class is DigitsDataset
# d_test = DigitsDataset('C:\ML\ml\pytorch\digits_rec\dataset',
#                        train=False, transform_func=transforms)


# If dataset class is ImageFolder
d_test = ImageFolder('digits_rec/dataset/test', transform=transforms)

test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)


Q = 0
for x, y in test_data:
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        # targ = y.argmax(dim=1) # If dataset class is DigitsDataset
        # Q += torch.sum(pred == targ).item()
        Q += torch.sum(pred == y).item()  # If dataset class is ImageFolder


Q /= len(d_test)
print(Q)


# Showing one image, it's target and pediction
test_idx = 287

test_img, test_targ = d_test[test_idx]

test_pred = model(test_img)
test_pred = test_pred

print('targ:', test_targ.argmax(), 'pred:', test_pred)

test_img = test_img.reshape(28, 28).numpy() * 255.0
img = Image.fromarray(test_img)
img.show()
