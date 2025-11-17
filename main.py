import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as tfs
from PIL import Image
import numpy as np

from Digits_dataset_class import DigitsDataset
from Linear_model_class import LinearModel


epochs = 10
batch_size = 32
lr = 0.01
to_tensor = tfs.ToImage()

model = LinearModel()

d_train = DigitsDataset('C:\ML\ml\pytorch\digits_rec\dataset',
                        train=True, transform_func=to_tensor)
train_data = data.DataLoader(d_train, batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

model.train()

for _ in range(epochs):
    for x, y in train_data:
        predict = model(x)
        loss = loss_func(predict, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(_)


model.eval()


d_test = DigitsDataset('C:\ML\ml\pytorch\digits_rec\dataset',
                       train=False, transform_func=to_tensor)

test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)
Q = 0
for x, y in test_data:
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        targ = y.argmax(dim=1)
        Q += torch.sum(pred != targ).item()

Q /= len(d_test)
print(Q)

test_idx = 287

test_img, test_targ = d_test[test_idx]

test_pred = model(test_img)
test_pred = test_pred.argmax()

print('targ:', test_targ.argmax(), 'pred:', test_pred)

test_img = test_img.reshape(28, 28).numpy() * 255.0
img = Image.fromarray(test_img)
img.show()
