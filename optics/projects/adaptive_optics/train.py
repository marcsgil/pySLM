
# importing the libraries
# import pandas as pd
import time
import numpy as np
from datetime import datetime, timezone

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
# % matplotlib inline

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from projects.adaptive_optics import log
from pathlib import Path
#
# 1. Define a Convolutional Neural Network
M = 2
N = 10

num_cv1 = 8
num_cv2 = 16
num_cv3 = 32
num_cv4 = 64
num_cv5 = 128
cv_kernel_size = 3
in_features = M+num_cv1+num_cv2+num_cv3+num_cv4+num_cv5


class Net(nn.Module):
    def __init__(self, M, num_classes):
        super(Net, self).__init__()
        self.M = M

        # 4 Convolutional Layers with max-pooling
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=M, out_channels=num_cv1, kernel_size=cv_kernel_size, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=num_cv1, out_channels=num_cv2, kernel_size=cv_kernel_size, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Tanh()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_cv2, out_channels=num_cv3, kernel_size=cv_kernel_size, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Tanh()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=num_cv3, out_channels=num_cv4, kernel_size=cv_kernel_size, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Tanh()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=num_cv4, out_channels=num_cv5, kernel_size=cv_kernel_size, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Tanh()
        )

        # Concatenated Fully Connected Layer
        self.fc1 = nn.Linear(in_features=in_features, out_features=32)

        # Output Layer
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

        # Activation Function
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # Global Max-Pooling and Concatenation
        x1_global = nn.functional.max_pool2d(x, kernel_size=x.size()[2:]).view(-1, self.M)
        x2_global = nn.functional.max_pool2d(x1, kernel_size=x1.size()[2:]).view(-1, 8)
        x3_global = nn.functional.max_pool2d(x2, kernel_size=x2.size()[2:]).view(-1, 16)
        x4_global = nn.functional.max_pool2d(x3, kernel_size=x3.size()[2:]).view(-1, 32)
        x5_global = nn.functional.max_pool2d(x4, kernel_size=x4.size()[2:]).view(-1, 64)
        x6_global = nn.functional.max_pool2d(x5, kernel_size=x5.size()[2:]).view(-1, 128)

        x_concat = torch.cat((x1_global, x2_global, x3_global, x4_global, x5_global, x6_global), 1)

        x_fc1 = self.fc1(x_concat)
        x_tanh = self.tanh(x_fc1)
        x_fc2 = self.fc2(x_tanh)
        x_linear = self.linear(x_fc2)

        return x_linear


def load_data_from_npz_file(__folderpath, __filename, __corrected_modes):
    __output_folder = Path(__folderpath)
    __filename = __output_folder / __filename
    __pseudo_psf = np.load(str(__filename))
    __train_data = __pseudo_psf['train_data']
    __train_label = __pseudo_psf['train_label']
    __train_label = __train_label[:, __corrected_modes]
    __train_data = np.float32(__train_data)
    __train_label = np.float32(__train_label)
    return __train_data, __train_label


def load_data_from_npz_file_batches(__folderpath, __filename1, __filename2, __corrected_modes, nb_batches):

    for __batch_idx in range(nb_batches):
        __filename = __filename1 + str(__batch_idx) + __filename2
        __train_data, __train_label = load_data_from_npz_file(__folderpath, __filename, __corrected_modes)
        if __batch_idx == 0:
            __train_data_all = __train_data
            __train_label_all = __train_label
        else:
            __train_data_all = np.concatenate((__train_data_all, __train_data), axis=0)
            __train_label_all = np.concatenate((__train_label_all, __train_label), axis=0)
    return __train_data_all, __train_label_all


class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index])
        y = torch.Tensor(self.y[index])
        return (x, y)

    def __len__(self):
        count = self.x.shape[0]
        return count


# 2 define training hyperparameters
timestamp_start = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-3]
batch_number = 50
batch_size = 10000
learning_rate = 0.0008
num_epochs = 1000
# define the train and val splits
test_size = 0.1

# set the device we will be using to train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datatype = torch.float32
log.info(f'Using PyTorch device {device}...')
# Create an instance of the CNN network

model_name = 'ast2'
model = Net(M, N).to(device=device, dtype=datatype)
# Define the loss function
loss_fn = nn.MSELoss()
# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0)

# 3 load training data
# load .npz data from the folder
foldername = 'E:/Adaptive Optics/Test of model from multiple batches/Matched resolution/'
# dave the trained model in this folder
model_path = Path.home() / Path('E:/Adaptive Optics/Test of model from multiple batches/Trained Model/')

bias_modes = [3]
simulate_noise = True
background_noise_level = 0.02
magnitude = 2/(np.sqrt(3))
# corrected_modes = [3, 5, 6, 7, 8, 9, 11, 12, 13, 22]
corrected_modes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# 6 Train the model

train_loss_all = torch.zeros(num_epochs)
test_loss_all = torch.zeros(num_epochs)
# test_loss_all_0 = torch.zeros(num_epochs)
# test_loss_all_1 = torch.zeros(num_epochs)
# test_loss_noise_all = torch.zeros(num_epochs)
predicted_labels_epoch = []
test_loss_best = np.inf

filename1 = 'PseudoPSF_Match_resolution_batchidx'
filename2 = '_batchsize10000_nb_batches50_corrected_modes[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]_biasmode[3]_biasdepths[1]_magnitude_1.155_simulate_noise_True_background_noise_level_0.02_seed_none.npz'
filename = filename1 + filename2
train_data, train_label = load_data_from_npz_file_batches(foldername, filename1, filename2, corrected_modes, batch_number)
train_data, test_data, train_label, test_label = train_test_split(train_data, train_label, test_size=test_size)

#convert numpy array to a pytorch tensor

test_data = torch.from_numpy(test_data)
test_label = torch.from_numpy(test_label)
test_data = test_data.to(device=device, dtype=datatype)
test_label = test_label.to(device=device, dtype=datatype)

# split into batches with DataLoader
print(train_data.shape)
print(test_data.shape)

train_dataset = Dataset(train_data, train_label)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    # Forward pass
    for train_data, train_label in train_loader:
        train_data = train_data.to(device=device, dtype=datatype)
        train_label = train_label.to(device=device, dtype=datatype)
        training_loss = loss_fn(model(train_data), train_label)
        optimizer.zero_grad()  # zero the parameter gradients
        training_loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_predicted_label = model(test_data).to(device=device, dtype=datatype)
        test_loss = loss_fn(test_predicted_label, test_label.to(device=device, dtype=datatype))
        log.info(f'Epoch [{epoch}/{num_epochs}]. Training loss: {training_loss:0.6f}. Test loss: {test_loss:0.6f}')

        train_loss_all[epoch] = training_loss
        test_loss_all[epoch] = test_loss

        if epoch % 10 == 0:
            predicted_labels_epoch.append(test_predicted_label)
        if test_loss < test_loss_best:
            test_loss_best = test_loss
            best_model = model

timestamp = datetime.now(timezone.utc).strftime('%H_%M_%S.%f')[:-3]
model_save_name = model_path / f'CNN_Model_Resolution_name_{model_name}_M_{M}_N_{N}_Num_epochs_{num_epochs}_Num_batches_{batch_number}_Batch_Size_{batch_size}_Corrected_modes_{corrected_modes}_CV_kernel_size_{cv_kernel_size}_learningrate_{learning_rate}_in_features{in_features}_timestamp_{timestamp}_magnitude_{magnitude:0.3f}_noiselevel_{background_noise_level}.pt'
full_file_name = model_path / f'CNN_training_Data_Resolution_Model_name_{model_name}_M_{M}_N_{N}_Num_epochs_{num_epochs}_Num_batches_{batch_number}_Corrected_modes_{corrected_modes}_CV_kernel_size_{cv_kernel_size}_learningrate_{learning_rate}_in_features{in_features}_timestamp_{timestamp}_magnitude_{magnitude:0.3f}_noiselevel_{background_noise_level}.npz'
log.info(f'Saving trained model to {model_path}...')
model_scripted = torch.jit.script(best_model)  # Export to TorchScript
model_scripted.save(model_save_name)  # Save
settings = dict(M=M,
                N=N,
                num_cv1=num_cv1,
                num_epochs=num_epochs,
                num_batches=batch_number,
                batch_size=batch_size,
                cv_kernel_size=cv_kernel_size,
                model_name=model_name,
                learning_rate=learning_rate,
                in_features=in_features,
                background_noise_level=background_noise_level,
                filename=filename,
                full_file_name=full_file_name,
                timestamp_start=timestamp_start,
                timestamp=timestamp
                )
log.info(f'Saving results to {full_file_name}...')
predicted_labels_epoch = torch.stack(predicted_labels_epoch)
np.savez(full_file_name, test_label=test_label.cpu().data.numpy(), test_data=test_data.cpu().data.numpy(), predicted_labels_epoch=predicted_labels_epoch.cpu().data.numpy(), train_loss_all=train_loss_all.detach().numpy(), test_loss_all=test_loss_all.detach().numpy(), settings=settings)
# print(predicted_labels_epoch)

# make plot
train_loss_min = torch.min(train_loss_all, 0, True)
test_loss_min = torch.min(test_loss_all, 0, True)
train_data_shape = train_data.shape
title_str = f' model_MLAO_{model_name}_training_data{train_data_shape}_cv_kernel_size{cv_kernel_size}_train_loss_min_{test_loss_min.values.detach().numpy()[0]:0.3f}_min_idx_{test_loss_min.indices.detach().numpy()[0]}'
# print(title_str)

plt.figure()
plt.plot(train_loss_all.detach().numpy(), label="train_loss")
plt.plot(test_loss_all.detach().numpy(), label="test_loss")

plt.title(title_str)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.show()

