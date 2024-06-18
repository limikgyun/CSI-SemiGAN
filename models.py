import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def custom_activation(output):
    logexpsum = torch.sum(torch.exp(output), dim=-1, keepdim=True)
    result = logexpsum / (logexpsum + 1.0)
    return result

class Discriminator(nn.Module):
    def __init__(self, n_classes, in_shape=(1, 120, 1)):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 5))
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 5))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 5))
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 1 * 108, n_classes)
        self.classifier = nn.Softmax(dim=1)
        self.custom_activation = custom_activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.flatten(x)
        x = self.dropout(x)
        classifier_output = self.fc(x)
        classifier_output = self.classifier(classifier_output)
        discriminator_output = self.custom_activation(x)
        return classifier_output, discriminator_output

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        n_nodes = 32 * 1 * 108
        self.fc = nn.Linear(self.latent_dim, n_nodes)
        self.relu = nn.ReLU()
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(1, 5), stride=1)
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(1, 5), stride=1)
        self.conv_transpose3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(1, 5), stride=1)
        self.conv_transpose4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 5), stride=1, padding=(0, 2), activation=nn.Tanh())

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = x.view(-1, 32, 1, 108)
        x = self.conv_transpose1(x)
        x = self.relu(x)
        x = self.conv_transpose2(x)
        x = self.relu(x)
        x = self.conv_transpose3(x)
        x = self.relu(x)
        x = self.conv_transpose4(x)
        return x

class GAN(nn.Module):
    def __init__(self, g_model, d_model):
        super(GAN, self).__init__()
        self.g_model = g_model
        self.d_model = d_model

    def forward(self, x):
        g_output = self.g_model(x)
        d_output = self.d_model(g_output)
        return d_output, g_output
   
class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 5))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 5))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 5))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 1 * 108, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


