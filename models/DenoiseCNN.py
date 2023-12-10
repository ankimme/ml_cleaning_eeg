import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim


class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(19, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.15),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.15),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.15),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.15),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.15),
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.15),
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.15),
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.15),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.15),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.15),
            nn.ConvTranspose1d(64, 19, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.net(x)
        return x
