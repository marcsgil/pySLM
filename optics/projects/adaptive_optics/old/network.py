import torch
import torch.nn as nn


class AdaptiveOpticsNet(nn.Module):
    def __init__(self, M, num_classes):
        super().__init__()
        num_cv1 = 8
        num_cv2 = 16
        num_cv3 = 32
        num_cv4 = 64
        num_cv5 = 128
        cv_kernel_size = 3
        in_features = M + num_cv1 + num_cv2 + num_cv3 + num_cv4 + num_cv5

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
