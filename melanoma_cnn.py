import numpy
from torch import nn

class SkinCancerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 224x224 → 112x112

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 112x112 → 56x56

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 56x56 → 28x28

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 28x28 → 14x14


            nn.Flatten(),
            nn.Dropout(0.5),                 # Turn off 50% of neuron
            nn.Linear(512 * 14 * 14, 512),   # Linear= final output*hight*lenght
            nn.ReLU(),
            nn.Linear(512, 2)                # Final output (Melanoma, Not Melanoma)   
        )

    def forward(self, x):
        return self.model(x)  