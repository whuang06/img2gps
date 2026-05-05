import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights

class ResnetClassifier(nn.Module):
    def __init__(self, num_classes, dropout_p=0):
        super(ResnetClassifier, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)



class ViTClassifier(nn.Module):
    def __init__(self, num_classes, dropout_p=0, attention_dropout_p=0):
        super(ViTClassifier, self).__init__()
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1,
                                   dropout=dropout_p,
                                   attention_dropout=attention_dropout_p)
        
        for param in self.vit.parameters():
            param.requires_grad = False
            
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
