import torch
import torch.nn as nn
import torch.nn.functional as F


class ClayTargetCNN(nn.Module):

    def __init__(self, num_classes=2):
        super(ClayTargetCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # [x, y, width, height]
        )

    def forward(self, x):
        features = self.conv_layers(x)

        class_logits = self.fc_layers(features)

        bbox = self.bbox_regressor(features)

        return class_logits, bbox

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            class_logits, bbox = self.forward(x)
            class_probs = F.softmax(class_logits, dim=1)
            class_pred = torch.argmax(class_probs, dim=1)

        return class_pred, class_probs, bbox


def create_model(device='cpu'):
    model = ClayTargetCNN(num_classes=2)
    model.to(device)
    return model


if __name__ == "__main__":
    model = create_model()
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    batch_size = 4
    channels = 3
    height = 224
    width = 224
    dummy_input = torch.randn(batch_size, channels, height, width)

    class_logits, bbox = model(dummy_input)
    print(f"Выход классификации: {class_logits.shape}")
    print(f"Выход bounding box: {bbox.shape}")