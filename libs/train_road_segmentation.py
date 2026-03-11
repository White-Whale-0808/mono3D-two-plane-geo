
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101,  DeepLabV3_ResNet101_Weights

# =========================
# Create the data loader
# =========================

class CityscapeDataset(Dataset):
    def __init__(self, img_root, label_root):
        self.img_root = img_root
        self.label_root = label_root

        self.images = []

        for root, _, files in os.walk(img_root):  # expected to input "datasets/leftImg8bit/train"
            for file in files:
                self.images.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = img_path.replace("leftImg8bit", "gtFine").replace(
            "_gtFine", "_gtFine_labelTrainIds"
        )

        image = np.array(Image.open(img_path))
        label = np.array(Image.open(label_path))
        road_mask = (label == 0).astype(np.float32)

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        road_mask = torch.tensor(road_mask).unsqueeze(0)
        return image, road_mask

# =========================
# Build the model
# =========================
def build_model(device):
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)

    model.classifier[4] = nn.Conv2d(
        in_channels= 256,
        out_channels= 1,
        kernel_size= 1
    )

    model.to(device)
    return model

# =========================
# Train the model
# =========================
def train_model(model, dataloader, device, num_epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, road_masks in dataloader:
            images, road_masks = images.to(device), road_masks.to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, road_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

def main():
    dataset = CityscapeDataset("datasets/leftImg8bit/train", "datasets/gtFine/train")
    dataloader = DataLoader(dataset, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_model(device)
    train_model(model, dataloader, device, num_epochs=1)
    
if __name__ == "__main__":
    main()