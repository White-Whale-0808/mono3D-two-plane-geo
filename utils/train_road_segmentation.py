import torch
import yaml
from libs.dataset.cityscape_dataset import CityscapeDataset
from torch.utils.data import DataLoader
from libs.model.restnet101 import build_train_model
import torch.nn as nn
import torch.optim as optim
from libs.engine.train import train_one_epoch
from libs.engine.validate import validate_one_epoch

with open("config/train_road_segmentation.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

train_image_dir = config["dataset"]["train"]["image_dir"]
train_label_dir = config["dataset"]["train"]["label_dir"]
val_image_dir = config["dataset"]["val"]["image_dir"]
val_label_dir = config["dataset"]["val"]["label_dir"]

batch_size = config["training"]["batch_size"]
epochs = config["training"]["epochs"]
lr = config["training"]["learning_rate"]
shuffle = config["training"]["shuffle"]

device = config["model"]["device"]
save_path = config["checkpoint"]["save_best_path"]

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, path)

def main():

    train_dataset = CityscapeDataset(train_image_dir, train_label_dir)
    val_dataset = CityscapeDataset(val_image_dir, val_label_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = build_train_model(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_iou = 0.0

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Training for epoch {epoch+1}/{epochs} completed.")

        val_iou = validate_one_epoch(model, val_loader, loss_fn, device)
        print(f"Validation for epoch {epoch+1}/{epochs} completed.")

        if val_iou > best_iou:
            best_iou = val_iou
            print(f"New best IoU: {best_iou:.4f}. Saving checkpoint.")
            save_checkpoint(model, optimizer, epoch, save_path)


if __name__ == "__main__":
    main()