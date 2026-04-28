from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
import numpy as np


def predict_road_from_pil(model, pil_image, device, resize_size, threshold):
    # 接受 PIL Image 輸入，邏輯與 predict_road 相同，供 CARLA 即時串流使用
    resize = transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR)
    resized_image = resize(pil_image)

    transformed_image = transforms.ToTensor()(resized_image)
    input_image = transformed_image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_image)["out"]
        probs = torch.sigmoid(logits)
        pred_mask = (probs > threshold).float()
    pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)

    return resized_image, pred_mask
