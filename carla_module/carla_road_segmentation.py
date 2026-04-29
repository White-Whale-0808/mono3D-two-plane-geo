import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
ROAD_CLASS = 0


def predict_road_from_pil(model, pil_image, device, resize_size):
    """PIDNet 版本：接受 PIL Image 輸入（CARLA 即時串流使用），以 argmax 取代 sigmoid+threshold。"""
    resized_image = pil_image.resize((resize_size[1], resize_size[0]), Image.BILINEAR)
    img = np.array(resized_image).astype(np.float32)
    img = (img / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(tensor)
        pred = F.interpolate(pred, size=tensor.shape[-2:], mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

    pred_mask = (pred == ROAD_CLASS).astype(np.uint8)
    return resized_image, pred_mask
