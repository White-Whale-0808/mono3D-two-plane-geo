from PIL import Image
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import pidnet_models as models
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
ROAD_CLASS = 0  # In cityscapes dataset, the road class is labeled as 0.

# PIDNet: our new powerful model for road segmentation. 
# The function is directly copied from the official codebase of PIDNet, which is available at custom.py
def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

def load_pidnet(model_name, weight_path, device):
    model = models.pidnet.get_pred_model(model_name, 19)
    model = load_pretrained(model, weight_path)
    return model.to(device).eval()

def predict_road(model, image_path, device, resize_size):
    '''
    Image is read by PIL, and the model expects RGB input, so we convert it to RGB format.
    The output of the model is a tensor of shape (1, num_classes, H, W). We take the argmax along the class dimension to get the predicted class for each pixel.
    '''
    image = Image.open(image_path).convert('RGB')
    resized_image = image.resize(
        (resize_size[1], resize_size[0]),  # PIL expects (width, height)
        Image.BILINEAR
    )
    img = np.array(resized_image).astype(np.float32)
    img = (img / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(tensor)
        pred = F.interpolate(pred, size=tensor.shape[-2:],
                             mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

    pred_mask = (pred == ROAD_CLASS).astype(np.uint8)
    return resized_image, pred_mask

# The result from the resnet101 model is bad, so we will not use it for now. We use another powerful model for road segmentation
# def predict_road(model, image_path, device, resize_size, threshold):
#     image = Image.open(image_path).convert("RGB")  # The torch model expects RGB input

#     resize = transforms.Resize(
#         resize_size,
#         interpolation=InterpolationMode.BILINEAR
#     )
#     resized_image = resize(image)

#     transformed_image = transforms.ToTensor()(resized_image)
#     input_image = transformed_image.unsqueeze(0).to(device)
#     with torch.no_grad(): 
#         logits = model(input_image)["out"]
#         probs = torch.sigmoid(logits)
#         pred_mask = (probs > threshold).float()
#     pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)

#     return resized_image, pred_mask

def apply_road_mask(resized_image, pred_mask):
    image = np.array(resized_image)
    road_mask_255 = (pred_mask * 255).astype(np.uint8)

    """
    The erosion step is crucial to mitigate jagged boundary artifacts that can arise from the segmentation process.
    Give up doing erosion: it will remove some lane pixels that will make the lane fitting later more difficult.
    """
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_kernel, erosion_kernel))
    # road_mask_255 = cv2.erode(road_mask_255, kernel)  # Shrink mask inward to remove jagged boundary artifacts
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (601, 601))
    # road_mask_255 = cv2.morphologyEx(road_mask_255, cv2.MORPH_CLOSE, kernel)

    # blurred = cv2.GaussianBlur(road_mask_255, (51, 51), 0)
    # road_mask_255 = (blurred > 127).astype(np.uint8) * 255

    # smoothed_mask = (road_mask_255 // 255).astype(np.uint8)  # Convert back to 0/1 for create_overlay
    masked_road = cv2.bitwise_and(image, image, mask=road_mask_255)  # Image and Image when mask is true, else black. The cv2.bitwise_and accepts RGB images.
    return masked_road
