from torchvision.io.image import decode_image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.transforms.functional import to_pil_image

img = decode_image("./assets/uneven_road.png")
img = img[:3, :, :]  # convert from RGBA to RGB

# Step 1: Initialize model with the best available weights
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and visualize the prediction
prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
# print(class_to_idx)
mask = normalized_masks[0, class_to_idx["__background__"]]
to_pil_image(mask).show()