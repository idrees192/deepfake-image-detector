import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

IMG_SIZE = 224

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess(image):
    img = np.array(image)
    return transform(image=img)["image"].unsqueeze(0)
