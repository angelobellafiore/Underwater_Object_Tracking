import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

def simple_features_extractor(image, bbox):
    x, y, w, h = bbox
    cropped = image[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
    if cropped.size == 0:
        return np.zeros(512)

    resized = cv2.resize(cropped, (128, 256))
    flipped=cv2.flip(resized, 1)
    feature=flipped.flatten()[:128]
    return feature

def resnet50_features_extractor(image_array, bbox):
    image = Image.fromarray(image_array).convert("RGB")
    x, y, w, h = bbox
    cropped = image.crop((x, y, x + w, y + h))

    input_tensor = transform(cropped).unsqueeze(0)

    with torch.no_grad():
        features = model(input_tensor)

    return features.numpy().flatten()

def sift_features_extractor(image, bbox):
    x, y, w, h = bbox
    cropped = image[int(y):int(y+h), int(x):int(x+w)]
    if cropped.size == 0:
        return np.zeros(128)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(128)

    feature_vector = np.mean(descriptors, axis=0)
    if feature_vector.shape[0] < 128:
        feature_vector = np.pad(feature_vector, (0, 128 - feature_vector.shape[0]), mode='constant')
    else:
        feature_vector = feature_vector[:128]

    if np.isnan(feature_vector).any():
        return np.zeros(128)

    return feature_vector


""" IF YOU PREFER TO USE 'RESNET-50' AS EXTRACTOR, UNCOMMENT THE FOLLOWING LINES: """
#model = models.resnet50(pretrained=True)
#model.eval()
#transform = transforms.Compose([
#    transforms.Resize(256),
#    transforms.CenterCrop(224),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#])


""" IF YOU PREFER TO USE 'SIFT' AS EXTRACTOR, UNCOMMENT THE FOLLOWING LINE: """
#sift = cv2.SIFT_create()
