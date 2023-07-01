import requests
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('kakaobrain/vit-large-patch16-384')
model = ViTForImageClassification.from_pretrained('kakaobrain/vit-large-patch16-384')


# preprocess image or list of images
inputs = processor(images=image, return_tensors="pt")

# inference
with torch.no_grad():
    outputs = model(**inputs)

# apply SoftMax to logits to compute the probability of each class
preds = torch.nn.functional.softmax(outputs.logits, dim=-1)

# print the top 5 class predictions and their probabilities
top_class_preds = torch.argsort(preds, descending=True)[0, :5]

for c in top_class_preds:
    print(f"{model.config.id2label[c.item()]} with probability {round(preds[0, c.item()].item(), 4)}")
