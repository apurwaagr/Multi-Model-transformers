import requests
from PIL import Image
import torch
from transformers import AlignProcessor, AlignModel

# step-1: download the pre-trained model and its processor
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AlignProcessor.from_pretrained('kakaobrain/align-base')
model = AlignModel.from_pretrained('kakaobrain/align-base')






# step-2: zero-shot image classification
candidate_labels = ['an image of a cat', 'an image of a dog']

# preprocess both the image and text inputs and feed the preprocessed input to the AlignModel
inputs = processor(images=image, text=candidate_labels, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# this is the image-text similarity score
logits_per_image = outputs.logits_per_image  

# we can take the softmax to get the label probabilities
probs = logits_per_image.softmax(dim=1)  
print(probs)





# step -3: Retrieve vision and text embeddings
text_embeds = model.get_text_features(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    token_type_ids=inputs['token_type_ids'],
)
image_embeds = model.get_image_features(
    pixel_values=inputs['pixel_values'],
)







# alternate step: standalong vision and text of ALIGN to retrieve multi-modal embeddings
#ALIGN Text Model
from transformers import AlignTextModel


processor = AlignProcessor.from_pretrained('kakaobrain/align-base')
model = AlignTextModel.from_pretrained('kakaobrain/align-base')

# get embeddings of two text queries
inputs = processor(['an image of a cat', 'an image of a dog'], return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# get the last hidden state and the final pooled output 
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output




#return all hidden states and attention values for align text model

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

# print what information is returned
for key, value in outputs.items():
    print(key)











#ALIGN Vision Model 
print("--------------------Vision Model--------------------")

from transformers import AlignVisionModel


processor = AlignProcessor.from_pretrained('kakaobrain/align-base')
model = AlignVisionModel.from_pretrained('kakaobrain/align-base')

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# print the last hidden state and the final pooled output 
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output


