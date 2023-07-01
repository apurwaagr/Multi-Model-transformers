from transformers import pipeline

classifier = pipeline(task='zero-shot-image-classification', model='kakaobrain/align-base')
output1 = classifier('https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png',candidate_labels=['animals', 'humans', 'landscape'],)
print("Output for label: " + "['animals', 'humans', 'landscape']")
print()
print(output1)
print()
print()

output2 = classifier('https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png',candidate_labels=['black and white', 'photorealist', 'painting'],)
print("Output for label: " + "['black and white', 'photorealist', 'painting']")
print()
print(output2)

