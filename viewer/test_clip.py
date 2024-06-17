import torch
import clip
from PIL import Image

IF_REMOTE = True
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

if IF_REMOTE:
    image_path = "/medar_smart/ORganizAR/viewer/test_timer.png"
else:
    image_path = "/mnt/c/Users/Marc/Desktop/CS/MARPROJECT/viewer/test_timer.png"	
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = clip.tokenize(["a timer"]).to(device)

with torch.no_grad():
    # image_features = model.encode_image(image)
    # text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text) #Given a batch of images and a batch of text tokens, returns two Tensors, 
    #containing the logit scores corresponding to each image and text input. 
    # The values are cosine similarities between the corresponding image and text features, times 100.
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
print(logits_per_image, logits_per_text)  
print("Label probs:", probs)  