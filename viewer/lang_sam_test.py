from lang_sam import LangSAM
from PIL import Image
import time

model_LangSAM = LangSAM()
image_pil_timer = Image.open("/mnt/c/Users/Marc/Desktop/CS/MARPROJECT/viewer/test_timer.png")
image_pil_bed = Image.open("/mnt/c/Users/Marc/Desktop/CS/MARPROJECT/viewer/test_bed.png")

images = {'timer': image_pil_timer, 'bed': image_pil_bed}
box_threshold = 0.3
text_threshold = 0.25
masks, boxes, phrases, logits = model_LangSAM.predict(image_pil_timer, "timer") #run once because first is slower

for key, image in images.items():
    start_time = time.time()
    masks, boxes, phrases, logits = model_LangSAM.predict(image, key)
    print("Inference Time {}: ".format(key), time.time() - start_time, "seconds")

    start_time = time.time()
    boxes, phrases, logits = model_LangSAM.predict_dino(image, key, box_threshold, text_threshold)
    print("Dino: Inference Time {}: ".format(key), time.time() - start_time, "seconds")

    start_time = time.time()
    masks = model_LangSAM.predict_sam(image, boxes)
    masks = masks.squeeze(1)
    print("SAM: Inference Time {}: ".format(key), time.time() - start_time, "seconds")
