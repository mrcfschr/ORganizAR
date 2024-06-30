import pdb

import torch
import clip
from lang_sam import LangSAM

import cv2
import os


class ViewManager:

    def __init__(self, thresh=0.1):
        self._novel = []
        self._length = 0
        self.orb = cv2.ORB.create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.thresh = thresh

    @staticmethod
    def compute_feature_similarity(img1_des, img2_des, matcher):
        matches = matcher.match(img1_des, img2_des)
        valid_matches = [i for i in matches if i.distance < 40]
        # print(len(valid_matches), " ", len(matches), "  ",  len(valid_matches)/len(matches))
        if (len(matches) == 0):
            return 1
        else:
            return len(valid_matches) / len(matches)

    def new_view(self, img) -> (bool, int):
        img = self.crop_view(img)
        # img = cv2.imread(color_file_names[i],cv2.IMREAD_COLOR)
        fp, descrip = self.orb.detectAndCompute(img, None)

        if self._length == 0:
            self._append_image(img, fp, descrip)
            # print("added image Nr.", i)
            # cv2.imwrite(os.path.join(output_path,str(i)+".jpg"),img)
            # print("wrote image Nr. ",i, "to location: ", os.path.join(output_path,str(i)+".jpg") )
            return (True, self._length - 1, img)
        # compare current image to all the recorded images
        isolated = True
        for t in self._novel:
            similarity = ViewManager.compute_feature_similarity(descrip, t["descriptor"], self.matcher)
            if similarity > self.thresh:
                isolated = False
                return False, -1, img
        # print("the minimum similarity for Nr. ", i , " is: ", similarity_min)
        if isolated:
            self._append_image(img, fp, descrip)
            return True, self._length - 1, img
            # print("added image Nr.", i)
            # cv2.imwrite(os.path.join(output_path,str(i)+".jpg"),img)
            # print("wrote image Nr. ",i, "to location: ", os.path.join(output_path,str(i)+".jpg") )

    def pop_view(self, index: int):
        if index < 0:
            raise IndexError
        else:
            self._novel.pop(index)
            self._length -= 1

    def crop_view(self, image):  # TODO perfect overlay with depth? corners cut off still working well
        height, width = image.shape[:2]
        crop_width = int(width * 0.9)
        crop_height = int(height * 0.9)
        x_start = (width - crop_width) // 2
        y_start = (height - crop_height) // 2
        x_end = x_start + crop_width
        y_end = y_start + crop_height

        return image[y_start:y_end, x_start:x_end]

    def _append_image(self, img, fp, des):
        self._novel.append({"image": img, "descriptor": des, "feature": fp})
        self._length += 1




# view_manager = ViewManager()
# for filename in os.listdir(save_path_rgb):
#     if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other extensions if needed
#         img_path = os.path.join(save_path_rgb, filename)
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)

#         if img is not None:
#             is_new, index, img = view_manager.new_view(img)

#             if is_new:
#                 save_filename = f"{index}.jpg"
#                 save_path = os.path.join(save_path_rgb, save_filename)
#                 cv2.imwrite(save_path, img)
#                 print(f"Saved novel image at {save_path}")
#         else:
#             print(f"Failed to read image {img_path}")


def crop_image(image_pil, box):
    left, upper, right, lower = [int(coordinate.item()) for coordinate in box]
    cropped_image = image_pil.crop((left, upper, right, lower))
    return cropped_image

# this class takes in novel views and manages the prompt boxes, keeps track of
# the most similar boxes to the prompt
class PromptBoxManager:
    def __init__(self, prompts1, prompts2):
        self.prompts = prompts1
        self.look_up_prompts = prompts2

        self.device = "mps"
        self.embed_box = torch.empty(0).to(self.device)
        self.num_boxes = 0
        self.number_of_frames = 0
        self.embed_prompt = None
        self.prompt_box = None

        self.frame_index_record = []
        self.current_box_det=None
        self.boxes = torch.empty(0)
        self.prompt_assigned_boxes = None

        self.snapshots = []
        self.output_folder = "./data/debug_faster4/"
        self.previous_assign = None



        self.CLIP_SIM_THRESHOLD = 0.1
        self.DINO_THRESHOLD = 0.5
        self.MIN_FRAME_NUM = 15



        self.model_LangSAM = LangSAM()
        self.model_clip, self.preprocess_clip = clip.load("ViT-B/32", device=self.device)
        #self.model_clip = model_clip.to(self.device)
        self.model_clip.eval()
        self.model_clip = self.model_clip.to(self.device)
        #self.preprocess_clip = self.preprocess_clip.to(self.device)
        self.initialize_prompt_embedings()

        #restrict the number of frames to be stored
        self.frame_capacity = 10


    def initialize_prompt_embedings(self):
        print("initializing prompt embeddings: ", self.look_up_prompts)
        tokens = clip.tokenize(self.look_up_prompts).to(self.device)

        self.embed_prompt = self.model_clip.encode_text(tokens)
        self.embed_prompt = torch.nn.functional.normalize(self.embed_prompt, p=2, dim=1)
        print("initialized prompt embeddings, with shape: ", self.embed_prompt.shape)


    #main function to process the frame
    def new_frame(self, frame,box_threshold=0.3, text_threshold=0.25):
        print("view: ", self.number_of_frames)
        #print("processing frame with timestamp: ", frame["timestamp"])
        image = frame
        boxes = torch.empty(0)
        ph = []
        lo = []
        for i in self.prompts:
            box, confidence, phrases = self.model_LangSAM.predict_dino(image, i, box_threshold,
                                                                         text_threshold)
            #pdb.set_trace()
            boxes = torch.cat((boxes, box), dim=0)
            # ph.append(phrases)
            # lo.append(confidence)
        # print(ph)
        # print(lo)
        # for i, con in enumerate(confidence):
        #     if con > self.DINO_THRESHOLD:
        #         crop_image(image, box[i]).save(self.output_folder + str(self.number_of_frames) + "DINO" + str(confidence[i]) + phrases[i])

        # boxes, confidence, phrases = self.model_LangSAM.predict_dino(image, self.prompts, box_threshold, text_threshold)

        #the fram_index_record records the number of boxes in each frame
        self.frame_index_record.append(len(boxes))
        #register the boxes
        self._add_boxes(boxes, image)
        #calculate the similarity: update the embed_box and prompt_box
        self.calculate_similarity()
        #extract the boxes that are similar to the prompt
        self.assign_boxes()
        self.number_of_frames += 1




    def _add_boxes(self, boxes, image):
        #first crop the image in to patches with the boxes
        image_patches = []
        prepro_images = torch.empty(0).to(self.device)
        for box in boxes:
            cropped = crop_image(image, box)
            image_patches.append(cropped)
            self.snapshots.append(cropped)
        #then preprocess the image_patches
        #image_patches = torch.stack(image_patches)
        #image_patches = image_patches.to(self.device)
        image_embeddings = []
        for im in image_patches:
            im = self.preprocess_clip(im).to(self.device)
            image_embeddings.append(im)

        embed_patches = self.model_clip.encode_image(torch.stack(image_embeddings).to(self.device))
        embed_patches = torch.nn.functional.normalize(embed_patches, p=2, dim=1)
        # print(embed_patches)
        # print(torch.norm(embed_patches, dim=1, p="fro"))

        self.embed_box = torch.cat((self.embed_box, embed_patches), dim=0)
        #print(self.embed_box.shape)

        #update the number of boxes
        self.num_boxes += len(boxes)
        self.boxes = torch.cat((self.boxes, boxes), dim=0)
        print("added boxes, now the number of boxes is: ", self.boxes.shape)


    def calculate_similarity(self):
        #calculate the similarity between the prompt and the embed_box
        self.prompt_box = torch.matmul(self.embed_box, self.embed_prompt.T)
        print("prompt_box shape: ", self.prompt_box.shape)
        print("max value: ", torch.max(self.prompt_box))

        return self.prompt_box

    def assign_boxes(self):
        #extract the boxes that are similar to the prompt
        #get the indices of the boxes that are similar to the prompt
        print(self.prompt_box.shape)
        print("number of boxes:", self.num_boxes)
        value, index = torch.max(self.prompt_box, dim=0)
        mask = value> self.CLIP_SIM_THRESHOLD
        index = torch.where(mask, index, torch.tensor(-1))
        self.prompt_assigned_boxes = index
        box_assign = self._get_index_array(index, self.num_boxes)
        # update the box_det
        self.current_box_det = box_assign
        print("promt_assign: ", self.prompt_assigned_boxes)



    def _get_frame_from_box_index(self, index):
        #get the frame index from the box index
        pass


    def _get_index_array(self,input_tensor, N):

        # Initialize the output tensor filled with -1 (default value)
        output_tensor = torch.full((N,), -1, dtype=torch.int32).to(self.device)
        index = torch.arange(0, len(input_tensor), dtype=torch.int32).to(self.device)

        # Find indices where values in input_tensor are within [0, N-1]
        mask = (input_tensor >= 0) & (input_tensor < N)

        # Update output_tensor at valid indices
        output_tensor[input_tensor[mask]] = index[mask]

        return output_tensor

    def output_det(self):

        mask = self.prompt_assigned_boxes!=-1
        if self.previous_assign is None:

            for i, box_index in enumerate(self.prompt_assigned_boxes):
                if box_index == -1:
                    print("prompt ", self.look_up_prompts[i], " has no assigned box")
                    continue
                print("saving frame ", self.number_of_frames, " with prompt ", self.look_up_prompts[i])
                self.snapshots[box_index].save(
                    self.output_folder + "frame" + str(self.number_of_frames) + self.look_up_prompts[i] + ".png")
        else:
            for i, (box_index, previous_index) in enumerate(zip(self.prompt_assigned_boxes, self.previous_assign)):
                if box_index == -1:
                    print("prompt ", self.look_up_prompts[i], " has no assigned box")
                    continue
                if box_index == previous_index:
                    continue
                print("saving frame ", self.number_of_frames, " with prompt ", self.look_up_prompts[i])
                self.snapshots[box_index].save(
                    self.output_folder + "frame" + str(self.number_of_frames) + self.look_up_prompts[i] + ".png")

        self.previous_assign = self.prompt_assigned_boxes





