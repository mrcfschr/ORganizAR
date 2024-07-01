import pdb

import torch
import clip
from lang_sam import LangSAM
import numpy as np


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
        self.output_folder = "./data/debug_faster16/"
        self.previous_assign = None

        self.prompt_predict = None
        self.prompt_in_frame = None



        self.CLIP_SIM_THRESHOLD = 0.25
        self.DINO_THRESHOLD = 0.3
        self.MIN_FRAME_NUM = 10



        self.model_LangSAM = LangSAM()

        self.model_clip, self.preprocess_clip = clip.load("ViT-L/14", device=self.device)
        #self.model_clip = model_clip.to(self.device)
        self.model_clip.eval()
        self.model_clip = self.model_clip.to(self.device)
        #self.preprocess_clip = self.preprocess_clip.to(self.device)
        self.initialize_prompt_embedings()


    def initialize_prompt_embedings(self):
        self.prompt_predict = {}

        print("initializing prompt embeddings: ", self.look_up_prompts)
        tokens = clip.tokenize(self.look_up_prompts).to(self.device)
        with torch.no_grad():
            self.embed_prompt = self.model_clip.encode_text(tokens)
            self.embed_prompt = torch.nn.functional.normalize(self.embed_prompt, p=2, dim=1)
        print("initialized prompt embeddings, with shape: ", self.embed_prompt.shape)
        for prompt in self.look_up_prompts:
            self.prompt_predict[prompt] = {"exist": False,
                                           "image": None,
                                           "box": None,
                                           "frame": None,
                                           "confidence": None,
                                           "phrase": None,
                                           }
        self.prompt_in_frame = torch.full((len(self.look_up_prompts),), -1)


    #main function to process the frame
    def new_frame(self, frame,box_threshold=0.3, text_threshold=0.25):

        #print("processing frame with timestamp: ", frame["timestamp"])
        print("processing frame Nr. "+ str(self.number_of_frames))
        image = frame
        boxes = torch.empty(0)
        ph = []
        lo = []
        for i in self.prompts:
            with torch.no_grad():
                boxes, confidence, phrases = self.model_LangSAM.predict_dino(image, i, self.DINO_THRESHOLD,
                                                                             self.DINO_THRESHOLD)

            #pdb.set_trace()
            #boxes = torch.cat((boxes, box), dim=0)
            prompts_index = self.get_promt_from_phrase(phrases)
            mask = prompts_index != -1

            # crop the image and verify by clip
            image_patches = []
            prepro_images = torch.empty(0).to(self.device)
            for box in boxes:
                cropped = crop_image(image, box)
                image_patches.append(cropped)

            # then preprocess the image_patches
            # image_patches = torch.stack(image_patches)
            # image_patches = image_patches.to(self.device)
            image_embeddings = []
            for im in image_patches:
                im = self.preprocess_clip(im).to(self.device)
                image_embeddings.append(im)
            with torch.no_grad():
                embed_patches = self.model_clip.encode_image(torch.stack(image_embeddings).to(self.device))
                embed_patches = torch.nn.functional.normalize(embed_patches, p=2, dim=1)

            embed_patches = embed_patches[mask]
            embed_prompt = self.embed_prompt[prompts_index[mask]]

            similarity = torch.diagonal(embed_prompt @ embed_patches.T, 0)

            mask_thresh = similarity >= self.CLIP_SIM_THRESHOLD
            mask_thresh = mask_thresh.to("cpu")

            end_box = boxes[mask]
            end_box = end_box[mask_thresh]




            patch_images = [im for im, m in zip(image_patches,mask) if m]
            patch_images = [im for im, m in zip(patch_images,mask_thresh) if m]

            relevant_phrase = [ph for ph, m in zip(phrases, mask) if m]
            relevant_phrase= [ph for ph, m in zip(relevant_phrase, mask_thresh) if m]

            end_confidences = confidence[mask]
            end_confidences = end_confidences[mask_thresh]

            end_prompts_indexes = prompts_index[mask]
            end_prompts_indexes = end_prompts_indexes[mask_thresh]

            end_similarity = similarity[mask_thresh]

            for ii, index in enumerate(end_prompts_indexes):
                self.prompt_predict[self.look_up_prompts[index]] = {"exist": True,
                                           "image": patch_images[ii],
                                           "box": end_box[ii],
                                           "frame": self.number_of_frames,
                                           "confidence": end_confidences[ii],
                                           "phrase": relevant_phrase[ii],
                                            "clip_con": end_similarity[ii]}
            self.prompt_in_frame[end_prompts_indexes] = self.number_of_frames
            print("prompts in frame: ", self.prompt_in_frame)
            if self.previous_assign!=None:
                print("previous assign: ", self.previous_assign)

        #     ph.append(phrases)
        #     lo.append(confidence)
        # print(ph)
        # print(lo)
        # for i, con in enumerate(confidence):
        #     if con > self.DINO_THRESHOLD:
        #         crop_image(image, box[i]).save(self.output_folder + str(self.number_of_frames) + "DINO" + str(confidence[i]) + phrases[i]+".png")

        # boxes, confidence, phrases = self.model_LangSAM.predict_dino(image, self.prompts, box_threshold, text_threshold)

        #the fram_index_record records the number of boxes in each frame
        # self.frame_index_record.append(len(boxes))
        # #register the boxes
        # self._add_boxes(boxes, image)
        # #calculate the similarity: update the embed_box and prompt_box
        # self.calculate_similarity()
        # #extract the boxes that are similar to the prompt
        # self.assign_boxes()
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
        with torch.no_grad():
            embed_patches = self.model_clip.encode_image(torch.stack(image_embeddings).to(self.device))
            embed_patches = torch.nn.functional.normalize(embed_patches, p=2, dim=1)

        self.embed_box = torch.cat((self.embed_box, embed_patches), dim=0)


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

        #mask = self.prompt_assigned_boxes!=-1
        if self.previous_assign is None:
            for i, box_index in enumerate(self.prompt_in_frame):
                prompt = self.look_up_prompts[i]
                if box_index == -1:
                    print("prompt ", prompt, " has no assigned box")
                    continue

                print("saving frame ", self.number_of_frames, " with prompt ", prompt)
                self.prompt_predict[prompt]["image"].save(
                    self.output_folder + "frame" + str(
                        self.prompt_predict[prompt]["frame"]) + self.prompt_predict[prompt]["phrase"] + "_con" + str(
                        self.prompt_predict[prompt]["confidence"].item()) +"_sim"+str(self.prompt_predict[prompt]["clip_con"].item())+ ".png")
        else:
            for i, (box_index, previous_index) in enumerate(zip(self.prompt_in_frame, self.previous_assign)):
                prompt = self.look_up_prompts[i]
                if box_index == -1:
                    print("prompt ", prompt, " has no assigned box")
                    continue
                if box_index == previous_index:
                    continue
                print("saving frame ", self.number_of_frames, " with prompt ", prompt, "_confidence", self.prompt_predict[prompt]["confidence"])
                self.prompt_predict[prompt]["image"].save(
                    self.output_folder + "frame" + str(self.prompt_predict[prompt]["frame"]) + self.prompt_predict[prompt]["phrase"]+"_con"+str(self.prompt_predict[prompt]["confidence"].item())+"_sim"+str(self.prompt_predict[prompt]["clip_con"].item())+".png")
        self.previous_assign = torch.clone(self.prompt_in_frame)

    def get_promt_from_phrase(self, phrases):
        #out put the vector of the index corresponding prompts from phrases
        output = torch.full((len(phrases),), -1)
        for index, prompt in enumerate(self.look_up_prompts):
            for index2, phrase in enumerate(phrases):
                if phrase in prompt:
                    output[index2] = index
        return output

    def _segment_mask(self, image, boxes):
        #get the bounding box and the patch image as input, save the bounding box mask in the original image
        masks = self.model_LangSAM.predict_sam(image, boxes)
        masks = masks.squeeze(1)
        masks_np = [mask.cpu().detach().numpy() for mask in masks]
        masks_np = [(mask * 255).astype(np.uint8) for mask in masks_np]
        combined_mask = np.zeros(masks_np[0].shape, dtype=np.uint8)
        for mask in masks_np:
            combined_mask = np.maximum(combined_mask, mask)
        return combined_mask






