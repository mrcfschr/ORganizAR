import pdb

import torch
import clip
from lang_sam import LangSAM


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

        self.device = "cuda:0"
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
        self.output_folder = "./data/debug_faster42/"
        self.previous_assign = None

        self.prompt_predict = None
        self.prompt_in_frame = None


        self.unique_objects= ["ultra sound", "c arm", "bed with dummy", "grey shelf"]
        self.multiple_objects = ["chair","backpack", "monitor"]


        self.CLIP_SIM_THRESHOLD = 0.24 #0.25
        self.DINO_THRESHOLD = 0.27 #0.3
        self.MIN_FRAME_NUM = 10



        self.model_LangSAM = LangSAM()
        
        self.model_clip, self.preprocess_clip = clip.load("ViT-L/14", device=self.device)
        #self.model_clip = model_clip.to(self.device)
        self.model_clip.eval()
        #self.model_clip = self.model_clip.to(self.device)
        

        self.frame_masks = []

        self.initialize_prompt_embedings()


    def initialize_prompt_embedings(self):
        self.prompt_predict = {}

        print("initializing prompt embeddings: ", self.look_up_prompts)
        tokens = clip.tokenize(self.look_up_prompts).to(self.device)
        with torch.no_grad():
            self.embed_prompt = self.model_clip.encode_text(tokens)
        self.embed_prompt = torch.nn.functional.normalize(self.embed_prompt, p=2, dim=1)
        #print("initialized prompt embeddings, with shape: ", self.embed_prompt.shape)
        for prompt in self.look_up_prompts:
            self.prompt_predict[prompt] = {"exist": False,
                                           "image": None,
                                           "box": None,
                                           "frame": None,
                                           "confidence": None,
                                           "phrase": None,
                                           }
        self.prompt_in_frame = torch.full((len(self.look_up_prompts),), -1)
        #print("initialize_prompt: ", "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)) 
        


    #main function to process the frame
    def new_frame(self, frame,pv_timestamp, depth_timestamp, depth_image, color_intrinsics, color_extrinsics, depth_extrinsics, box_threshold=0.3, text_threshold=0.25):

        #print("processing frame with timestamp: ", frame["timestamp"])
        print("processing frame Nr. "+ str(self.number_of_frames))
        image = frame
        boxes = torch.empty(0)
        for i in self.prompts:
            with torch.no_grad():
                boxes, confidence, phrases = self.model_LangSAM.predict_dino(image, i, self.DINO_THRESHOLD,
                                                                         self.DINO_THRESHOLD)
            #print("new_frame_beginning: ", "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)) 
        

            if len(boxes) == 0:
                continue

            #pdb.set_trace()
            #boxes = torch.cat((boxes, box), dim=0)
            prompts_index = self.get_promt_from_phrase(phrases)
            mask = prompts_index != -1

            # crop the image and verify by clip
            image_patches = []
           
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
           
            #print("after_embedding: ", "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)) 
            
            
            
            with torch.no_grad():
                embed_patches = self.model_clip.encode_image(torch.stack(image_embeddings))
            #del image_embeddings
            #del image_embeddings_stack
            #torch.cuda.empty_cache()

            #print("del_after_embedding: ", "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)) 
        
            embed_patches = torch.nn.functional.normalize(embed_patches, p=2, dim=1)

            embed_patches = embed_patches[mask]
            embed_prompt = self.embed_prompt[prompts_index[mask]]

            
            similarity = torch.diagonal(embed_prompt @ embed_patches.T, 0)

            mask_thresh = similarity >= self.CLIP_SIM_THRESHOLD
            mask_thresh = mask_thresh.to("cpu")

            end_box = boxes[mask]
            end_box = end_box[mask_thresh]

            #print("after_after_embedding: ", "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)) 
        



            patch_images = [im for im, m in zip(image_patches,mask) if m]
            patch_images = [im for im, m in zip(patch_images,mask_thresh) if m]

            relevant_phrase = [ph for ph, m in zip(phrases, mask) if m]
            relevant_phrase= [ph for ph, m in zip(relevant_phrase, mask_thresh) if m]

            end_confidences = confidence[mask]
            end_confidences = end_confidences[mask_thresh]

            end_prompts_indexes = prompts_index[mask]
            end_prompts_indexes = end_prompts_indexes[mask_thresh]

            end_similarity = similarity[mask_thresh]

            #print("new_frame_middle: ", "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)) 
        

            #get the segmentation masks from images patches
            if len(end_box)!= 0:
                #print(len(end_box))
                masks = self.model_LangSAM.predict_sam(image,end_box)
                #self.frame_masks.append(masks)
                #print(masks.shape)
                #print(masks[0])
                self.formulate_mask_result(pv_timestamp,depth_timestamp,masks,end_prompts_indexes,self.number_of_frames,
                                           frame, depth_image, color_intrinsics,color_extrinsics, depth_extrinsics, end_confidences)
            
                #self.frame_masks.append(None)

            
            for ii, index in enumerate(end_prompts_indexes):
                self.prompt_predict[self.look_up_prompts[index]] = {"exist": True,
                                           "image": patch_images[ii],
                                           "box": end_box[ii],
                                           "frame": self.number_of_frames,
                                           "confidence": end_confidences[ii],
                                           "phrase": relevant_phrase[ii],
                                            "clip_con": end_similarity[ii]}
            self.prompt_in_frame[end_prompts_indexes] = self.number_of_frames
            print("prompts in frame",self.number_of_frames,":", self.prompt_in_frame)
            if self.previous_assign!=None:
                print("previous assign: ", self.previous_assign)
            #print("new_frame_ending: ", "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)) 
        
            
        self.number_of_frames += 1





    # def _add_boxes(self, boxes, image):
    #     #first crop the image in to patches with the boxes
    #     image_patches = []
    #     prepro_images = torch.empty(0).to(self.device)
    #     for box in boxes:
    #         cropped = crop_image(image, box)
    #         image_patches.append(cropped)
    #         self.snapshots.append(cropped)
    #     #then preprocess the image_patches
    #     #image_patches = torch.stack(image_patches)
    #     #image_patches = image_patches.to(self.device)
    #     image_embeddings = []
    #     for im in image_patches:
    #         im = self.preprocess_clip(im).to(self.device)
    #         image_embeddings.append(im)

    #     embed_patches = self.model_clip.encode_image(torch.stack(image_embeddings).to(self.device))
    #     embed_patches = torch.nn.functional.normalize(embed_patches, p=2, dim=1)


    #     self.embed_box = torch.cat((self.embed_box, embed_patches), dim=0)


    #     #update the number of boxes
    #     self.num_boxes += len(boxes)
    #     self.boxes = torch.cat((self.boxes, boxes), dim=0)
    #     print("added boxes, now the number of boxes is: ", self.boxes.shape)


    # def calculate_similarity(self):
    #     #calculate the similarity between the prompt and the embed_box
    #     self.prompt_box = torch.matmul(self.embed_box, self.embed_prompt.T)
    #     print("prompt_box shape: ", self.prompt_box.shape)
    #     print("max value: ", torch.max(self.prompt_box))

    #     return self.prompt_box

    # def assign_boxes(self):
    #     #extract the boxes that are similar to the prompt
    #     #get the indices of the boxes that are similar to the prompt

    #     print("number of boxes:", self.num_boxes)
    #     value, index = torch.max(self.prompt_box, dim=0)
    #     mask = value> self.CLIP_SIM_THRESHOLD
    #     index = torch.where(mask, index, torch.tensor(-1))
    #     self.prompt_assigned_boxes = index
    #     box_assign = self._get_index_array(index, self.num_boxes)
    #     # update the box_det
    #     self.current_box_det = box_assign
    #     print("promt_assign: ", self.prompt_assigned_boxes)



    def _get_frame_from_box_index(self, index):
        #get the frame index from the box index
        pass


    # def _get_index_array(self,input_tensor, N):

    #     # Initialize the output tensor filled with -1 (default value)
    #     output_tensor = torch.full((N,), -1, dtype=torch.int32).to(self.device)
    #     index = torch.arange(0, len(input_tensor), dtype=torch.int32).to(self.device)

    #     # Find indices where values in input_tensor are within [0, N-1]
    #     mask = (input_tensor >= 0) & (input_tensor < N)

    #     # Update output_tensor at valid indices
    #     output_tensor[input_tensor[mask]] = index[mask]

    #     return output_tensor

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
        #print("output_det: ", "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)) 
        

    def get_promt_from_phrase(self, phrases):
        #out put the vector of the index corresponding prompts from phrases
        output = torch.full((len(phrases),), -1)
        for index, prompt in enumerate(self.look_up_prompts):
            for index2, phrase in enumerate(phrases):
                if phrase in prompt:
                    output[index2] = index
        # print("get_prompt: ", "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)) 
        
        return output
    
    
    def formulate_mask_result(self,timestamp_rgbd,timestamp_depth,masks,prompts,frame_index,frame,depth_image, color_intrinsics,color_extrinsics, depth_extrinsics,confidence):
        self.frame_masks.append(
            {"masks": masks,
             "frame_id": frame_index,
             "classes": prompts,
             "time_stamp_rgb": timestamp_rgbd,
             "time_stamp_depth": timestamp_depth,
             "color_pil": frame,
             "depth_image": depth_image,
             "color_intrinsics": color_intrinsics,
             "color_extrinsics": color_extrinsics,
             "depth_extrinsics": depth_extrinsics,
             "confidence": confidence
            }
        )
        # print("formulate mask: ", "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)) 
        
    def save_mask_as_pth(self, path):
        torch.save(self.frame_masks, path)
    
    def restrict_number(self,prompt_index, num_of_detections,frame_stride):
        count = num_of_detections 
        frame_masks = self.frame_masks
        #if all the frame is gone through once and no new mask is added, set to false 
        added = True
        loop_added = False
        record = [0]*len(frame_masks)
        current_window = frame_stride
        list_of_masks = []

        #initialize_mask_list
        for i, frame in enumerate(frame_masks):
            if frame["classes"] is None:
                list_of_masks.append([])
                continue
            list_of_masks.append([True]*len(frame["classes"]))
        
        #repeat the scanning until enough masks are selected and a new mask was added during the last loop
        #in the end the selected will be in the record
        while(count>0 and added):
            #reinitialize added
            added = False
            #reinitialize current window
            current_window = frame_stride
            for i , frame in enumerate(frame_masks):
                
                if i < current_window:
                    if frame["classes"] is None:
                        continue
                    if loop_added:
                        continue
                    if prompt_index in frame["classes"]:
                        budget = record[i]
                        for c in frame["classes"]:
                            #loop though the current frame and check whether to add a new mask
                            if c == prompt_index:
                                if budget==0:
                                    record[i]+=1
                                    count = count - 1
                                    added = True
                                    loop_added=True
                                else:
                                    budget=budget-1
                elif i == current_window:
                    #shift the current window and reinitialize the loop_added
                    current_window += frame_stride
                    loop_added = False
                    if frame["classes"] is None:
                        continue
                    if prompt_index in frame["classes"]:
                        budget = record[i]
                        for c in frame["classes"]:
                            #loop though the current frame and check whether to add a new mask
                            if c == prompt_index:
                                if budget==0:
                                    record[i]+=1
                                    count = count - 1
                                    added = True
                                    loop_added=True
                                else:
                                    budget=budget-1

                else:
                    continue
        print(record)
        #print(list_of_masks)
        for i, frame in enumerate(frame_masks):
            #modify
            if frame["classes"] is None:
                    continue
            if prompt_index in frame["classes"]:
                if record[i] ==0:
                    #delete all masks of class prompt_index
                    for index, c in enumerate(frame["classes"]):
                        if c == prompt_index:
                            list_of_masks[i][index]=False
                            # frame["masks"].pop(index)
                            # frame["classes"].pop(index)
                else:
                    #reserve the first record[i] of masks of class prompt_index, delete the rest
                    for index, c in enumerate(frame["classes"]):
                        if c == prompt_index:
                            if record[i]>0:
                                record[i]=record[i]-1
                                continue
                            else:
                                list_of_masks[i][index]=False
                                #frame["masks"].pop(index)
                                #frame["classes"].pop(index)
        #modify the frame_masks using the list_of_masks
        for index, frame in enumerate(frame_masks):
            if frame["classes"] is None:
                continue
            frame["classes"] = frame["classes"][torch.tensor(list_of_masks[index])]
            #frame["classes"] = [c for c, k in zip(frame["classes"],list_of_masks[index]) if k]
            frame["masks"] = frame["masks"][torch.tensor(list_of_masks[index])]
            #frame["masks"] = [m for m, k in zip(frame["masks"],list_of_masks[index]) if k]
        print("restricted masks: ", list_of_masks)
        # print("restrict_numbers: ", "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)) 
        
            
                
            



    





