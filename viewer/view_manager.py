import cv2
import os 
save_path_rgb = "C:/Users/Marc/Desktop/CS/MARPROJECT/viewer/data/rgb/"
class ViewManager:
    
    def __init__(self, thresh = 0.1):
        self._novel=[]
        self._length=0
        self.orb = cv2.ORB.create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
        self.thresh = thresh

    @staticmethod
    def compute_feature_similarity(img1_des, img2_des, matcher):
        matches  = matcher.match(img1_des, img2_des)
        valid_matches = [i for i in matches if i.distance<40]
        #print(len(valid_matches), " ", len(matches), "  ",  len(valid_matches)/len(matches))
        if(len(matches) == 0):
            return 1
        else:
            return len(valid_matches)/len(matches)

    def new_view(self, img) -> (bool, int):
        img = self.crop_view(img)
        #img = cv2.imread(color_file_names[i],cv2.IMREAD_COLOR)
        fp, descrip = self.orb.detectAndCompute(img,None)
        
        if self._length==0:
            self._append_image(img, fp, descrip)
            #print("added image Nr.", i)
            #cv2.imwrite(os.path.join(output_path,str(i)+".jpg"),img)
            #print("wrote image Nr. ",i, "to location: ", os.path.join(output_path,str(i)+".jpg") )
            return (True, self._length-1,img)
        #compare current image to all the recorded images
        isolated = True
        for t in self._novel:
            similarity = ViewManager.compute_feature_similarity(descrip,t["descriptor"],self.matcher)
            if similarity>self.thresh:
                isolated=False
                return False, -1,img
        #print("the minimum similarity for Nr. ", i , " is: ", similarity_min)
        if isolated:
            self._append_image(img,fp, descrip)
            return True, self._length-1, img
            #print("added image Nr.", i)
            #cv2.imwrite(os.path.join(output_path,str(i)+".jpg"),img)
            #print("wrote image Nr. ",i, "to location: ", os.path.join(output_path,str(i)+".jpg") )
    
    def pop_view(self, index:int):
        if index<0:
            raise IndexError
        else:
            self._novel.pop(index)
            self._length -= 1
    def crop_view(self,imgage): #TODO perfect overlay with depth? corners cut off still working well
        height, width = imgage.shape[:2]
        crop_width = int(width * 0.9)
        crop_height = int(height * 0.9)
        x_start = (width - crop_width) // 2
        y_start = (height - crop_height) // 2
        x_end = x_start + crop_width
        y_end = y_start + crop_height

        return imgage[y_start:y_end, x_start:x_end]

    
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
