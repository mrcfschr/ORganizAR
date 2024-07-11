
import torch

# Step 1: Create a tensor (matrix)
import torch
import pdb
import open3d as o3d



def restrict_number(frame_masks,prompt_index, num_of_detections,frame_stride):
    count = num_of_detections 
    
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
    print(list_of_masks)
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
        frame["classes"] = [c for c, k in zip(frame["classes"],list_of_masks[index]) if k]
        frame["masks"] = [m for m, k in zip(frame["masks"],list_of_masks[index]) if k]


path = '/data/projects/medar_smart/ORganizAR/viewer/recorded_data/dataset1seg_mask.pth'
seg_mask = torch.load(path)
print([0]*8)


prompts_lookup = ["A table with cloth on it","bed","a grey shelf with a pole on it","c arm, which is a medical machine with a large c shaped metal arm","backpack"," ultra sound machine, that has flashlight shape probe attached and a machine tower","chair","computer monitor"]
print(". ".join(prompts_lookup))
for i, p in enumerate(prompts_lookup):
    print(i, " ", p)


test_data = [{'masks': [0, 0, 0], 'classes': [2, 2, 6]},
 {'masks': None, 'classes': None},
 {'masks': [0, 0, 0, 0], 'classes': [5, 1, 2, 3]},
 {'masks': [0, 0, 0, 0], 'classes': [5, 1, 4, 4]},
 {'masks': [0], 'classes': [5]},
 {'masks': [0, 0, 0, 0], 'classes': [1, 5, 1, 3]},
 {'masks': [0], 'classes': [4]},
 {'masks': None, 'classes': None},
 {'masks': None, 'classes': None},
 {'masks': [0], 'classes': [3]},
 {'masks': [0, 0, 0], 'classes': [4, 4, 0]},
 {'masks': [0, 0, 0, 0], 'classes': [6, 1, 4, 0]},
 {'masks': [0], 'classes': [2]},
 {'masks': None, 'classes': None},
 {'masks': [0, 0], 'classes': [5, 0]},
 {'masks': [0], 'classes': [4]},
 {'masks': [0, 0], 'classes': [5, 6]},
 {'masks': None, 'classes': None},
 {'masks': [0], 'classes': [1]},
 {'masks': [0], 'classes': [3]},
 {'masks': [0, 0, 0], 'classes': [5, 2, 4]},
 {'masks': None, 'classes': None},
 {'masks': [0, 0, 0], 'classes': [6, 4, 2]},
 {'masks': None, 'classes': None},
 {'masks': [0, 0, 0, 0], 'classes': [4, 5, 6, 4]},
 {'masks': [0, 0, 0], 'classes': [1, 6, 4]},
 {'masks': [0, 0], 'classes': [6, 0]},
 {'masks': [0], 'classes': [3]},
 {'masks': [0, 0, 0, 0], 'classes': [0, 6, 0, 4]},
 {'masks': [0, 0], 'classes': [3, 1]}]


test_result = [
 {'masks': [0, 0, 0], 'classes': [2, 2, 6]},
 {'masks': None, 'classes': None},
 {'masks': [0, 0, 0, 0], 'classes': [5, 1, 2, 3]},
 {'masks': [0, 0, 0, 0], 'classes': [5, 1, 4, 4]},
 {'masks': [0], 'classes': [5]},

 {'masks': [0, 0, 0, 0], 'classes': [1, 5, 1, 3]},
 {'masks': [0], 'classes': [4]},
 {'masks': None, 'classes': None},
 {'masks': None, 'classes': None},
 {'masks': [0], 'classes': [3]},

 {'masks': [0, 0, 0], 'classes': [4,4, 0]},
 {'masks': [0, 0, 0], 'classes': [6, 1, 0]},
 {'masks': [0], 'classes': [2]},
 {'masks': None, 'classes': None},
 {'masks': [0, 0], 'classes': [5, 0]},

 {'masks': [0], 'classes': [4]},
 {'masks': [0, 0], 'classes': [5, 6]},
 {'masks': None, 'classes': None},
 {'masks': [0], 'classes': [1]},
 {'masks': [0], 'classes': [3]},

 {'masks': [0, 0, 0], 'classes': [5, 2, 4]},
 {'masks': None, 'classes': None},
 {'masks': [0, 0], 'classes': [6, 2]},
 {'masks': None, 'classes': None},
 {'masks': [0, 0], 'classes': [ 5, 6]},

 {'masks': [0, 0, 0], 'classes': [1, 6, 4]},
 {'masks': [0, 0], 'classes': [6, 0]},
 {'masks': [0], 'classes': [3]},
 {'masks': [0, 0, 0], 'classes': [0, 6, 0]},
 {'masks': [0, 0], 'classes': [3, 1]}
 ]


#restrict_number(test_data,4,8,5)
print(test_data)

#assert test_data == test_result



# logical_or = torch.tensor([[1,0,0,0,1],[1,1,0,0,0]]) == 1
# logical_or = logical_or.numpy()
# pdb.set_trace()
# print(logical_or)
# print(torch.any(logical_or,0))

#pdb.set_trace()
# print(type(tt))
#
#
# print(type(mm))
# #print(mm.shape)
# print(tt)
# print(tt[mm])
# def custom_function(input_list, N):
#     # Convert input_list to a PyTorch tensor
#     input_tensor = torch.tensor(input_list)
#
#     # Initialize the output tensor filled with -1 (default value)
#     output_tensor = torch.full((N,), -1, dtype=torch.int32)
#     index = torch.arange(0, len(input_tensor), dtype=torch.int32)
#     pdb.set_trace()
#     # Find indices where values in input_tensor are within [0, N-1]
#     mask = (input_tensor >= 0) & (input_tensor < N)
#
#
#     # Update output_tensor at valid indices
#     output_tensor[input_tensor[mask]] = index[mask]
#
#     return output_tensor.tolist()
#
#
# # Example usage:
# input_list = [-1, -1, 5, 3]
# N = 6
# output = custom_function(input_list, N)
# print(output)  # Output: [-1, -1, 3, -1, 0, 2]



##
csv_path = '/data/projects/medar_smart/ORganizAR/viewer/recorded_data/dataset_test_3/rm_depth_longthrow.csv'
import pandas as pd

df = pd.read_csv(csv_path)
print(df)
print(df.columns)
print(df['timestamp'])
print("rows: ", df.shape)


stttt = "....."
sssss = stttt.join("mmmmm")
print(sssss)
print(stttt)