
import torch

# Step 1: Create a tensor (matrix)
import torch
import pdb



tt = [0,1,2]
mm =torch.tensor([True, False,False]).numpy()
res = [x for x,b in zip(tt,mm) if b ]
print(res)
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

