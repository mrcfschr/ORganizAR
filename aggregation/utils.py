import numpy as np

# import open3d as o3d
import cv2
import yaml

import matplotlib.pyplot as plt
import copy
import torch
import os
import argparse


from tqdm import tqdm
import sys
from typing import List, Dict, Tuple


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

"""
0. Data Preparation
"""

"""
1. Project 2d masks to 3d point cloud
"""


def project_2d_to_3d_single_frame(
    backprojected_3d_masks,
    cam_intr,
    depth_im,
    cam_pose,
    depth_pose,
    masks_2d,
    pcd_3d,
    photo_width,
    photo_height,
    depth_intr=np.array(
            [
                [174.79669, 0.0, 156.74863],
                [0.0, 181.98889, 172.73248],
                [
                    0.0,
                    0.0,
                    1.0,
                ],
            ]
        ),
    depth_thresh=0.08,
    depth_scale=1000,
):
    """project 2d masks to 3d point cloud

    args:
        backprojected_3d_masks: add results to this dict
        cam_intr: 3x3 camera intrinsic matrix of rgb camera
        depth_im: depth image of current frame readed by cv2.imread
        cam_pose: camera pose of current frame (camera to world)
        depth_pose: depth camera pose of current frame (depth camera to world)
        masks_2d: 2d masks of current frame
            {
                "frame_id": frame_id,
                "segmented_frame_masks": segmented_frame_masks.to(torch.bool),  # (M, 1, W, H)
                "confidences": confidences,
                "labels": labels,
            }
        pcd_3d: numpy array of 3d point cloud, shape (3, N)
        photo_width: width of the rgb photo
        photo_height: height of the rgb photo
        depth_thresh: depth threshold for visibility check
        depth_scale: depth scale for depth image

    return:
        backprojected_3d_masks: backprojected 3d masks
            {
                "ins": [],  # (Ins, N)
                "conf": [],  # (Ins, )
                "final_class": [],  # (Ins,)
            }
    """

    # process 2d masks
    segmented_frame_masks = masks_2d["segmented_frame_masks"].to(device)  # (M, 1, W, H)
    confidences = masks_2d["confidences"].to(device)  # (M, )
    labels = masks_2d["labels"]  # List[str] (M,)
    pred_masks = segmented_frame_masks.squeeze(dim=1).cpu().numpy()  # (M, H, W)

    # process 3d point cloud
    scene_pts = copy.deepcopy(pcd_3d)
    scene_pts_depth = copy.deepcopy(pcd_3d)

    # process depth image
    depth_im = cv2.resize(depth_im, (photo_width, photo_height))
    depth_im = torch.tensor(depth_im, device=device) / depth_scale

    # convert 3d points to camera coordinates
    scene_pts = (np.linalg.inv(cam_pose) @ scene_pts).T[:, :3]  # (N, 3)
    scene_pts_depth = (np.linalg.inv(depth_pose) @ scene_pts_depth).T[:, :3]  # (N, 3)

    # convert tensor
    scene_pts = torch.tensor(scene_pts).to(device)  # (N, 3)
    scene_pts_depth = torch.tensor(scene_pts_depth).to(device)  # (N, 3)
    """Start projectiom"""
    """Method 1"""
    # # project 3d points to current 2d rgb frame
    # projected_pts = compute_projected_pts_tensor(scene_pts, cam_intr)

    # # check visibility of projected 2d points
    # visibility_mask = compute_visibility_mask_tensor(
    #     scene_pts_depth, projected_pts, depth_im, depth_thresh=depth_thresh
    # )

    # # Select visible 3D points in 2D masks as backprojected 3D masks
    # masked_pts = compute_visible_masked_pts_tensor(
    #     scene_pts, projected_pts, visibility_mask, pred_masks
    # )

    """Method 2 : seperated depth and rgb camera"""
    # # check depth for each masked 3d points
    # visibility_mask = check_depth_for_3d_masked_pts(
    #     scene_pts_depth, depth_im, depth_intr, depth_thresh=depth_thresh
    # )  # (N,)

    # # project 3d points to current 2d rgb frame and get masked 3d points
    # masked_pts = compute_projected_masked_pts_tensor_rgb(
    #     scene_pts_depth, cam_intr, pred_masks, visibility_mask
    # )  # (M, N)

    """Method 3 : first hit"""
    masked_pts = firsthit_compute_projected_masked_pts_tensor_rgb(
        scene_pts, cam_intr, pred_masks, depth_thresh
    )  # (M, N)

    masked_pts = torch.tensor(masked_pts).to(device)  # (M, N)
    mask_area = torch.sum(masked_pts, dim=1).detach().cpu().numpy()  # (M,)
    print(
        "number of 3d mask points:",
        mask_area,
        "number of 2d masks:",
        pred_masks.sum(axis=(1, 2)),
    )

    # add backprojected 3d masks to backprojected_3d_masks
    for i in range(masked_pts.shape[0]):
        backprojected_3d_masks["ins"].append(masked_pts[i])
        backprojected_3d_masks["conf"].append(confidences[i])
        backprojected_3d_masks["final_class"].append(labels[i])

    return backprojected_3d_masks

def firsthit_compute_projected_masked_pts_tensor_rgb(pts, cam_intr, pred_masks, depth_thresh=0.08):
    # map 3d pointclouds in camera coordinates system to 2d
    # bool mask of pointcloud (M, N) which has xy within mask

    pts = pts.T  # (3, N)
    # print("cam_int", cam_intr)
    projected_pts = cam_intr @ pts / pts[2]  # (3, N)
    projected_pts = projected_pts[:2].T  # (N, 2)
    projected_pts = (torch.round(projected_pts)).to(torch.int64)


    print("DEBUG", projected_pts.shape, pred_masks.shape)
    inbounds = (
        (projected_pts[:, 0] >= 0)
        & (projected_pts[:, 0] < pred_masks.shape[2])
        & (projected_pts[:, 1] >= 0)
        & (projected_pts[:, 1] < pred_masks.shape[1])
    )  # (N,)


    # first hit as visible
    visibility_mask = torch.zeros(projected_pts.shape[0], device=device).to(torch.bool)
    # visible if depth is positive
    # visibility_mask = pts[2] > 0
    # calculate first hit in 3d space for each 2d point
    lowest_depth = torch.ones((pred_masks.shape[1], pred_masks.shape[2]), device=device) * 1000 # 1000 is just a large number that won't be reached
    for i in range(projected_pts.shape[0]):
        # print("DEBUG", i, projected_pts[i])
        if not inbounds[i]:
            # print("skip")
            continue
        x, y = projected_pts[i]
        if lowest_depth[y][x] >=1000 and pts.T[i][2] > 0:
            lowest_depth[y][x] = pts.T[i][2]
        elif pts.T[i][2] < lowest_depth[y][x] and pts.T[i][2] > 0:
            lowest_depth[y][x] = pts.T[i][2]
    print("DEBUG lowest depth", lowest_depth.max(), lowest_depth.min())

    """cause we have ~280000 2d points, which is much more than 3d points in this view, first hit also includes background, 
    so we need to filter out background points"""
    # convert to tensor
    lowest_depth = torch.tensor(lowest_depth, device=device)
    # perfrom min pooling of 3x3 kernel to filter out background points
    lowest_depth = lowest_depth * -1
    lowest_depth = torch.nn.functional.max_pool2d(lowest_depth.unsqueeze(0).unsqueeze(0), 15 , stride=1, padding=7, ).squeeze(0).squeeze(0) * -1
    print("DEBUG lowest depth", lowest_depth.max(), lowest_depth.min())
    


    for i in range(projected_pts.shape[0]):
        if not inbounds[i]:
            continue
        x, y = projected_pts[i]
        if torch.abs(pts.T[i][2] - lowest_depth[y][x]) < depth_thresh and pts.T[i][2] > 0:
            visibility_mask[i] = True
    print("DEBUG visibility mask", visibility_mask.sum())

    # return masked 3d points
    N = projected_pts.shape[0]
    M, _, _ = pred_masks.shape  # (M, H, W)
    # print("DEBUG M value", M)
    projected_pts = projected_pts[inbounds]  # (X, 2)
    masked_pts = torch.zeros((M, N), dtype=torch.bool, device=device)
    for m in range(M):
        x, y = projected_pts.T  # (X,)
        mask_check = pred_masks[m, y, x]  # (X,)
        # print("debug mask", masked_pts[m].shape, mask_check.shape)
        masked_pts[m, inbounds] = mask_check # shape (M, N)
        masked_pts[m, ~visibility_mask] = False

    return masked_pts


def check_depth_for_3d_masked_pts(depth_pts, depth_im, depth_intr, depth_thresh=0.08):
    # compare z in camera coordinates and depth image
    # to check if there projected points are visible
    im_h, im_w = depth_im.shape

    visibility_mask = np.zeros(depth_pts.shape[0]).astype(np.bool8)
    projected_pts_depth = depth_intr @ depth_pts.T  # (3, N)
    projected_pts_depth = projected_pts_depth[:2].T  # (N, 2)
    projected_pts_depth = (np.round(projected_pts_depth)).astype(np.int64)

    inbounds = (
        (projected_pts_depth[:, 0] >= 0)
        & (projected_pts_depth[:, 0] < im_w)
        & (projected_pts_depth[:, 1] >= 0)
        & (projected_pts_depth[:, 1] < im_h)
    )  # (N,)
    projected_pts_depth = projected_pts_depth[inbounds]  # (X, 2)
    depth_check = (
        depth_im[projected_pts_depth[:, 1], projected_pts_depth[:, 0]] != 0
    ) & (
        np.abs(  # check if the depth is within the threshold
            depth_pts[inbounds][:, 2]
            - depth_im[projected_pts_depth[:, 1], projected_pts_depth[:, 0]]
        )
        < depth_thresh
    )

    visibility_mask[inbounds] = depth_check
    return visibility_mask  # (N,)

def compute_projected_masked_pts_tensor_rgb(pts, cam_intr, pred_masks, visibility_mask):
    # map 3d pointclouds in camera coordinates system to 2d
    # bool mask of pointcloud (M, N) which has xy within mask

    pts = pts.T  # (3, N)
    # print("cam_int", cam_intr)
    projected_pts = cam_intr @ pts / pts[2]  # (3, N)
    projected_pts = projected_pts[:2].T  # (N, 2)
    projected_pts = (np.round(projected_pts)).astype(np.int64)

    #    # first hit as visible
    #     visibility_mask = np.zeros(projected_pts.shape[0]).astype(np.bool8)
    #     inbounds = (
    #         (projected_pts[:, 0] >= 0)
    #         & (projected_pts[:, 0] < pred_masks.shape[2])
    #         & (projected_pts[:, 1] >= 0)
    #         & (projected_pts[:, 1] < pred_masks.shape[1])
    #     )  # (N,)
    #     first_hits =

    inbounds = (
        (projected_pts[:, 0] >= 0)
        & (projected_pts[:, 0] < pred_masks.shape[2])
        & (projected_pts[:, 1] >= 0)
        & (projected_pts[:, 1] < pred_masks.shape[1])
    )  # (N,)

    # return masked 3d points
    N = projected_pts.shape[0]
    M, _, _ = pred_masks.shape  # (M, H, W)
    # print("DEBUG M value", M)
    projected_pts = projected_pts[inbounds]  # (X, 2)
    masked_pts = np.zeros((M, N), dtype=np.bool_)
    for m in range(M):
        x, y = projected_pts.T  # (X,)
        mask_check = pred_masks[m, y, x]  # (X,)
        # print("debug mask", masked_pts[m].shape, mask_check.shape)
        masked_pts[m, inbounds] = mask_check
        masked_pts[m, ~visibility_mask] = False

    return masked_pts



def compute_projected_pts_tensor(pts, cam_intr):
    # map 3d pointclouds in camera coordinates system to 2d
    # pts shape (N, 3)

    pts = pts.T  # (3, N)
    # print("cam_int", cam_intr)
    projected_pts = cam_intr @ pts / pts[2]  # (3, N)
    # print("pts0", pts[:,0])
    # print("projected_pts0", (cam_intr @ pts[:,0]).astype(np.int64))
    projected_pts = projected_pts[:2].T  # (N, 2)
    projected_pts = (np.round(projected_pts)).astype(np.int64)
    return projected_pts


def compute_visibility_mask_tensor(pts, projected_pts, depth_im, depth_thresh=0.08):
    # compare z in camera coordinates and depth image
    # to check if there projected points are visible
    im_h, im_w = depth_im.shape

    visibility_mask = np.zeros(projected_pts.shape[0]).astype(np.bool8)
    inbounds = (
        (projected_pts[:, 0] >= 0)
        & (projected_pts[:, 0] < im_w)
        & (projected_pts[:, 1] >= 0)
        & (projected_pts[:, 1] < im_h)
    )  # (N,)
    projected_pts = projected_pts[inbounds]  # (X, 2)
    depth_check = (depth_im[projected_pts[:, 1], projected_pts[:, 0]] != 0) & (
        np.abs(pts[inbounds][:, 2] - depth_im[projected_pts[:, 1], projected_pts[:, 0]])
        < depth_thresh
    )

    visibility_mask[inbounds] = depth_check
    return visibility_mask  # (N,)


def compute_visible_masked_pts_tensor(
    scene_pts, projected_pts, visibility_mask, pred_masks
):
    # return masked 3d points
    N = scene_pts.shape[0]
    M, _, _ = pred_masks.shape  # (M, H, W)
    # print("DEBUG M value", M)
    masked_pts = np.zeros((M, N), dtype=np.bool_)
    visiable_pts = projected_pts[visibility_mask]  # (X, 2)
    for m in range(M):
        x, y = visiable_pts.T  # (X,)
        mask_check = pred_masks[m, y, x]  # (X,)
        masked_pts[m, visibility_mask] = mask_check

    return masked_pts


"""
2. Aggregating 3d masks
"""


def aggregate(
    backprojected_3d_masks: dict, iou_threshold=0.25, feature_similarity_threshold=0.75
) -> dict:
    """
    calculate iou
    calculate feature similarity

    if iou >= threshold and feature similarity >= threshold:
        aggregate
    else:
        create new mask
    """
    labels = backprojected_3d_masks["final_class"]  # List[str]
    semantic_matrix = calculate_feature_similarity(labels)

    ins_masks = backprojected_3d_masks["ins"].to(device)  # (Ins, N)
    iou_matrix = calculate_iou(ins_masks)

    confidences = backprojected_3d_masks["conf"].to(device)  # (Ins, )

    merge_matrix = semantic_matrix & (
        iou_matrix > iou_threshold
    )  # dtype: bool (Ins, Ins)

    # aggregate masks with high iou
    (
        aggregated_masks,
        aggregated_confidences,
        aggregated_labels,
        mask_indeces_to_be_merged,
    ) = merge_masks(ins_masks, confidences, labels, merge_matrix)

    # solve overlapping
    final_masks, masked_counts = solve_overlapping(
        aggregated_masks, mask_indeces_to_be_merged, backprojected_3d_masks
    )

    return {
        "ins": final_masks,  # torch.tensor (Ins, N)
        "conf": aggregated_confidences,  # torch.tensor (Ins, )
        "final_class": aggregated_labels,  # List[str] (Ins,)
    }, masked_counts


def calculate_iou(ins_masks: torch.Tensor) -> torch.Tensor:
    """calculate iou between all masks

    args:
        ins_masks: torch.tensor (Ins, N)

    return:
        iou_matrix: torch.tensor (Ins, Ins)
    """
    ins_masks = ins_masks.float()
    intersection = torch.matmul(ins_masks, ins_masks.T)  # (Ins, Ins)
    union = (
        torch.sum(ins_masks, dim=1).unsqueeze(1)
        + torch.sum(ins_masks, dim=1).unsqueeze(0)
        - intersection
    )
    iou_matrix = intersection / union
    return iou_matrix


def calculate_feature_similarity(labels: List[str]) -> torch.Tensor:
    """calculate feature similarity between all masks

    args:
        labels: list[str]

    return:
        feature_similarity_matrix: torch.tensor (Ins, Ins)
    """  # TODO: add clip feature similarity
    feature_similarity_matrix = torch.zeros(len(labels), len(labels), device=device)
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            if labels[i] == labels[j]:
                feature_similarity_matrix[i, j] = 1
                feature_similarity_matrix[j, i] = 1

    # convert to boolean
    feature_similarity_matrix = feature_similarity_matrix.bool()
    return feature_similarity_matrix


def merge_masks(
    ins_masks: torch.Tensor,
    confidences: torch.Tensor,
    labels: List[str],
    merge_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[List[int]]]:

    # find masks to be merged
    merge_matrix = merge_matrix.float()
    mask_indeces_to_be_merged = find_unconnected_subgraphs_tensor(merge_matrix)
    print("masks_to_be_merged", mask_indeces_to_be_merged)

    # merge masks
    aggregated_masks = []
    aggregated_confidences = []
    aggregated_labels = []
    for mask_indeces in mask_indeces_to_be_merged:

        if mask_indeces == []:
            continue

        mask = torch.zeros(ins_masks.shape[1], dtype=torch.bool, device=device)
        conf = []
        for index in mask_indeces:
            mask |= ins_masks[index]
            conf.append(confidences[index])
        aggregated_masks.append(mask)
        aggregated_confidences.append(sum(conf) / len(conf))
        aggregated_labels.append(labels[mask_indeces[0]])

    # convert type
    aggregated_masks = torch.stack(aggregated_masks)  # (Ins, N)
    aggregated_confidences = torch.tensor(aggregated_confidences)  # (Ins, )

    return (
        aggregated_masks,
        aggregated_confidences,
        aggregated_labels,
        mask_indeces_to_be_merged,
    )


def find_unconnected_subgraphs_tensor(adj_matrix: torch.Tensor) -> List[List[int]]:
    num_nodes = adj_matrix.size(0)
    # Create an identity matrix for comparison
    identity = torch.eye(num_nodes, dtype=torch.float32)
    # Start with the adjacency matrix itself
    reachability_matrix = adj_matrix.clone()

    # Repeat matrix multiplication to propagate connectivity
    for _ in range(num_nodes):
        reachability_matrix = torch.matmul(reachability_matrix, adj_matrix) + adj_matrix
        reachability_matrix = torch.clamp(reachability_matrix, 0, 1)

    # Identify unique connected components
    components = []
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    for i in range(num_nodes):
        if not visited[i]:
            component_mask = reachability_matrix[i] > 0
            component = torch.nonzero(component_mask, as_tuple=False).squeeze().tolist()
            # Ensure component is a list even if it's a single element
            component = [component] if isinstance(component, int) else component
            components.append(component)
            visited[component_mask] = True

    return components


def solve_overlapping(
    aggregated_masks: torch.Tensor,
    mask_indeces_to_be_merged: List[List[int]],
    backprojected_3d_masks: dict,
) -> torch.Tensor:
    """
    solve overlapping among all masks
    """
    # mask_counts for filtering (considering classes)
    mask_counts = torch.zeros(
        aggregated_masks.shape[1], dtype=torch.int32, device=device
    )  # shape (N,)

    # number of aggrated inital masks in each aggregated mask
    num_masks = [len(mask_indeces) for mask_indeces in mask_indeces_to_be_merged]

    # find overlapping masks in aggregated_masks
    overlapping_masks = []
    for i in range(len(aggregated_masks)):
        for j in range(i + 1, len(aggregated_masks)):
            if torch.any(aggregated_masks[i] & aggregated_masks[j]):
                overlapping_masks.append((i, j))

    # only keep overlapped points for masks aggregated from more masks
    for i, j in overlapping_masks:
        # solve overlapping
        if num_masks[i] > num_masks[j]:
            aggregated_masks[j] &= ~aggregated_masks[i]
        else:
            aggregated_masks[i] &= ~aggregated_masks[j]

    # update mask_counts
    for i in range(len(aggregated_masks)):
        # update mask_counts for mask i
        for index in mask_indeces_to_be_merged[i]:
            mask_counts += (
                backprojected_3d_masks["ins"][index] & aggregated_masks[i]
            ).int()

    return aggregated_masks, mask_counts


"""
3. Filtering 3d masks
"""


def filter(
    aggregated_3d_masks,
    masked_counts,
    if_occurance_threshold=True,
    occurance_thres=0.3,
    if_detection_rate_threshold=False,
    detection_rate_thres=0.35,
    small_mask_thres=50,
    filtered_mask_thres=0.5,
):
    """filter masks

    args:
        aggregated_3d_masks: aggregated 3d masks
            {
                "ins": [],  # (Ins, N)
                "conf": [],  # (Ins, )
                "final_class": [],  # (Ins,)
            }
        masked_counts: counts of being detected as part of a mask for everypoint  (N)
        if_occurance_threshold: if apply occurance filter
        occurance_thres: points bottom `occurance_thres` percentage of occurance will be filtered out
        small_mask_thres: masks with less than `small_mask_thres` points will be filtered out
        filtered_mask_thres: masks with less than `filtered_mask_thres` percentage of points after filtering will be filtered out



    """

    if if_occurance_threshold:
        point_mask = occurance_filter(masked_counts, occurance_thres)
    elif if_detection_rate_threshold:
        point_mask = detection_rate_thres
    else:
        print("No filtering applied")
        return aggregated_3d_masks

    # apply filtering on aggregated_3d_masks["ins"]
    num_ins_points_before_filtering = (
        aggregated_3d_masks["ins"].sum(dim=1).cpu()
    )  # (Ins,)
    aggregated_3d_masks["ins"] &= point_mask.unsqueeze(0)  # (Ins, N)
    num_ins_points_after_filtering = (
        aggregated_3d_masks["ins"].sum(dim=1).cpu()
    )  # (Ins,)

    # delete the masks with less than 1/2 points after filtering and have more than 50 points
    aggregated_3d_masks["ins"] = aggregated_3d_masks["ins"][
        (num_ins_points_after_filtering > small_mask_thres)
        & (
            num_ins_points_after_filtering
            > filtered_mask_thres * num_ins_points_before_filtering
        )
    ]
    # also delete the corresponding confidences and labels
    aggregated_3d_masks["conf"] = aggregated_3d_masks["conf"][
        (num_ins_points_after_filtering > small_mask_thres)
        & (
            num_ins_points_after_filtering
            > filtered_mask_thres * num_ins_points_before_filtering
        )
    ]
    aggregated_3d_masks["final_class"] = [
        aggregated_3d_masks["final_class"][i]
        for i in range(len(aggregated_3d_masks["final_class"]))
        if num_ins_points_after_filtering[i] > small_mask_thres
        and num_ins_points_after_filtering[i]
        > filtered_mask_thres * num_ins_points_before_filtering[i]
    ]

    print("after filtering", aggregated_3d_masks["ins"].shape)
    print("num_ins_points_after_filtering", aggregated_3d_masks["ins"].sum(dim=1))

    return aggregated_3d_masks


def occurance_filter(masked_counts: torch.Tensor, occurance_thres: float):
    """
    filter out masks that occur less than min_occurance

    args:
        masked_counts: counts of being detected as part of a mask for everypoint  (N)
        occurance_thres: points bottom `occurance_thres` percentage of occurance will be filtered out
    """

    occurance_counts = masked_counts.unique()
    print("occurance count", masked_counts.unique())

    occurance_thres_value = occurance_counts[
        round(occurance_thres * occurance_counts.shape[0])
    ]
    print("occurance thres value", occurance_thres_value)

    # remove all the points under median occurance
    masked_counts[masked_counts < occurance_thres_value] = 0

    point_mask = masked_counts > 0
    return point_mask


def detection_rate_filter(masked_counts: torch.Tensor, detection_rate_thres: float):
    """TODO: Not modified yet

    probabally not suitable for our case, cause it requires a full iteration of all the RGB images with visibility check
    and it takes extra 10-20 seconds in total

    """
    # image_dir = os.path.join(scene_2d_dir, scene_id, "color")
    # image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    # image_files.sort(
    #     key=lambda x: int(x.split(".")[0])
    # )  # sort numerically, 1.jpg, 2.jpg, 3.jpg ...
    # downsampled_image_files = image_files[::10]  # get one image every 10 frames
    # downsampled_images_paths = [
    #     os.path.join(image_dir, f) for f in downsampled_image_files
    # ]

    # viewed_counts = torch.zeros(scene_pcd.shape[1]).to(device=device)
    # for i, image_path in enumerate(
    #     tqdm(
    #         downsampled_images_paths,
    #         desc="Calculating viewed counts for every point",
    #         leave=False,
    #     )
    # ):
    #     frame_id = image_path.split("/")[-1][:-4]
    #     cam_pose = np.loadtxt(os.path.join(cam_pose_dir, f"{frame_id}.txt"))

    #     scene_pts = copy.deepcopy(scene_pcd)
    #     scene_pts = (np.linalg.inv(cam_pose) @ scene_pts).T[:, :3]  # (N, 3)
    #     projected_pts = compute_projected_pts_tensor(scene_pts, cam_intr)

    #     photo_width = int(cfg.width_2d)
    #     photo_height = int(cfg.height_2d)

    #     # frame_id_num = frame_id.split('.')[0]
    #     depth_im_path = os.path.join(depth_im_dir, f"{frame_id}.png")
    #     depth_im = (
    #         cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    #         / depth_scale
    #     )
    #     depth_im = cv2.resize(
    #         depth_im, (photo_width, photo_height)
    #     )  # Width x Height
    #     visibility_mask = compute_visibility_mask_tensor(
    #         scene_pts, projected_pts, depth_im, depth_thresh=0.08
    #     )
    #     viewed_counts += torch.tensor(visibility_mask).to(device=device)
    return None


if __name__ == "__main__":

    mask_2d_path = "aggregation/test_data/mask_2d/scene0435_00.pth"
    pcd_path = "aggregation/test_data/scannet200_3d/scene0435_00.npy"

    depth_dir = "aggregation/test_data/scannet200_2d/depth"
    cam_pose_dir = "aggregation/test_data/scannet200_2d/pose"

    # load all files
    masks_2d = torch.load(mask_2d_path)

    # load 3d point cloud and add 1 to the end for later transformation
    pcd_3d = np.load(pcd_path)[:, :3]
    pcd_3d = np.concatenate([pcd_3d, torch.ones([pcd_3d.shape[0], 1])], axis=1).T

    cam_intr = np.array(
        [[1170.1, 0.0, 647.7], [0.0, 1170.187988, 483.750000], [0.0, 0.0, 1.0]]
    )

    backprojected_3d_masks = {
        "ins": [],  # (Ins, N)
        "conf": [],  # (Ins, )
        "final_class": [],  # (Ins,)
    }

    """ 1. Project 2d masks to 3d point cloud"""
    for i in tqdm(range(len(masks_2d))):
        frame_id = masks_2d[i]["frame_id"][:-4]
        print("-------------------------frame", frame_id, "-------------------------")

        """Test data, replace with real images from bin files"""
        # load depth image
        depth_im = cv2.imread(
            os.path.join(depth_dir, frame_id + ".png"), cv2.IMREAD_ANYDEPTH
        )
        # load camera pose
        cam_pose = np.loadtxt(os.path.join(cam_pose_dir, f"{frame_id}.txt"))

        backprojected_3d_masks = project_2d_to_3d_single_frame(
            backprojected_3d_masks,
            cam_intr,
            depth_im,
            cam_pose,
            masks_2d[i],
            pcd_3d,
            1296,
            968,
        )

    # convert to tensor
    backprojected_3d_masks["ins"] = torch.stack(
        backprojected_3d_masks["ins"]
    )  # (Ins, N)
    backprojected_3d_masks["conf"] = torch.tensor(
        backprojected_3d_masks["conf"]
    )  # (Ins, )

    """ 2. Aggregating 3d masks"""
    # start aggregation
    aggregated_3d_masks, masked_counts = aggregate(backprojected_3d_masks)

    """ 3. Filtering 3d masks"""
    # start filtering
    filtered_3d_masks = filter(aggregated_3d_masks, masked_counts)

    print("print filtered_3d_masks", filtered_3d_masks)
    torch.save(filtered_3d_masks, "aggregation/test_data/test_3d_masks.pth")
