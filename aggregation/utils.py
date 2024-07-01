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
    masks_2d,
    pcd_3d,
    photo_width,
    photo_height,
    depth_thresh=0.08,
    depth_scale=1000,
):
    """project 2d masks to 3d point cloud

    args:
        backprojected_3d_masks: add results to this dict
        cam_intr: 3x3 camera intrinsic matrix
        depth_im: depth image of current frame readed by cv2.imread
        cam_pose: camera pose of current frame
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

    # process depth image
    depth_im = cv2.resize(depth_im, (photo_width, photo_height))
    depth_im = depth_im.astype(np.float32) / depth_scale

    # convert 3d points to camera coordinates
    scene_pts = (np.linalg.inv(cam_pose) @ scene_pts).T[:, :3]  # (N, 3)

    """Start projectiom"""
    # project 3d points to current 2d frame
    projected_pts = compute_projected_pts_tensor(scene_pts, cam_intr)

    # check visibility of projected 2d points
    visibility_mask = compute_visibility_mask_tensor(
        scene_pts, projected_pts, depth_im, depth_thresh=depth_thresh
    )

    # Select visible 3D points in 2D masks as backprojected 3D masks
    masked_pts = compute_visible_masked_pts_tensor(
        scene_pts, projected_pts, visibility_mask, pred_masks
    )

    masked_pts = torch.from_numpy(masked_pts).to(device)  # (M, N)
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
    final_masks = solve_overlapping(aggregated_masks, mask_indeces_to_be_merged)

    return {
        "ins": final_masks,  # torch.tensor (Ins, N)
        "conf": aggregated_confidences,  # torch.tensor (Ins, )
        "final_class": aggregated_labels,  # List[str] (Ins,)
    }


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
) -> torch.Tensor:
    """
    solve overlapping among all masks
    """
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
        if num_masks[i] > num_masks[j]:
            aggregated_masks[j] &= ~aggregated_masks[i]
        else:
            aggregated_masks[i] &= ~aggregated_masks[j]

    return aggregated_masks


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

    # Calculate masked counts for filtering
    masked_counts = backprojected_3d_masks["ins"].sum(dim=0)  # (N,)

    """ 2. Aggregating 3d masks"""
    # start aggregation
    aggregated_3d_masks = aggregate(backprojected_3d_masks)

    """ 3. Filtering 3d masks"""
    # start filtering
    filtered_3d_masks = filter(aggregated_3d_masks, masked_counts)

    print("print filtered_3d_masks", filtered_3d_masks)
