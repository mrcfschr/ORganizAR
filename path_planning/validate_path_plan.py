from path_planning import get_floor_grid, rotation_matrix_from_vectors
from path_planning import get_z_norm_of_plane, get_floor_mean
from path_planning import astar
import open3d as o3d
import argparse
import torch
import numpy as np

from peaceful_pie.unity_comms import UnityComms

if __name__ == '__main__':
    """
    This script takes in the .ply file and the segmentation masks from the model and then visualizes the path planning in the Unity
    """
    device = torch.device("mps")
    # Load the point cloud
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--point_cloud_path", type=str, help="Path to the point cloud file")
    parser.add_argument("-s","--seg_masks_path", type=str, help="Path to the segmentation masks file")
    args = parser.parse_args()
    pcd:o3d.geometry.PointCloud = o3d.io.read_point_cloud(args.point_cloud_path)
    # Load the segmentation masks from .pth file
    seg_masks: dict = torch.load(args.seg_masks_path, map_location=device)
    # Get the floor grid
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1,
                                                     ransac_n=3,
                                                     num_iterations=1000)
    #z_norm.shape = (3,)
    z_norm = get_z_norm_of_plane(plane_model[0], plane_model[1], plane_model[2], plane_model[3], np.asarray(pcd.points))
    #get the rotation matrix
    rot_matrix = rotation_matrix_from_vectors(z_norm, [0,0,1])
    #transform the point cloud
    transform_matrix = np.eye(4, dtype=float)
    transform_matrix[0:3, 0:3] = rot_matrix
    pcd = pcd.transform(transform_matrix)
    #get the mean of the z coordinates of the inliers
    z_mean = get_floor_mean(np.asarray(pcd.points)[inliers])
    #get the floor grid
    grid, bb, voxel_size, coor_to_grid, grid_to_coor = get_floor_grid(np.asarray(pcd.points), z_mean, 1000)
    #get the path_plan
    path_plan = astar(grid, (0,0), (100,100))






