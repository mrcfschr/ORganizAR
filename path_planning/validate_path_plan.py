from path_planning import get_floor_grid, rotation_matrix_from_vectors
from path_planning import get_z_norm_of_plane, get_floor_mean
from path_planning import astar, get_object_grid_coordinates
from path_planning import get_starting_point
import open3d as o3d
import argparse
import torch
import numpy as np


from peaceful_pie.unity_comms import UnityComms

if __name__ == '__main__':
    """
    This script takes in the .ply file and the segmentation masks from the model and then visualizes the path planning in the Unity
    """
    device = torch.device("cpu")
    # Load the point cloud
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--point_cloud_path", type=str, help="Path to the point cloud file",
                        default="./data/downsampled_pcd.ply")
    parser.add_argument("-s","--seg_masks_path", type=str, help="Path to the segmentation masks file",
                        default="./data/filtered_3d_masks.pth")
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
    rot_matrix = rotation_matrix_from_vectors(z_norm,np.asarray([0.0,0.0,1.0]))
    #transform the point cloud
    transform_matrix = np.eye(4, dtype=float)
    transform_matrix[0:3, 0:3] = rot_matrix

    coords_for_pathplanning = np.asarray(pcd.points).copy()
    #transform the coords using the rotation matrix:
    coords_for_pathplanning = np.dot(coords_for_pathplanning, rot_matrix)
    #get the mean of the z coordinates of the inliers
    z_mean = get_floor_mean(np.asarray(coords_for_pathplanning)[inliers])
    #get the floor grid
    grid, bb, voxel_size, coor_to_grid, grid_to_coor = get_floor_grid(coords_for_pathplanning, z_mean, 100)
    path_plan_point_clouds = []
    #get the starting point and end point of the path planning
    for instance_index, instance in enumerate(seg_masks["ins"]):
        #shape of instance: (N,) mask
        instance_coords = coords_for_pathplanning[instance]
        #get the grid coordinates of the object
        instance_grid_coords = get_object_grid_coordinates(instance_coords, coor_to_grid)
        #get the starting point in the grid(non occulusion)
        start_point = get_starting_point(instance_grid_coords, grid)
        #get the end point in the grid
        grid_center = np.array([grid.shape[0]//2, grid.shape[1]//2])
        end_point = get_starting_point(np.asarray([grid_center]), grid)
        #get the path plan
        path_plan = astar(grid, start_point, end_point)
        #visualize the path plan o3d
        path_plan_coords = np.array([grid_to_coor(point[0],point[1]) for point in path_plan])
        #go back to world coordinates
        path_plan_coords = np.dot(path_plan_coords, rot_matrix.T)
        #visualize the path plan in open3d
        path_plan_pcd = o3d.geometry.PointCloud()
        path_plan_pcd.points = o3d.utility.Vector3dVector(path_plan_coords)
        path_plan_pcd.paint_uniform_color([0.0, 0.0, 1.0])
        path_plan_point_clouds.append(path_plan_pcd)
    path_plan_point_clouds.append(pcd)
    o3d.visualization.draw_geometries(path_plan_point_clouds)










