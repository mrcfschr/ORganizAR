import time
import numpy as np
import cv2
# from pynput import keyboard
import multiprocessing as mp
import open3d as o3d
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_io
import hl2ss_rus
import view_manager
from typing import List, Tuple, Any, Dict, Callable
from lang_sam import LangSAM
import PIL
from PIL import Image
import torch
import clip
import os
import open3d.core as o3c
import pdb
from tqdm import tqdm
import sys
import pandas as pd

from faster_haoyang import PromptBoxManager
from utils import project_2d_to_3d_single_frame, aggregate, filter

# Settings --------------------------------------------------------------------
log_file_path = "./log.txt"
#sys.stdout = open(log_file_path,"w",buffering=1,encoding='utf-8')

from_recording = True # set to run live on HL vs from recorded dataset
visualization_enabled = True
write_data = True
wsl = False
remote_docker = False
Alienware = True
reconstruct_point_cloud = True
visualize_reconstruction = False



if wsl:
    visualization_enabled = False
    path_start = "/mnt/c/Users/Marc/Desktop/CS/MARPROJECT/"
elif remote_docker:
    path_start = "/medar_smart/"
elif Alienware:
    path_start = "/data/projects/medar_smart/ORganizAR/"
    print("start path: ", path_start)
else:
    path_start = "/Users/haoyangsun/Documents/ORganizAR/"
    print("start path: ", path_start)

# HoloLens address
host = '192.168.195.253'

# Directory containing the recorded data
path = path_start + 'viewer/recorded_data/dataset_test_3'

df = pd.read_csv(path+"/rm_depth_longthrow.csv")
stop_num_frames = 60
sample_frequency = df.shape[0] // stop_num_frames#dataset1: 5   dataset2: 2
# Directory containing the calibration data of the HoloLens
calibration_path: str = path_start + 'calibration/rm_depth_longthrow/'

point_cloud_path: str = path + '/point_cloud_recon.pcd'
mask_path: str = path + 'seg_mask.pth'

# rgb images from recorded data
write_data_path = "viewer/data/debug_faster42/"
if os.path.exists(path_start + write_data_path) == False:
    os.makedirs(path_start + write_data_path)

# Camera parameters
pv_width = 640
pv_height = 360
pv_framerate = 30

# Buffer length in seconds
buffer_length = 10

# Integration parameters
voxel_length = 1 / 100
sdf_trunc = 0.04
max_depth = 7  # 3.0 in sample, changed to room size






# Semantic search setup
device_search = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
o3d_device = o3d.core.Device("CUDA:0" if torch.cuda.is_available() else "CPU:0")
print("using device: ", o3d_device, " for reconstruction")

# Test data setup
image_pil_timer = Image.open(path_start + "viewer/test_timer.png")
image_pil_bed = Image.open(path_start + "/viewer/test_bed.png")
# prompts_lookup = [ 
#     "table with a light-blue cloth",
#     "medical equipment cart, has multiple shelves, a handle on top, and is grey with some blue sections",
#     "c arm machine, has its distinctive c shaped arm which houses an X-ray source and detector and connected control unit with dials and buttons",
#     # "backpack",
#     "The ultrasound machine tower with a large monitor on top" ,#  a console with buttons and knobs, it is white with grey accents",
#     "chair",
#     # "monitor"
#     ]
# prompts_lookup = [ 
#     "table with a light-blue cloth",
#     "medical equipment cart, has multiple shelves, a handle on top, and is grey with some blue sections",
#     "c arm machine, has its distinctive c shaped arm and it is connected control unit with dials and buttons",
#     "backpack",
#     "The ultrasound machine tower has a large monitor on top, a console with buttons and knobs, it is white with grey accents",
#     "chair",
#     "monitor"
#     ]

prompts_lookup = ["bed with a dummy on it","a grey shelf with a pole on it",
                  "c arm machine, which is a medical machine with a large c shaped metal arm",
                  "backpack"," ultra sound machine with a monitor on it","chair","monitor"]
# prompts_lookup = ["table with a light-blue cloth","a grey shelf with a pole on it","c arm machine, which is a medical machine with a large c shaped metal arm and its connected control unit with dials and buttons","backpack"," ultra sound machine, a console with buttons and knobs, it is white with grey accents","chair"]

#prompts_lookup = ["table with a light-blue cloth","bed","a grey shelf with a pole on it","c arm, which is a medical machine with a large c shaped metal arm","backpack"," ultra sound machine with a monitor on it","chair","monitor"]
#prompts_lookup = ["A table with cloth on it","bed","a grey shelf with a pole on it","c arm, which is a medical machine with a large c shaped metal arm","backpack"," ultra sound machine, that has flashlight shape probe attached and a machine tower","chair","computer monitor"]
prompts = [". ".join(prompts_lookup)]


# prompts = ["C-Arm, which has a large C-shaped arm and machine tower"]
# prompts = ["C Arm", "Ultrasound Machine", "Laparoscopic Tower", "Chairs", "Table with cloth", "Patient Bed"]#["timer","bed","monitors", "orange pen", "keyboard"]


boxManager = PromptBoxManager(prompts, prompts_lookup)

images = [image_pil_timer, image_pil_bed]

data = {}
# CLIP_SIM_THRESHOLD = 0.25
# DINO_THRESHOLD = 0.3
# MIN_FRAME_NUM = 5
enable = True

# unity pc vis secctiond
# quad scale in meters
quad_scale: List[float] = [0.005, 0.005, 0.005]
sphere_scale: List[float] = [0.001, 0.001, 0.001]
total_quad_count: int = 500


if reconstruct_point_cloud:
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=sdf_trunc,
                                                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    # vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight'),
    #                                         attr_dtypes=(o3c.float32,
    #                                                      o3c.float32),
    #                                         attr_channels=((1), (1)),
    #                                         voxel_size=3.0 / 512,
    #                                         block_resolution=16,
    #                                         block_count=50000,
    #                                         device=o3d_device)
    print("created volume")
    #initialize visualizer
    print("initializing visualizer")
    if visualize_reconstruction:
        recon_vis = o3d.visualization.Visualizer()
        recon_vis.create_window()
        print("initialized visualizer")

def apply_clip_embedding_prompt(prompt):
    text = clip.tokenize([prompt]).to(device_search)
    return text


def set_up_data_struct(prompts):
    data = {}
    for prompt in prompts:
        tokenized_prompt = apply_clip_embedding_prompt(prompt)  # Tokenizing the prompt for CLIP
        data[prompt] = {
            'tokenized_prompt': tokenized_prompt,
            'frames': {}  # This will store information per frame, to be populated later
        }
    return data

def display_point_cloud(points: np.ndarray, prompt_index: int) -> np.ndarray:
    """
    Displays a subsampled point cloud in a Unity scene as small red quads.

    Args:
    - points: Array containing the points of the point cloud.

    Returns:
    - Array containing the results from the server after pushing commands.
    """
    # Add quad to Unity scene
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list()  # Begin sequence
    # Check if subsampling is needed
    num_of_quads_per_object = total_quad_count // len(data.keys())
    offset = prompt_index * num_of_quads_per_object
    if len(points) >= num_of_quads_per_object:
        points = points[np.random.choice(len(points), num_of_quads_per_object, replace=False)]

    for index, point in enumerate(points):
        point[2] = -point[2]  # unity is lefthanded
        display_list.set_target_mode(0)  # default target mode by key
        display_list.set_world_transform(keys[index + offset], point, [0, 0, 0, 1],
                                         sphere_scale)  # Set the quad's world transform
        display_list.set_active(keys[index + offset], 1)  # Make the quad visible

    display_list.end_display_list()  # End sequence
    ipc.push(display_list)  # Send commands to server
    return ipc.pull(display_list)  # Get results from server


def display_centroid(points: np.ndarray, prompt_index: int) -> np.ndarray:
    print("DEBUG display_centroid", points.shape, type(points))
    # Add quad to Unity scene
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list()  # Begin sequence
    x_median = np.median(points[:, 0])
    y_median = np.median(points[:, 1])
    z_median = np.median(points[:, 2])

    point = [x_median, y_median, z_median]
    print(f"median{point}")
    point[2] = -point[2]  # unity is lefthanded
    display_list.create_primitive(hl2ss_rus.PrimitiveType.Sphere)  # TODO store key for later manipulation
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast)
    display_list.set_world_transform(prompt_index, point, [0, 0, 0, 1], sphere_scale)  # Set the quad's world transform
    display_list.set_active(prompt_index, 1)  # Make the quad visible
    display_list.end_display_list()  # End sequence
    ipc.push(display_list)  # Send commands to server

    return ipc.pull(display_list)  # Get results from server

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable


def get_segmented_points(box_data, lt_scale, max_depth, xy1, calibration_lt):
    data_depth = box_data["data_depth"]
    data_pv = box_data["data_pv"]
    mask = box_data["combined_mask"]

    # Update PV intrinsics ------------------------------------------------
    # PV intrinsics may change between frames due to autofocus
    pv_intrinsics = hl2ss.create_pv_intrinsics(data_pv.payload.focal_length, data_pv.payload.principal_point)
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)
    pv_intrinsics, pv_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

    # Preprocess frames --------------------------------------------------- TODO ealier
    depth = hl2ss_3dcv.rm_depth_normalize(data_depth.payload.depth, lt_scale)
    depth[depth > max_depth] = 0

    points = hl2ss_3dcv.rm_depth_to_points(depth, xy1)
    depth_to_world = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(
        data_depth.pose)
    points = hl2ss_3dcv.transform(points, depth_to_world)

    # Project pointcloud image --------------------------------------------
    world_to_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(
        pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)
    pixels = hl2ss_3dcv.project(points, world_to_image)

    map_u = pixels[:, :, 0]
    map_v = pixels[:, :, 1]

    # Get 3D points labels -------------------------------------
    labels = cv2.remap(mask, map_u, map_v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0)  # pixels outside to 0
    points_mask = labels == 255
    return points[points_mask]

if __name__ == '__main__':
    # Keyboard events ---------------------------------------------------------
    enable = True
    # listener = keyboard.Listener(on_press=on_press)
    # listener.start()

    initial_transform = np.empty(0)

    if from_recording:
        # Create readers --------------------------------------------------------------
        rd_lt = hl2ss_io.create_rd(f'{path}/{hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)}.bin',
                                   hl2ss.ChunkSize.SINGLE_TRANSFER, True)
        rd_pv = hl2ss_io.sequencer(f'{path}/{hl2ss.get_port_name(hl2ss.StreamPort.PERSONAL_VIDEO)}.bin',
                                   hl2ss.ChunkSize.SINGLE_TRANSFER, 'bgr24')

        # Open readers ----------------------------------------------------------------
        rd_lt.open()
        rd_pv.open()

        calibration_lt = hl2ss_3dcv._load_calibration_rm_depth_longthrow(calibration_path)
        uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH,
                                         hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
        xy1, depth_scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
    else:
        # Start PV Subsystem ------------------------------------------------------
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

        # Get RM Depth Long Throw calibration -------------------------------------
        # Calibration data will be downloaded if it's not in the calibration folder
        calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)

        uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH,
                                         hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
        xy1, depth_scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
        # Start PV and RM Depth Long Throw streams --------------------------------
        producer = hl2ss_mp.producer()
        producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO,
                           hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height,
                                           framerate=pv_framerate, decoded_format='rgb24'))
        producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
                           hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
        producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_framerate * buffer_length)
        producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
                            hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS * buffer_length)
        producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
        producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

        consumer = hl2ss_mp.consumer()
        manager = mp.Manager()
        sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
        sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, ...)

        sink_pv.get_attach_response()
        sink_depth.get_attach_response()

        # init unity comm
        ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
        ipc.open()
        key = 0
        command_buffer = hl2ss_rus.command_buffer()
        command_buffer.remove_all()
        ipc.push(command_buffer)
        results = ipc.pull(command_buffer)
        # keys = instantiate_gos(ipc)

    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    view_mana = view_manager.ViewManager()

    data = set_up_data_struct(prompts)

    intrinsics_depth = o3d.camera.PinholeCameraIntrinsic(hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH,
                                                         hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT,
                                                         calibration_lt.intrinsics[0, 0],
                                                         calibration_lt.intrinsics[1, 1],
                                                         calibration_lt.intrinsics[2, 0],
                                                         calibration_lt.intrinsics[2, 1])
    

    # Main Loop ---------------------------------------------------------------
    counter = 0
    print("start detecting")
    while (enable):

        if from_recording:
            # Get LT Depth frame ------------------------------------------------------------
            data_depth = rd_lt.get_next_packet()
            if ((data_depth is None) or (not hl2ss.is_valid_pose(data_depth.pose))):
                continue
            # Find PV corresponding to the current Depth frame ----------------
            data_pv = rd_pv.get_next_packet(data_depth.timestamp)  # Get nearest (in time) pv frame
            if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
                continue
        else:
            # Wait for RM Depth Long Throw frame ----------------------------------
            sink_depth.acquire()

            # Get RM Depth Long Throw frame and nearest (in time) PV frame --------
            _, data_depth = sink_depth.get_most_recent_frame()
            if ((data_depth is None) or (not hl2ss.is_valid_pose(data_depth.pose))):
                continue

            _, data_pv = sink_pv.get_nearest(data_depth.timestamp)
            if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
                continue



        # Preprocess frames ---------------------------------------------------
        depth = hl2ss_3dcv.rm_depth_undistort(data_depth.payload.depth, calibration_lt.undistort_map)
        depth = hl2ss_3dcv.rm_depth_normalize(depth, depth_scale)
        color_np = data_pv.payload.image
        #change to pil rbg data
        color_np = color_np[:,:,[2,1,0]]
        color_pil = Image.fromarray(color_np)

        # get depth and color alignment
        pv_intrinsics = hl2ss.update_pv_intrinsics(pv_intrinsics, data_pv.payload.focal_length,
                                                   data_pv.payload.principal_point)
        color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

        lt_points = hl2ss_3dcv.rm_depth_to_points(xy1, depth)
        lt_to_world = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(
            data_depth.pose)
        world_to_lt = hl2ss_3dcv.world_to_reference(data_depth.pose) @ hl2ss_3dcv.rignode_to_camera(
            calibration_lt.extrinsics)

        world_to_color = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(
            color_extrinsics)

        world_to_pv_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(
            color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)
        world_points = hl2ss_3dcv.transform(lt_points, lt_to_world)
        pv_uv = hl2ss_3dcv.project(world_points, world_to_pv_image)
        
        
        
        color = cv2.remap(color_np, pv_uv[:, :, 0], pv_uv[:, :, 1], cv2.INTER_LINEAR)
        mask_uv = hl2ss_3dcv.slice_to_block(
            (pv_uv[:, :, 0] < 0) | (pv_uv[:, :, 0] >= pv_width) | (pv_uv[:, :, 1] < 0) | (pv_uv[:, :, 1] >= pv_height))
        depth_recon = np.copy(depth)
        depth_recon[mask_uv] = 0
        
        #pdb.set_trace()
        #(image_isNovel_view, index, img) = view_mana.new_view(color_np)
        if counter%sample_frequency ==0:
            image_isNovel_view = True
            index = counter // sample_frequency
            img = color_np
            #print(counter)
        else:
            image_isNovel_view = False
            index = -1
            img = None
            #print(counter)

        if image_isNovel_view:
            pv_timestamp = data_pv.timestamp
            depth_timestamp = data_depth.timestamp
            print("saving frame ", str(index))
            #color_pil.save(path_start + write_data_path + "selected_frame" + str(index) + ".jpeg")
            boxManager.new_frame(color_pil, pv_timestamp, depth_timestamp, np.copy(depth),color_intrinsics,world_to_color.transpose(),world_to_lt.transpose())
            # boxManager.output_det()

            #if counter == 55:
            #    boxManager.save_mask_as_pth(mask_path)
            #    print("saved segment mask")
            if reconstruct_point_cloud:
                #print("new_recon_start: ", "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)) 
        
                color_image = o3d.geometry.Image(color)
                depth_image = o3d.geometry.Image(depth_recon)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, depth_scale=1,
                                                                       depth_trunc=max_depth,
                                                                       convert_rgb_to_intensity=False)
                volume.integrate(rgbd, intrinsics_depth, world_to_lt.transpose())
                #print("new_recon_end: ", "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)) 
        
                
                
                #depth = o3d.t.geometry.Image(depth_recon).to(o3d_device)
                #extrinsic = o3d.core.Tensor.from_numpy(world_to_lt.transpose())
                #depth_intrinsic = color_intrinsics[0:3,0:3].transpose()

                #pdb.set_trace()
                # frustum_block_coords = vbg.compute_unique_block_coordinates(
                #     depth, depth_intrinsic, extrinsic, 1000.0,
                #     3.0)
                
                pcd_tmp = volume.extract_point_cloud()
                #pcd_tmp = vbg.extract_point_cloud()

                if counter == 0:
                  pcd = pcd_tmp
                  if visualize_reconstruction:
                    recon_vis.add_geometry(pcd)
                else:
                    if visualize_reconstruction:
                        print("updating the point cloud")
                        pcd.points = pcd_tmp.points
                        pcd.colors = pcd_tmp.colors
                        recon_vis.update_geometry(pcd)
                    #recon_vis.update_geometry(pcd)
                if index == stop_num_frames:
                    #restrict the number of masks in the desired class
                    restricted_class= 4


                    restricted_number =8
                    restricted_frame_stride=5
                    #boxManager.restrict_number(restricted_class,restricted_number,restricted_frame_stride)
                    #print("restricted class: ",restricted_class, ", restricted number: ", restricted_number, ", frame_stride: ", restricted_frame_stride)

                    mask_2d = boxManager.frame_masks
                    for m in mask_2d:
                        m["segmented_frame_masks"] = m["masks"]
                        m["labels"] = m["classes"]
                        m["confidences"] = torch.ones(len(m["masks"]))
                    pcd_down = pcd_tmp.voxel_down_sample(voxel_size=0.02)
                    o3d.io.write_point_cloud(path+"/downsampled_pcd.ply", pcd_down)
                    pcd_3d = np.asarray(pcd_down.points)
                    
                    pcd_3d = np.concatenate([pcd_3d, torch.ones([pcd_3d.shape[0], 1])], axis=1).T # shape (3, N)

                    cam_intr = mask_2d[0]["color_intrinsics"].T[:3, :3]
                    
                    """ 1. Project 2d masks to 3d point cloud"""
                    backprojected_3d_masks = {
                        "ins": [],  # (Ins, N)
                        "conf": [],  # (Ins, )
                        "final_class": [],  # (Ins,)
                    }
                    for i in tqdm(range(len(mask_2d))):
                        print("-------------------------frame", i, "-------------------------")
                        """Test data, rehplace with real images from bin files"""
                        # load depth image
                        depth_im = mask_2d[i]["depth_image"]
                        # load camera pose
                        # mask_2d[i]["color_extrinsics"] is world to camera pose, our function requires camera to world pose
                        cam_pose = np.linalg.inv(mask_2d[i]["color_extrinsics"])  #
                        depth_pose = np.linalg.inv(mask_2d[i]["depth_extrinsics"])

                        backprojected_3d_masks = project_2d_to_3d_single_frame(
                            backprojected_3d_masks,
                            cam_intr,
                            depth_im,
                            cam_pose,
                            depth_pose,
                            mask_2d[i],
                            pcd_3d,
                            760,
                            428,
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
                            depth_thresh=0.1,
                            depth_scale=1,
                        )
                    
                    print("sucsessfully saved point cloud")
                    pcd.points=pcd_down.points
                    pcd.colors = pcd_down.colors
                    # convert to tensor
                    backprojected_3d_masks["ins"] = torch.stack(backprojected_3d_masks["ins"])  # (Ins, N)
                    backprojected_3d_masks["conf"] = torch.tensor(backprojected_3d_masks["conf"])  # (Ins, )
                    torch.save(backprojected_3d_masks,path+"/backprojected_3d_masks.pth")
                    """ 2. Aggregating 3d masks"""
                    # start aggregation
                    aggregated_3d_masks, mask_indeces_to_be_merged = aggregate(backprojected_3d_masks, iou_threshold=0.1)
                    torch.save(aggregated_3d_masks, path+"/aggregated_3d_masks.pth")

                    """ 3. Filtering 3d masks"""
                    # start filtering
                    filtered_3d_masks = filter(aggregated_3d_masks, mask_indeces_to_be_merged, backprojected_3d_masks, if_occurance_threshold=True,occurance_thres= 0.2, small_mask_thres=200, filtered_mask_thres=0.1)
                    # boxManager.save_mask_as_pth(path=path+"/segmentation_masks.pth")
                    

                    # """ 2. Aggregating 3d masks"""
                    # # start aggregation
                    # aggregated_3d_masks, masked_counts = aggregate(backprojected_3d_masks, iou_threshold=0.25)
                    # torch.save(aggregated_3d_masks, path+"/aggregated_3d_masks.pth")
                    # """ 3. Filtering 3d masks"""
                    # #start filtering
                    # filtered_3d_masks = filter(aggregated_3d_masks, masked_counts, occurance_thres= 0.10, small_mask_thres=200, filtered_mask_thres=0.2)
                    # boxManager.save_mask_as_pth(path=path+"/segmentation_masks.pth")
                    

                    """
                    {
                        "ins": torch.Tensor,  # (Ins, N)
                        "conf": torch.Tensor,  # (Ins, )
                        "final_class": list of torch.Tensor# (Ins,)
                    }
                    """
                    torch.save(filtered_3d_masks, path+"/filtered_3d_masks.pth")

                    pcd_3d = pcd_3d.T # shape (N,3)

                    
                    for points_filtered_mask, class_filtered in zip(filtered_3d_masks["ins"], filtered_3d_masks["final_class"]): # points_filtered_mask shape (N)
                        points_filtered = pcd_3d[points_filtered_mask.cpu().numpy()]
                        print("sent centroid to HL2: ", prompts_lookup[class_filtered.cpu().numpy()])
                        display_centroid(points_filtered,int(class_filtered.cpu().numpy()))
                    if not from_recording:
                        #display centroid to unity
                        None
                    
                    print("finished")
                    if visualize_reconstruction:
                        recon_vis.destroy_window()


                if visualize_reconstruction:
                    recon_vis.poll_events()
                    recon_vis.update_renderer()

        counter += 1
    # shutdown server
    if from_recording:
        rd_pv.close()
        rd_lt.close()
    else:
        # Stop PV and RM Depth Long Throw streams ---------------------------------
        sink_pv.detach()
        sink_depth.detach()
        producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
        producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

        # Stop PV subsystem -------------------------------------------------------
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
    # Stop keyboard events ----------------------------------------------------
    # listener.join()



