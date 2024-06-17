#------------------------------------------------------------------------------
# RGBD integration using Open3D. Color information comes from the front RGB
# camera.
# Press space to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import numpy as np
import multiprocessing as mp
import open3d as o3d
import cv2
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_io
import view_manager
from typing import List, Tuple, Any, Dict,Callable
from lang_sam import LangSAM
import PIL
from PIL import Image
import torch

# Settings --------------------------------------------------------------------
from_recording = True #set to run live on HL vs from recorded dataset
visualization_enabled = False
write_data = False
wsl = False
remote_docker = True

if wsl:
    visualization_enabled = False
    path_start = "/mnt/c/Users/Marc/Desktop/CS/MARPROJECT/"
elif remote_docker:
    path_start = "/medar_smart/ORganizAR/"
else:
    path_start = "C:/Users/Marc/Desktop/CS/MARPROJECT/"

# HoloLens address
host = '192.168.0.102'

# Directory containing the recorded data
path = path_start + 'viewer/data'

# Directory containing the calibration data of the HoloLens
calibration_path: str = path_start + 'calibration/rm_depth_longthrow/'

#rgb images from recorded data
write_data_path = "viewer/data/debug/"

# Camera parameters
pv_width = 640
pv_height = 360
pv_framerate = 30
#settings from sample, adjust for optimal detection
# pv_exposure_mode = hl2ss.PV_ExposureMode.Manual
# pv_exposure = hl2ss.PV_ExposureValue.Max // 4
# pv_iso_speed_mode = hl2ss.PV_IsoSpeedMode.Manual
# pv_iso_speed_value = 1600
# pv_white_balance = hl2ss.PV_ColorTemperaturePreset.Manual

# Buffer length in seconds
buffer_length = 10

# Integration parameters
voxel_length = 1/100
sdf_trunc = 0.04
max_depth = 7 #3.0 in sample, changed to room size

prompts =  ["bed" , "pen", "timer", "chair", "backpack", "black computer monitor", "door", "coffee machine", "lamp"]
#------------------------------------------------------------------------------
def get_segmented_points(data_pv: Any,
                         data_depth: Any,
                         lt_scale: np.ndarray,
                         max_depth: int,
                         xy1: np.ndarray,
                         calibration_lt: Any,
                         prompt: str) -> Tuple[bool, np.ndarray]:
    """
    Retrieves segmented 3D points from images using lang-segment-anything. https://github.com/luca-medeiros/lang-segment-anything

    Args:
    - data_pv: PV data containing the rgb image for segmentation and extrinsics and intrinsics for projecting the points on the computed mask.
    - data_depth: Depth data to get point cloud.
    - lt_scale: Scale for depth normalization.
    - max_depth: Maximum depth threshold. Points that are farther away are ignored.
    - xy1: Rays to project depth image to points.
    - calibration_lt: Depth sensor Calibration data e.g extrinsics to go from depth sensor space to rignode space (relative to HoloLens device origin) using the extrinsics inverse.

    Returns:
    - numpy array of segmented points in the HoloLens world space.
    """
    # Update PV intrinsics ------------------------------------------------
    # PV intrinsics may change between frames due to autofocus
    pv_intrinsics = hl2ss.create_pv_intrinsics(data_pv.payload.focal_length, data_pv.payload.principal_point)
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)
    pv_intrinsics, pv_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

    # Preprocess frames --------------------------------------------------- TODO ealier
    depth = hl2ss_3dcv.rm_depth_normalize(data_depth.payload.depth, lt_scale)
    depth[depth > max_depth] = 0
    image_pil = Image.fromarray(cv2.cvtColor(data_pv.payload.image, cv2.COLOR_BGR2RGB))
    
    masks, boxes, phrases, logits = model_LangSAM.predict(image_pil, prompt)
    if masks == None or len(masks) == 0:
        print("nothing found")
        return False, np.array([])

    confidence_threshold_cutoff = 0.3
  
    
    combined_mask = np.zeros(masks[0].shape, dtype=np.uint8)
    high_conf = 0
    #TODO median max conf threshold
    for mask, logit in zip(masks, logits):
        if logit >= confidence_threshold_cutoff:
            high_conf = logit
            mask_np = (mask.cpu().detach().numpy() * 255).astype(np.uint8)
            combined_mask = np.maximum(combined_mask, mask_np)

    mask_3ch = cv2.merge([combined_mask, combined_mask, combined_mask])
    image_rgb = cv2.cvtColor(data_pv.payload.image, cv2.COLOR_BGR2RGB)
    alpha = 0.5
    overlay = cv2.addWeighted(image_rgb, 1, mask_3ch, alpha, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    if high_conf > 0:
        cv2.imwrite(write_data_path+"mask"+prompt+str(high_conf)+".png", overlay_bgr)

    points = hl2ss_3dcv.rm_depth_to_points(depth, xy1)
    depth_to_world = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_depth.pose)
    points = hl2ss_3dcv.transform(points, depth_to_world)

    # Project pointcloud image --------------------------------------------
    world_to_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)
    pixels = hl2ss_3dcv.project(points, world_to_image)
    
    map_u = pixels[:, :, 0]
    map_v = pixels[:, :, 1]

        # Get 3D points labels -------------------------------------
    labels = cv2.remap(combined_mask, map_u, map_v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) #pixels outside to 0
    points_mask = labels == 255
    return True, points[points_mask]


if __name__ == '__main__':
    # Keyboard events ---------------------------------------------------------
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.space
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()


    if from_recording:
        # Create readers --------------------------------------------------------------
        rd_lt = hl2ss_io.create_rd(f'{path}/{hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)}.bin', hl2ss.ChunkSize.SINGLE_TRANSFER, True)
        rd_pv = hl2ss_io.sequencer(f'{path}/{hl2ss.get_port_name(hl2ss.StreamPort.PERSONAL_VIDEO)}.bin', hl2ss.ChunkSize.SINGLE_TRANSFER, 'bgr24')

        # Open readers ----------------------------------------------------------------
        rd_lt.open()
        rd_pv.open()

        calibration_lt = hl2ss_3dcv._load_calibration_rm_depth_longthrow(calibration_path)
        uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
        xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
    else:
        # Start PV Subsystem ------------------------------------------------------
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

        # Wait for PV subsystem and fix exposure, iso speed, and white balance ----
        # settings from sample, adjust for optimal detection
        # ipc_rc = hl2ss_lnm.ipc_rc(host, hl2ss.IPCPort.REMOTE_CONFIGURATION)
        # ipc_rc.open()
        # ipc_rc.wait_for_pv_subsystem(True)
        # ipc_rc.set_pv_exposure(pv_exposure_mode, pv_exposure)
        # ipc_rc.set_pv_iso_speed(pv_iso_speed_mode, pv_iso_speed_value)
        # ipc_rc.set_pv_white_balance_preset(pv_white_balance)
        # ipc_rc.close()

        # Get RM Depth Long Throw calibration -------------------------------------
        # Calibration data will be downloaded if it's not in the calibration folder
        calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)

        uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
        xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
         # Start PV and RM Depth Long Throw streams --------------------------------
        producer = hl2ss_mp.producer()
        producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate, decoded_format='rgb24'))
        producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
        producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_framerate * buffer_length)
        producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS * buffer_length)
        producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
        producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

        consumer = hl2ss_mp.consumer()
        manager = mp.Manager()
        sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
        sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, ...)

        sink_pv.get_attach_response()
        sink_depth.get_attach_response()

    # Create Open3D integrator and visualizer ---------------------------------
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=sdf_trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    intrinsics_depth = o3d.camera.PinholeCameraIntrinsic(hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT, calibration_lt.intrinsics[0, 0], calibration_lt.intrinsics[1, 1], calibration_lt.intrinsics[2, 0], calibration_lt.intrinsics[2, 1])

    if visualization_enabled:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        first_pcd = True

    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)
 
    view_mana = view_manager.ViewManager()


    #initialize inferences
    #localization and segmentation 
    model_LangSAM: LangSAM = LangSAM()

    counter_all  = 0
    counter_selected = 0
    # Main Loop ---------------------------------------------------------------
    while (enable):
        
        if from_recording:
            # Get LT Depth frame ------------------------------------------------------------
            data_lt = rd_lt.get_next_packet() 
            if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
                continue
            # Find PV corresponding to the current Depth frame ----------------
            data_pv = rd_pv.get_next_packet(data_lt.timestamp) # Get nearest (in time) pv frame
            if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
                continue
        else:    
            # Wait for RM Depth Long Throw frame ----------------------------------
            sink_depth.acquire()

            # Get RM Depth Long Throw frame and nearest (in time) PV frame --------
            _, data_lt = sink_depth.get_most_recent_frame()
            if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
                continue

            _, data_pv = sink_pv.get_nearest(data_lt.timestamp)
            if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
                continue

        # Preprocess frames ---------------------------------------------------
        depth = hl2ss_3dcv.rm_depth_undistort(data_lt.payload.depth, calibration_lt.undistort_map)
        depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)
        color = data_pv.payload.image
        (image_isNovel_view,index,img) = view_mana.new_view(color)
        # image_isNovel_view = True
        counter_all +=1
        if image_isNovel_view:
            counter_selected +=1
            if write_data:
                cv2.imwrite(write_data_path + str(data_pv.timestamp) +".png", img)
            # async
            # get boxes of hardcoded objects we want to detect e.g. c arm
            # get segmentation mask of boxes with highest confidence 
            # for proof of concept just use lang sam
            for prompt in prompts:
                ret, points = get_segmented_points(data_pv, data_lt, scale, max_depth, xy1, calibration_lt, prompt)
            # render integrated point cloud one detected object at a time

            # Update PV intrinsics ------------------------------------------------
            # PV intrinsics may change between frames due to autofocus
            pv_intrinsics = hl2ss.update_pv_intrinsics(pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
            color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)
            
            # Generate aligned RGBD image -----------------------------------------
            lt_points         = hl2ss_3dcv.rm_depth_to_points(xy1, depth)
            lt_to_world       = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
            world_to_lt       = hl2ss_3dcv.world_to_reference(data_lt.pose) @ hl2ss_3dcv.rignode_to_camera(calibration_lt.extrinsics)
            world_to_pv_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)
            world_points      = hl2ss_3dcv.transform(lt_points, lt_to_world)
            pv_uv             = hl2ss_3dcv.project(world_points, world_to_pv_image)
            color             = cv2.remap(color, pv_uv[:, :, 0], pv_uv[:, :, 1], cv2.INTER_LINEAR)
            if write_data:
                cv2.imwrite(write_data_path + "_depth_aligned_"+str(data_pv.timestamp) +".png", color)
            mask_uv = hl2ss_3dcv.slice_to_block((pv_uv[:, :, 0] < 0) | (pv_uv[:, :, 0] >= pv_width) | (pv_uv[:, :, 1] < 0) | (pv_uv[:, :, 1] >= pv_height))
            depth[mask_uv] = 0

            # Convert to Open3D RGBD image ----------------------------------------
            color_image = o3d.geometry.Image(color)
            depth_image = o3d.geometry.Image(depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, depth_scale=1, depth_trunc=max_depth, convert_rgb_to_intensity=False)

            # Integrate RGBD and display point cloud ------------------------------
            volume.integrate(rgbd, intrinsics_depth, world_to_lt.transpose())
            pcd_tmp = volume.extract_point_cloud()
            
            if visualization_enabled:
                if (first_pcd):
                    first_pcd = False
                    pcd = pcd_tmp
                    vis.add_geometry(pcd)
                else:
                    pcd.points = pcd_tmp.points
                    pcd.colors = pcd_tmp.colors
                    vis.update_geometry(pcd)
            
                vis.poll_events()
                vis.update_renderer()
            print(counter_selected)         #23 286
            print(counter_all)            
            print(counter_selected/counter_all)
  
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
    listener.join()
    if visualization_enabled:
        # Show final point cloud --------------------------------------------------
        vis.run()
