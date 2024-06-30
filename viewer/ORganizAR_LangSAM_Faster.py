import time
import numpy as np
import cv2
from pynput import keyboard
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

from faster_haoyang import PromptBoxManager

# Settings --------------------------------------------------------------------
from_recording = True  # set to run live on HL vs from recorded dataset
visualization_enabled = False
write_data = True
wsl = False
remote_docker = False

if wsl:
    visualization_enabled = False
    path_start = "/mnt/c/Users/Marc/Desktop/CS/MARPROJECT/"
elif remote_docker:
    path_start = "/medar_smart/"
else:
    path_start = "/Users/haoyangsun/Documents/ORganizAR/"
    print("start path: ", path_start)

# HoloLens address
host = '192.168.160.253'

# Directory containing the recorded data
path = path_start + 'viewer/recorded_data/dataset1'

# Directory containing the calibration data of the HoloLens
calibration_path: str = path_start + 'calibration/rm_depth_longthrow/'

# rgb images from recorded data
write_data_path = "viewer/data/debug_faster9/"
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

model_LangSAM = LangSAM()

# Semantic search setup
device_search = torch.device("cuda" if torch.cuda.is_available() else "mps")
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device_search)
model_clip = model_clip.to(device_search)

# Test data setup
image_pil_timer = Image.open(path_start + "viewer/test_timer.png")
image_pil_bed = Image.open(path_start + "/viewer/test_bed.png")
prompts = ["A table with blue cloth on it. a grey shelf with a pole on it. c arm, which is a medical machine with a large c shaped metal arm. backpack. ultra sound machine, that has flashlight shape probe attached and a machine tower. wooden chair with metalic legs. black computer screen"]

prompts_lookup = ["A table with blue cloth on it","a grey shelf with a pole on it","c arm, which is a medical machine with a large c shaped metal arm","backpack"," ultra sound machine, that has flashlight shape probe attached and a machine tower","wooden chair with metalic legs","black computer screen"]
# prompts = ["C-Arm, which has a large C-shaped arm and machine tower"]
# prompts = ["C Arm", "Ultrasound Machine", "Laparoscopic Tower", "Chairs", "Table with blue cloth", "Patient Bed"]#["timer","bed","monitors", "orange pen", "keyboard"]
boxManager = PromptBoxManager(prompts, prompts_lookup)

images = [image_pil_timer, image_pil_bed]

data = {}
CLIP_SIM_THRESHOLD = 0.7
DINO_THRESHOLD = 0.2
MIN_FRAME_NUM = 5
enable = True

# unity pc vis secctiond
# quad scale in meters
quad_scale: List[float] = [0.005, 0.005, 0.005]
sphere_scale: List[float] = [0.001, 0.001, 0.001]
total_quad_count: int = 500


def get_boxes(image, prompt, box_threshold=0.3, text_threshold=0.25):
    boxes, logits, phrases = model_LangSAM.predict_dino(image, prompt, box_threshold, text_threshold)
    return boxes, logits, phrases


def crop_image(image_pil, box):
    left, upper, right, lower = [int(coordinate.item()) for coordinate in box]
    cropped_image = image_pil.crop((left, upper, right, lower))
    return cropped_image


def apply_clip_embedding_image(image):
    # query_image = preprocess_clip(image).unsqueeze(0).to(device_search)
    query_image = preprocess_clip(image).to(device_search)
    return query_image


def apply_clip_embedding_prompt(prompt):
    text = clip.tokenize([prompt]).to(device_search)
    return text


def get_mask(image, boxes):
    masks = model_LangSAM.predict_sam(image, boxes)
    masks = masks.squeeze(1)
    masks_np = [mask.cpu().detach().numpy() for mask in masks]
    masks_np = [(mask * 255).astype(np.uint8) for mask in masks_np]
    combined_mask = np.zeros(masks_np[0].shape, dtype=np.uint8)
    for mask in masks_np:
        combined_mask = np.maximum(combined_mask, mask)
    return combined_mask


def set_up_data_struct(prompts):
    data = {}
    for prompt in prompts:
        tokenized_prompt = apply_clip_embedding_prompt(prompt)  # Tokenizing the prompt for CLIP
        data[prompt] = {
            'tokenized_prompt': tokenized_prompt,
            'frames': {}  # This will store information per frame, to be populated later
        }
    return data


def apply_clip_one_img(image, prompt):
    with torch.no_grad():
        # logits_per_image, logits_per_text = model_clip(image, prompt)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        logits_per_image, _ = model_clip(image, prompt)  # tensor([[22.3750]], device='cuda:0', dtype=torch.float16)
        logits_value = logits_per_image.cpu().numpy()
        normalized_logits = (logits_value + 100) / 200
        return normalized_logits


def clip_calculate_similarity(prompt_embedding: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Calculates the similarity between prompt embedding and image embeddings.

    Args:
    - prompt_embedding: Tensor containing embedding of the text prompt e.g. "right monitor" or "yellow bell pepper etc".
    - image_embeddings: Tensor containing embeddings of the images of the users environment.

    Returns:
    - Tensor containing the similarities between prompt and image embeddings.
    """
    similariries = prompt_embedding @ image_embeddings.T
    return similariries


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


def apply_clip_batch(prompt, max_results, threshold_percentage):
    prompt_data = data[prompt]
    query_tokens = prompt_data["tokenized_prompt"]
    with torch.no_grad():
        prompt_embedding = model_clip.encode_text(query_tokens)

    image_embeddings = []
    box_references = []
    for _, frame_data in prompt_data['frames'].items():
        for box_data in frame_data['boxes']:
            logit = box_data['logit']
            box_data['top_result'] = False
            box_data['sim_score'] = None
            if logit > DINO_THRESHOLD:
                clip_embedded_img = box_data['embedding']
                # stack the preprocessed images together and then producd
                image_embeddings.append(clip_embedded_img)
                box_references.append(box_data)

    with torch.no_grad():
        image_embeddings = model_clip.encode_image(torch.stack(image_embeddings))
    sim = clip_calculate_similarity(prompt_embedding, image_embeddings)[0]
    print("sim")
    print(sim)
    if (len(sim) > 0):
        for box_data, score in zip(box_references, sim):
            box_data['sim_score'] = score.item()
        box_references.sort(key=lambda x: x['sim_score'], reverse=True)
        top_result_score = box_references[0]['sim_score']
        print("topscore:")
        print(top_result_score)

        for _, frame_data in prompt_data['frames'].items():
            threshold_score = top_result_score - (
                        top_result_score * threshold_percentage)  # TODO filter also based on detection rate
            for box_data in box_references:
                print(box_data['sim_score'])
                box_data['top_result'] = True
                # if box_data['sim_score'] and box_data['sim_score'] >= threshold_score:
                #    box_data['top_result'] = True
                # Limit to the maximum results specified
                max_results -= 1
                if max_results <= 0:
                    break
            if max_results <= 0:
                break


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


def instantiate_gos(ipc: hl2ss.ipc_umq) -> np.ndarray:
    """
    Instantiates quads on the server and returns their IDs.

    Args:
    - ipc: The interprocess communication object.

    Returns:
    - An array containing the IDs of the instantiated quads so we can later change their transforms, toggle their visibility etc.
    """
    global total_quad_count
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list()  # Begin command sequence
    display_list.remove_all()  # Remove all objects that were created remotely
    for i in range(total_quad_count):
        display_list.create_primitive(hl2ss_rus.PrimitiveType.Sphere)  # Create a quad, server will return its id
    display_list.end_display_list()  # End command sequence
    ipc.push(display_list)  # Send commands to server
    results = ipc.pull(display_list)  # Get results from server
    return results  # ids


if __name__ == '__main__':
    # Keyboard events ---------------------------------------------------------
    enable = True
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

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
        color_pil = Image.fromarray(color_np)
        (image_isNovel_view, index, img) = view_mana.new_view(color_np)
        if image_isNovel_view:
            # save selected images
            # os.makedirs(path_start + write_data_path)
            color_pil.save(path_start + write_data_path + "selected_frame" + str(index) + ".jpeg")
            if counter < MIN_FRAME_NUM:
                print(f"view: {counter}")
                counter += 1
            boxManager.new_frame(color_pil)

        if counter >= MIN_FRAME_NUM:  # TODO need to find better integration, e.g. UI for room scan complete then stop receiving data,already run inference before async
            color_pil.save(path_start + write_data_path + "selected_frame" + str(index) + ".jpeg")
            boxManager.new_frame(color_pil)
            boxManager.output_det()
            # for index, prompt in enumerate(data.keys()):
            #     print(prompt)
            #     apply_clip_batch(prompt, 3, 0.03)
            #     print("actually used")
            #     data[prompt]["point_cloud"] = np.array([])
            #     for timestamp, frame_data in data[prompt]['frames'].items():
            #         for box_data in frame_data['boxes']:
            #             if (box_data['top_result']):
            #                 print(box_data['sim_score'])
            #                 box_data['combined_mask'] = get_mask(box_data['color_pil'], box_data['box'])
            #                 if write_data:
            #                     combined_mask_pil = Image.fromarray(box_data['combined_mask'])
            #                     result = Image.blend(box_data["color_pil"], combined_mask_pil.convert("RGB"), alpha=0.5)
            #                     score = box_data["sim_score"]
            #                     logit = box_data["logit"]
            #                     pharse = box_data["phrase"]
            #                     print("saving to", f"{path_start}{write_data_path}{pharse}clip{score}dino{logit}.jpeg")
            #                     result.save(f"{path_start}{write_data_path}{pharse}clip{score}dino{logit}.jpeg")
            #                 box_data["points"] = get_segmented_points(box_data, depth_scale, max_depth, xy1,
            #                                                           calibration_lt)
            #                 if data[prompt]["point_cloud"].size == 0:
            #                     data[prompt]["point_cloud"] = box_data["points"]
            #                 else:
            #                     data[prompt]["point_cloud"] = np.concatenate(
            #                         (data[prompt]["point_cloud"], box_data["points"]), axis=0)
            #
            #     if (data[prompt]["point_cloud"].size == 0):
            #         continue
            #     if from_recording == False:
            #         # results = display_point_cloud(data[prompt]["point_cloud"],index ) #get bbox using open3d, create cube over bbox with outline shader, maybe add arrow TODO
            #         results = display_centroid(data[prompt]["point_cloud"], index)
            #
            # break

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
    listener.join()



