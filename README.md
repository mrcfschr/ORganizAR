## Server Setup:

# Create and activate a new Conda environment and make sure pip up to date
```
conda create --name organizar python=3.10.13
conda activate organizar
python -m pip install --upgrade pip
```

# Install necessary Python packages
```
pip install -r requirements.txt
```

# Using the app
- set hl ip, calib path, data path at top of the file you are running
- mar_simple_player.py shows recorded dataset:
  
![ezgif-2-b9b2dd4bda](https://github.com/mrcfschr/ORganizAR/assets/57159035/de67eaf0-969e-46b0-bdfd-892194197b99)
- full video of playback: https://drive.google.com/file/d/1MfWjMjsjJU1M9kBvx0dC8O5YyjcCmlVN/view?usp=sharing
- reconstruction.py allows to do rgbd integration on both the recorded data and live data (set from_recording flag)

![integrationgif](https://github.com/mrcfschr/ORganizAR/assets/57159035/daa3b720-da89-4452-969d-5596a36d0ea0)
- full video of final reconstruction based of live real-time data: https://drive.google.com/file/d/1UQcPHjkYZQimqTq96i-xyh11i3Llk3oz/view?usp=sharing

- for mvp using langsam and clip see ORganizAR_LangSAM.py https://drive.google.com/file/d/1ClUKq9olohGyWB9jhLV8bFvrOwVMUIJL/view?usp=sharing

# Using docker on remote machine
Use either `nvcr.io/nvidia/pytorch:23.09-py3` image for pure pytorch environment, or `coda:23.09-py3.10-tensorboardx` with all CoDA dependencies installed.

`orgnizar:lang_sam` image is now available with opencv and lang_sam installed. 
***TODO: add open3d in this docker***

To start a docker container from image, run the following command:
```
docker run --gpus all --ipc=host  -it --rm -p 6002-6008:6002-6008 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY -v /data/projects/medar_smart:/medar_smart YOUR_IMAGE_NAME:YOUR_TAG
```

The `/data/projects/medar_smart` directory will be mounted to `/medar_smart` inside the container. So please put all the files under that path. The created container will be deleted after exiting.

<details>
  <summary>Unity Side Setup (deploy ORganizAR Scene to HoloLens) and other documentation from HoloLens 2 Sensor Streaming</summary>

HoloLens 2 server software and Python client library for streaming sensor data via TCP. Created to stream HoloLens data in real time over WiFi to a Linux machine for research purposes but also works on Windows and macOS. The server is offered as a standalone application (appxbundle) or Unity plugin (dll).

**Supported interfaces**

- Research Mode Visible Light Cameras (640x480 @ 30 FPS, Grayscale, H264 or HEVC encoded)
  - Left Front
  - Left Left
  - Right Front
  - Right Right
- Research Mode Depth
  - AHAT (512x512 @ 45 FPS, 16-bit Depth + 16-bit AB, H264 or HEVC encoded or Lossless* Zdepth for Depth)
  - Long Throw (320x288 @ 5 FPS, 16-bit Depth + 16-bit AB, PNG encoded)
- Research Mode IMU
  - Accelerometer (m/s^2)
  - Gyroscope (deg/s)
  - Magnetometer
- Front Camera (1920x1080 @ 30 FPS, RGB, H264 or HEVC encoded)
- Microphone (2 channels @ 48000 Hz, 16-bit PCM, AAC encoded or 5 channels @ 48000 Hz, 32-bit Float)
- Spatial Input (30 Hz)
  - Head Tracking
  - Eye Tracking
  - Hand Tracking
- Spatial Mapping (3D Meshes)
- Scene Understanding (3D Meshes + Semantic labels for planar surfaces)
- Voice Input
- Extended Eye Tracking (30, 60, or 90 FPS)
- Extended Audio (Microphone + Application audio, 2 channels @ 48000 Hz, 16-bit PCM, AAC encoded)
  
**Additional features**

- Download calibration data (e.g., camera intrinsics, extrinsics, undistort maps) for the Front Camera and Research Mode sensors (except RM IMU Magnetometer).
- Optional per-frame pose for the Front Camera and Research Mode sensors.
- Support for Mixed Reality Capture (Holograms in Front Camera video).
- Client can configure the bitrate and properties of the [H264](https://learn.microsoft.com/en-us/windows/win32/medfound/h-264-video-encoder), [HEVC](https://learn.microsoft.com/en-us/windows/win32/medfound/h-265---hevc-video-encoder), and [AAC](https://learn.microsoft.com/en-us/windows/win32/medfound/aac-encoder) encoded streams.
- Client can configure the resolution and framerate of the Front Camera. See [here](https://github.com/jdibenes/hl2ss/blob/main/etc/pv_configurations.txt) for a list of supported configurations.
- Client can configure the focus, white balance, and exposure of the Front Camera. See [here](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_ipc_rc.py).
- Frame timestamps can be converted to [Windows FILETIME](https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-filetime) (UTC) for external synchronization. See [here](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_ipc_rc.py).
- Client can send messages to a Unity application using the plugin.
- Server application can run in background (alongside other applications) when running in flat mode.
- [C++ client library](https://github.com/jdibenes/hl2ss/tree/main/extensions).
- [MATLAB client (MEX)](https://github.com/jdibenes/hl2ss/tree/main/extensions).

**Technical Report** 

Our [paper](https://arxiv.org/abs/2211.02648) provides an overview of the code, features, and examples for the first released version of the application (1.0.11.0). For newer versions, please refer to the examples in the [viewer](https://github.com/jdibenes/hl2ss/tree/main/viewer) directory. If hl2ss is useful for your research, please cite our report:
```
@article{dibene2022hololens,
  title={HoloLens 2 Sensor Streaming},
  author={Dibene, Juan C and Dunn, Enrique},
  journal={arXiv preprint arXiv:2211.02648},
  year={2022}
}
```

## Preparation

Before using the server software configure your HoloLens as follows:

1. Update your HoloLens: Settings -> Update & Security -> Windows Update.
2. Enable developer mode: Settings -> Update & Security -> For developers -> Use developer features.
3. Enable device portal: Settings -> Update & Security -> For developers -> Device Portal.
4. Enable research mode: Refer to the Enabling Research Mode section in [HoloLens Research Mode](https://docs.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/research-mode).

Please note that enabling Research Mode on the HoloLens increases battery usage.

## Installation (sideloading)

The server application is distributed as a single appxbundle file and can be installed using one of the two following methods.

**Method 1 (local)**

1. On your HoloLens, open Microsoft Edge and navigate to this repository.
2. Download the [latest appxbundle](https://github.com/jdibenes/hl2ss/releases).
3. Open the appxbundle and tap Install.

**Method 2 (remote)**

1. Download the [latest appxbundle](https://github.com/jdibenes/hl2ss/releases).
2. Go to the Device Portal and navigate to Views -> Apps. Under Deploy apps, select Local Storage, click Browse, and select the appxbundle.
3. Click Install, wait for the installation to complete, then click Done.

You can find the server application (hl2ss) in the All apps list.

## Permissions

The first time the server runs it will ask for the necessary permissions to access sensor data. If there are any issues please verify that the server application (hl2ss.exe) has access to:

- Camera (Settings -> Privacy -> Camera).
- Eye tracker (Settings -> Privacy -> Eye tracker).
- Microphone (Settings -> Privacy -> Microphone).
- User movements (Settings -> Privacy -> User movements).

## Python client

The Python scripts in the [viewer](https://github.com/jdibenes/hl2ss/tree/main/viewer) directory demonstrate how to connect to the server, receive the data, unpack it, and decode it in real time. Additional samples show how to associate data from multiple streams. Run the server on your HoloLens and set the host variable of the Python scripts to your HoloLens IP address.

**Interfaces**

- RM VLC: [viewer/client_stream_rm_vlc.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_stream_rm_vlc.py)
- RM Depth AHAT: [viewer/client_stream_rm_depth_ahat.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_stream_rm_depth_ahat.py)
- RM Depth Long Throw: [viewer/client_stream_rm_depth_longthrow.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_stream_rm_depth_longthrow.py)
- RM IMU: [viewer/client_stream_rm_imu.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_stream_rm_imu.py)
- Front Camera: [viewer/client_stream_pv.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_stream_pv.py)
- Microphone: [viewer/client_stream_microphone.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_stream_microphone.py)
- Spatial Input: [viewer/client_stream_si.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_stream_si.py)
- Remote Configuration: [viewer/client_ipc_rc.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_ipc_rc.py)
- Spatial Mapping: [viewer/client_ipc_sm.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_ipc_sm.py)
- Scene Understanding: [viewer/client_ipc_su.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_ipc_su.py)
- Voice Input: [viewer/client_ipc_vi.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_ipc_vi.py)
- Unity Message Queue: [viewer/client_ipc_umq.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_ipc_umq.py) (Plugin Only)
- Extended Eye Tracking: [viewer/client_stream_eet.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_stream_eet.py)
- Extended Audio: [viewer/client_stream_extended_audio.py](https://github.com/jdibenes/hl2ss/blob/main/viewer/client_stream_extended_audio.py)

**Required packages**

- [OpenCV](https://github.com/opencv/opencv-python) `pip install opencv-python`
- [PyAV](https://github.com/PyAV-Org/PyAV) `pip install av`
- [NumPy](https://numpy.org/) `pip install numpy`

**Optional packages**

- [pynput](https://github.com/moses-palmer/pynput) `pip install pynput`
- [Open3D](http://www.open3d.org/) `pip install open3d`
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) `pip install PyAudio`
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [scipy](https://scipy.org/) `pip install scipy`
- [tqdm](https://github.com/tqdm/tqdm) `pip install tqdm`

## Unity plugin

For streaming sensor data from a Unity application.
All interfaces are supported.
A sample Unity project (2020.3.42f1) can be found in the [unity](https://github.com/jdibenes/hl2ss/tree/main/unity) directory. If you wish to create a new project you can start [here](https://learn.microsoft.com/en-us/training/modules/learn-mrtk-tutorials/1-1-introduction).

**Build and run the sample project**

1. Open the project in Unity. If the MRTK Project Configurator window pops up just close it.
2. Go to Build Settings (File -> Build Settings).
3. Switch to Universal Windows Platform.
4. Set Target Device to HoloLens.
5. Set Architecture to ARM64.
6. Set Build and Run on Remote Device (via Device Portal).
7. Set Device Portal Address to your HoloLens IP address (e.g., https://192.168.1.7) and set your Device Portal Username and Password.
8. Click Build and Run. Unity may ask for a Build folder. You can create a new one named Build.

**Adding the plugin to an existing Unity Project**

1. Download the [latest plugin zip file](https://github.com/jdibenes/hl2ss/releases) and extract the Assets folder into your Unity project folder.
2. In the Unity Editor configure the hl2ss, Eye Tracking, and Scene Understanding DLLs as UWP ARM64.
    1. In the Project window navigate to Assets/Plugins/WSA, select the DLL, and then go to the Inspector window.
    2. Set SDK to UWP.
    3. Set CPU to ARM64.
    4. Click Apply.
3. Add the Hololens2SensorStreaming.cs script to the Main Camera.
4. Enable the following capabilities (Edit -> Project Settings -> Player -> Publishing Settings):
    - InternetClientServer
    - InternetClient
    - PrivateNetworkClientServer
    - Webcam
    - Microphone    
    - Spatial Perception
    - Gaze Input
5. The plugin also requires the perceptionSensorsExperimental and backgroundSpatialPerception capabilities, which are not available in the Publishing Settings capabilities list. The Editor folder in the plugin zip file contains a script (BuildPostProcessor.cs) that adds the capabilities automatically after building the project. Just extract the Editor folder into the Assets folder of your Unity project. Alternatively, you can manually edit the Package.appxmanifest after building. See [here](https://github.com/jdibenes/hl2ss/blob/main/hl2ss/hl2ss/Package.appxmanifest) for an example.

**Remote Unity Scene**

The plugin has basic support for creating and controlling 3D primitives and text objects via TCP for the purpose of sending feedback to the HoloLens user. See the unity_sample Python scripts in the [viewer](https://github.com/jdibenes/hl2ss/tree/main/viewer) directory for some examples. Some of the supported features include:

- Create primitive: sphere, capsule, cylinder, cube, plane, and quad.
- Set active: enable or disable game object.
- Set world transform: position, rotation, and scale.
- Set local transform: position, rotation, and scale w.r.t. Main Camera.
- Set color: rgba with support for semi-transparency.
- Set texture: upload png or jpg file.
- Create text: creates a TextMeshPro object.
- Set text: sets the text, font size and color of a TextMeshPro object.
- Text to speech: upload text.
- Remove: destroy game object.
- Remove all: destroy all game objects created by the plugin.

To enable this functionality add the RemoteUnityScene.cs script to the Main Camera and set the Material field to BasicMaterial.

## Build from source and deploy

Building the server application and the Unity plugin requires a Windows 10 machine.

1. [Install the tools](https://docs.microsoft.com/en-us/windows/mixed-reality/develop/install-the-tools).
2. Open the Visual Studio solution (sln file in the [hl2ss](https://github.com/jdibenes/hl2ss/tree/main/hl2ss) folder) in Visual Studio 2022.
3. Set build configuration to Release ARM64. Building for x86 and x64 (HoloLens emulator), and ARM is not supported.
4. Right click the hl2ss project and select Properties. Navigate to Configuration Properties -> Debugging and set Machine Name to your HoloLens IP address.
5. Build (Build -> Build Solution). If you get an error saying that hl2ss.winmd does not exist, copy the hl2ss.winmd file from [etc](https://github.com/jdibenes/hl2ss/tree/main/etc) into the hl2ss\ARM64\Release\hl2ss folder.
6. Run (Remote Machine). You may need to [pair your HoloLens](https://learn.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/using-visual-studio?tabs=hl2#pairing-your-device) first. 

The server application will remain installed on the HoloLens even after power off. The Unity plugin is in the hl2ss\ARM64\Release\plugin folder.
If you wish to create the server application appxbundle, right click the hl2ss project and select Publish -> Create App Packages.

## Known issues and limitations

- Multiple streams can be active at the same time but only one client per stream is allowed.
- Occasionally, the server might crash when accessing the Front Camera and RM Depth Long Throw streams simultaneously. See https://github.com/microsoft/HoloLens2ForCV/issues/142.
- The RM Depth AHAT and RM Depth Long Throw streams cannot be accessed simultaneously.
- Spatial Input is not supported in flat mode.

## References

This project uses the HoloLens 2 Research Mode API and the Cannon library, both available at the [HoloLens2ForCV](https://github.com/microsoft/HoloLens2ForCV) repository.
Lossless* depth compression enabled by the [Zdepth](https://github.com/catid/Zdepth) library.

</details>


