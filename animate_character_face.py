


#@title # Step 1: Setup
!git clone https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model.git

%cd Thin-Plate-Spline-Motion-Model
!mkdir checkpoints

!wget -c https://cloud.tsinghua.edu.cn/f/da8d61d012014b12a9e4/?dl=1 -O checkpoints/vox.pth.tar
#!wget -c https://cloud.tsinghua.edu.cn/f/483ef53650b14ac7ae70/?dl=1 -O checkpoints/ted.pth.tar
#!wget -c https://cloud.tsinghua.edu.cn/f/9ec01fa4aaef423c8c02/?dl=1 -O checkpoints/taichi.pth.tar
#!wget -c https://cloud.tsinghua.edu.cn/f/cd411b334a2e49cdb1e2/?dl=1 -O checkpoints/mgif.pth.tar


# @title # Step 2: Settings
# @markdown ## Import your driving video and/or image before filling in the form
# @markdown #### # Note: paths are relative to the 'Thin-Plate-Spline-Motion-Model' folder

import torch

# edit the config
device = torch.device('cuda:0')
dataset_name = 'vox'  # ['vox', 'taichi', 'ted', 'mgif']
source_image_path = './assets/source.png'  # @param {type:"string"}
driving_video_path = './assets/driving.mp4'  # @param {type:"string"}
config_path = 'config/vox-256.yaml'
checkpoint_path = 'checkpoints/vox.pth.tar'
predict_mode = 'relative'  # ['standard', 'relative', 'avd']
find_best_frame = True  # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result

pixel = 256  # for vox, taichi and mgif, the resolution is 256*256
if (dataset_name == 'ted'):  # for ted, the resolution is 384*384
    pixel = 384

if find_best_frame:
    !pip
    install
    face_alignment

output_video_path = './generated.mp4'
try:
    import imageio
    import imageio_ffmpeg
except:
    !pip
    install
    imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
import warnings
import os

warnings.filterwarnings("ignore")

source_image = imageio.imread(source_image_path)
reader = imageio.get_reader(driving_video_path)

source_image = resize(source_image, (pixel, pixel))[..., :3]

fps = reader.get_meta_data()['fps']
driving_video = []
try:
    for im in reader:
        driving_video.append(im)
except RuntimeError:
    pass
reader.close()

driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]


def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani


HTML(display(source_image, driving_video).to_html5_video())



#@title # Step 3: Run Thin-Plate-Spline-Motion-Model
from demo import load_checkpoints
inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = config_path, checkpoint_path = checkpoint_path, device = device)

from demo import make_animation
from skimage import img_as_ubyte

if predict_mode=='relative' and find_best_frame:
    from demo import find_best_frame as _find
    i = _find(source_image, driving_video, device.type=='cpu')
    print ("Best frame: " + str(i))
    driving_forward = driving_video[i:]
    driving_backward = driving_video[:(i+1)][::-1]
    predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
    predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
    predictions = predictions_backward[::-1] + predictions_forward[1:]
else:
    predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)

#save resulting video
imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

HTML(display(source_image, driving_video, predictions).to_html5_video())



#@title # Step 4: upsize and export video
#@markdown ## Output file is upsized.mp4
%cd ..
!apt install ffmpeg
!apt install libmagic1 python3-yaml
!apt install libvulkan-dev
!pip install --user youtube-dl
!wget https://github.com/k4yt3x/video2x/archive/refs/tags/4.7.0.tar.gz

!tar -xvf 4.7.0.tar.gz
%cd video2x-4.7.0/src
!pip install -r /content/video2x-4.7.0/src/requirements.txt
!rm -rf /content/video2x-4.7.0/src/video2x.yaml
!wget -O /content/video2x-4.7.0/src/video2x.yaml https://raw.githubusercontent.com/lenardcarroll/video2x.yaml/main/video2x.yaml
%cd ../..
!wget https://github.com/nihui/realsr-ncnn-vulkan/releases/download/20200818/realsr-ncnn-vulkan-20200818-linux.zip
!unzip realsr-ncnn-vulkan-20200818-linux.zip
# !wget https://github.com/nihui/srmd-ncnn-vulkan/releases/download/20200818/srmd-ncnn-vulkan-20200818-linux.zip
# !unzip srmd-ncnn-vulkan-20200818-linux
# !wget https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20200818/waifu2x-ncnn-vulkan-20200818-linux.zip
# !unzip waifu2x-ncnn-vulkan-20200818-linux.zip
!rm *-linux.zip
!pip install -U PyYAML

!python video2x-4.7.0/src/video2x.py -i ./Thin-Plate-Spline-Motion-Model/generated.mp4 -o ./upsized.mp4 -d realsr_ncnn_vulkan -h 512 -w 512


#@title # Step 5 (optional): Save frames
!rm -rf frames
!rm frames.zip
!mkdir frames
!ffmpeg -i upsized.mp4 frames/out%03d.png
!zip -r frames.zip frames/


#@title # Step 6 (optional): fix face with GFPGAN
!rm -rf fixed
!mkdir fixed

#installing the dependencies
# Install pytorch
!pip install torch torchvision

# Check torch and cuda versions
import torch
print('Torch Version: ', torch.__version__)
print('CUDA Version: ', torch.version.cuda)
print('CUDNN Version: ', torch.backends.cudnn.version())
print('CUDA Available:', torch.cuda.is_available())


# Install basicsr - https://github.com/xinntao/BasicSR
# We use BasicSR for both training and inference.
# Set BASICSR_EXT=True to compile the cuda extensions in the BasicSR - It may take several minutes to compile, please be patient.
!BASICSR_EXT=True pip install basicsr


# Install facexlib - https://github.com/xinntao/facexlib
# We use face detection and face restoration helper in the facexlib package
!pip install facexlib
!mkdir -p /usr/local/lib/python3.7/dist-packages/facexlib/weights  # for pre-trained models


!rm -rf GFPGAN
!git clone https://github.com/TencentARC/GFPGAN.git
%cd GFPGAN

# install extra requirements
!pip install -r requirements.txt

!python setup.py develop

# If you want to enhance the background (non-face) regions with Real-ESRGAN,
# you also need to install the realesrgan package
!pip install realesrgan


#loading the pretrained GAN Models
!wget https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth -P experiments/pretrained_models


!python inference_gfpgan.py -i ../frames -o ../fixed --aligned

%cd ..

!rm fixed.zip
!zip -r fixed.zip fixed/restored_faces



#