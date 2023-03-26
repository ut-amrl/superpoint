<img src="assets/magicleap.png" width="240">

### Research @ Magic Leap

# SuperPoint Weights File and Demo Script

## Introduction 

* Full paper PDF: [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)

* Presentation PDF: [Talk at CVPR Deep Learning for Visual SLAM Workshop 2018](assets/DL4VSLAM_talk.pdf)

Freiburg RGBD:  
<img src="assets/processed_freiburg.gif" width="240">

KITTI:  
<img src="assets/processed_kitti.gif" width="480">

Microsoft 7 Scenes:  
<img src="assets/processed_ms7.gif" width="240">

MonoVO:  
<img src="assets/processed_monovo.gif" width="240">


## Dependencies: C++
* CMake and a C++ compiler, tested on Ubuntu 22.04:
    ```sh
    sudo apt-get install cmake g++
    ```
* [OpenCV](https://opencv.org/), tested with version 4.5.4
    ```sh
    sudo apt-get install libopencv-dev
    ```
* [LibTorch](https://pytorch.org/), tested with version 1.13.1+cu117
    ```sh
    wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
    unzip libtorch-cxx11-abi-shared-with-deps-1.9.0+cu111.zip
    sudo mv libtorch /opt/
    ```
* [TorchVision](https://github.com/pytorch/vision) compiled from source with CUDA support.
 To compile the TorchVision C++ library (assumes libtorch is installed in
 `/opt/libtorch`, modify based on your installation path):
    ```sh   
    git clone git@github.com:pytorch/vision.git torchvision
    cd torchvision
    mkdir -p build
    cd build
    cmake .. -DWITH_CUDA=ON -DTorch_DIR=/opt/libtorch/share/cmake/Torch
    make -j`nproc`
    sudo make install
    ```

## Dependencies: Python
The Conda library dependencies are broken for opencv + pytorch +
torchvision with CUDA support. We recommend using the pip dependencies:
```sh
pip install opencv-python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Running the Python Demo
This demo will run the SuperPoint network on an image sequence and compute points and descriptors from the images, using a helper class called `SuperPointFrontend`. The tracks are formed by the `PointTracker` class which finds sequential pair-wise nearest neighbors using two-way matching of the points' descriptors. The demo script uses a helper class called `VideoStreamer` which can process inputs from three different input streams:

1. A directory of images, such as .png or .jpg
2. A video file, such as .mp4 or .avi
3. A USB Webcam

* Run the demo on provided directory of images in CPU-mode:
  ```sh
  ./demo_superpoint.py assets/icl_snippet/
  ```
* Run the demo on provided .mp4 file in GPU-mode:
  ```sh
  ./demo_superpoint.py assets/nyu_snippet.mp4 --cuda
  ```
* Run a live demo via webcam (id #1) in CPU-mode:
  ```sh
  ./demo_superpoint.py camera --camid=1
  ```
* Run the demo without a display on 640x480 images and write the output to `myoutput/`
  ```sh
  ./demo_superpoint.py assets/icl_snippet/ --W=640 --H=480 --no_display --write --write_dir=myoutput/
  ```
  
**Additional useful command line parameters**

* Use `--H` to change the input image height (default: 120).
* Use `--W` to change the input image width (default: 160).
* Use `--display_scale` to scale the output visualization image height and width (default: 1).
* Use `--cuda` flag to enable the GPU.
* Use `--img_glob` to change the image file extension (default: *.png).
* Use `--min_length` to change the minimum track length (default: 2).
* Use `--max_length` to change the maximum track length (default: 5).
* Use `--conf_thresh` to change the point confidence threshold (default: 0.015).
* Use `--nn_thresh` to change the descriptor matching distance threshold (default: 0.7).
* Use `--show_extra` to show more computer vision outputs.
* Use `--save_matches` to save frame to frame matches (default: False).
* Press the `q` key to quit.


## Running the C++ demo

### Run PyTorch tracing to save the model as a TorchScript file
```sh
./cpp_export.py
```

* Use `--output` flag to change the output file name (default: superpoint_v1.pt).
* Use `--cuda` flag to enable the GPU.
* Use `--H` to change the input image height (default: 480).
* Use `--W` to change the input image width (default: 640).
* Use `--weights_path` to change the path to the pretrained weights file (default: superpoint_v1.pth).

### Run Inference
```sh
./bin/superpoint_script_test
```

* Use `--model` flag to change the model file name (default: superpoint_v1.pt).
* Use `--input` flag to specify the path to the image files directory (default: assets/ut_amrl_husky/).
* Use `--cuda` flag to enable the GPU.
* Use `--height` to change the input image height (default: 480).
* Use `--width` to change the input image width (default: 640).
* Use `--num` to change the number of images to perform inference on. Images
  will be reused if there are not enough (default: 100).
* Use `--no_display` to disable the display of the output image.
