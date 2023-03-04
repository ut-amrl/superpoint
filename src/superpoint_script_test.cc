// Copyright (c) 2023 Joydeep Biswas (joydeepb@cs.utexas.edu)

// MIT License

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <iostream>
#include <memory>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "torch/script.h" 
#include "torch/torch.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(model, "superpoint_traced_model.pt", "Path to the model file.");
DEFINE_string(input, "assets/icl_snippet/250.png", "Path to the image file.");
DEFINE_bool(cuda, false, "Use CUDA for inference.");
DEFINE_int32(width, 640, "Width of the image.");
DEFINE_int32(height, 480, "Height of the image.");
DEFINE_int32(repeat, 1, "Number of times to repeat the inference.");
DEFINE_bool(no_display, false, "Do not display the image.");

int main(int argc, char* argv[]) {
  // Initialize the gflags library.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // Initialize the Google's logging library.
  google::InitGoogleLogging(argv[0]);

  printf("CUDA: %d\n", FLAGS_cuda);
  if (FLAGS_cuda && !torch::cuda::is_available()) {
    std::cerr << "error: CUDA is not available" << std::endl;
    return -1;
  }
  
  printf("Loading model... ");
  fflush(stdout);
  torch::jit::script::Module module;
  const double t_load_start = cv::getTickCount();
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(FLAGS_model);
    if (FLAGS_cuda) module.to(at::kCUDA);
    // Hot-start the model with a random image.
    torch::Tensor tensor_image = torch::rand({1, 1, FLAGS_height, FLAGS_width});
    tensor_image = tensor_image.toType(torch::kFloat);
    if (FLAGS_cuda) tensor_image = tensor_image.to(torch::kCUDA);
    module.forward(std::vector<torch::jit::IValue>({tensor_image}));
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  const double t_load_end = cv::getTickCount();
  const double t_load = (t_load_end - t_load_start) / cv::getTickFrequency();
  printf("Done in %f ms.\n", t_load * 1000);
  
  // Load an example image from file and resize it to 
  // FLAGS_width x FLAGS_height.
  cv::Mat image = cv::imread(FLAGS_input, cv::IMREAD_GRAYSCALE);
  cv::resize(image, image, cv::Size(FLAGS_width, FLAGS_height));
  cv::Mat image_float;
  image.convertTo(image_float, CV_32F, 1.0 / 255.0);
  
  // Convert the image to a 1 x 1 x FLAGS_width x FLAGS_height tensor.
  torch::Tensor tensor_image = 
      torch::from_blob(image_float.data, {1, 1, FLAGS_height, FLAGS_width});
  tensor_image = tensor_image.toType(torch::kFloat);
  if (FLAGS_cuda) tensor_image = tensor_image.to(torch::kCUDA);

  // Interest point confidence and descriptor.
  torch::Tensor semi;
  torch::Tensor desc;
  try {
    const double start = cv::getTickCount();
    c10::intrusive_ptr<c10::ivalue::Tuple> output;
    for (int i = 0; i < FLAGS_repeat; ++i) {
      // Print a nice progress bar.
      printf("\rInference progress: %d/%d", i + 1, FLAGS_repeat);
      fflush(stdout);
      output = module.forward(
          std::vector<torch::jit::IValue>({tensor_image})).toTuple();
    }
    printf("\n");
    const double end = cv::getTickCount();
    const double time = (end - start) / cv::getTickFrequency();
    printf("Inference time: %f ms.\n", time * 1000 / FLAGS_repeat);
    CHECK_EQ(output->elements().size(), 2);
    semi = output->elements()[0].toTensor();
    desc = output->elements()[1].toTensor();
    if (FLAGS_cuda) {
      semi = semi.to(torch::kCPU);
      desc = desc.to(torch::kCPU);
    }
  } catch (const c10::Error& e) {
    std::cerr << "error running the model: " << e.msg() << std::endl;
    return -1;
  }
  
  
  // std::cout << "semi size: " << semi.sizes() << std::endl;
  // std::cout << "desc size: " << desc.sizes() << std::endl;
  // Expected sizes:
  // semi: N x 65 x H/8 x W/8.
  CHECK_EQ(semi.sizes(), 
           torch::IntArrayRef({1, 65, FLAGS_height / 8, FLAGS_width / 8}));
  // desc: N x 256 x H/8 x W/8.
  CHECK_EQ(desc.sizes(), 
           torch::IntArrayRef({1, 256, FLAGS_height / 8, FLAGS_width / 8}));
 
  if (FLAGS_no_display) {
    return 0;
  }

  semi = semi.squeeze();
  semi = semi.exp();
  semi = semi / (torch::sum(semi, 0) + 0.00001);
  
  const int kCell = 8;
  torch::Tensor nodust = semi.index({torch::indexing::Slice(0, -1), 
                                     torch::indexing::Slice(), 
                                     torch::indexing::Slice()});
  nodust = nodust.permute({1, 2, 0});
  nodust = nodust.reshape({FLAGS_height / kCell, FLAGS_width / kCell, kCell, kCell});
  nodust = nodust.permute({0, 2, 1, 3});  
  nodust = nodust.reshape({FLAGS_height, FLAGS_width});
  
  float min_conf = 0.001;
  nodust = torch::where(nodust < min_conf, torch::full_like(nodust, min_conf), nodust);
  nodust = -nodust.log();
  nodust = (nodust - nodust.min()) / (nodust.max() - nodust.min() + 0.00001);
  
  torch::Tensor myjet = torch::tensor({{0.        , 0.        , 0.5       },
                                       {0.        , 0.        , 0.99910873},
                                       {0.        , 0.37843137, 1.        },
                                       {0.        , 0.83333333, 1.        },
                                       {0.30044276, 1.        , 0.66729918},
                                       {0.66729918, 1.        , 0.30044276},
                                       {1.        , 0.90123457, 0.        },
                                       {1.        , 0.48002905, 0.        },
                                       {0.99910873, 0.07334786, 0.        },
                                       {0.5       , 0.        , 0.        }});
  nodust = nodust * 10;
  nodust = torch::clamp(nodust, 0, 9);
  nodust = nodust.round();
  nodust = myjet.index({nodust.to(torch::kInt64), 
                        torch::indexing::Slice()});
  nodust = nodust * 255;
  nodust = nodust.to(torch::kUInt8);
  
  cv::Mat heatmap = 
      cv::Mat(FLAGS_height, FLAGS_width, CV_8UC3, nodust.data_ptr());
  cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
  cv::Mat display;
  cv::hconcat(image, heatmap, display);
  cv::imshow("Display", display);
  cv::waitKey(0);
  
  return 0;
}