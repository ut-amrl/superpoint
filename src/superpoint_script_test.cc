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

#include <stdarg.h>
#include <stdio.h>
#include <iostream>
#include <memory>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "torch/script.h"
#include "torch/torch.h"
#include "torchvision/vision.h"
#include "torchvision/ops/nms.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "superpoint_script.h"

using superpoint_script::SuperPointScript;
using superpoint_script::SuperPointScript::Options;

DEFINE_string(model, "superpoint_v1.pt", "Path to the model file.");
DEFINE_string(input, "assets/ut_amrl_husky/",
    "Path to the image files directory.");
DEFINE_bool(cuda, false, "Use CUDA for inference.");
DEFINE_int32(width, 640, "Width of the image.");
DEFINE_int32(height, 480, "Height of the image.");
DEFINE_int32(num, 100, "Number of images to perform inference on -- will repeat"
    " images if there are not enough images in the directory.");
DEFINE_bool(no_display, false, "Do not display the image.");
DEFINE_double(min_conf, 0.015, "Minimum confidence for a keypoint.");
DEFINE_int32(nms_dist, 4, "Non-maximum suppression distance.");
DEFINE_double(border, 4, "Border to remove from the image.");
DEFINE_string(write_dir, "", "Directory to write the output images to.");
DECLARE_int32(v);

void LoadImages(const std::string& path, int N,  std::vector<cv::Mat>* images) {
  std::vector<std::string> files;
  cv::glob(path, files);
  CHECK(!files.empty()) << "No files found in " << FLAGS_input;
  CHECK(images != nullptr);
  images->clear();
  for (const auto& file : files) {
    printf("\rLoading image %d/%d: %s    ",
           static_cast<int>(images->size()), N, file.c_str());
    fflush(stdout);
    try {
      cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);
      // If the image is empty, skip it.
      if (image.empty()) {
        LOG(WARNING) << "Not a valid image: " << file;
        continue;
      }
      cv::resize(image, image, cv::Size(FLAGS_width, FLAGS_height));
      cv::Mat image_float;
      image.convertTo(image_float, CV_32F, 1.0 / 255.0);
      images->push_back(image_float);
    } catch (const std::exception& ex) {
      LOG(ERROR) << "Could not load image " << file << ": " << ex.what();
      continue;
    }
    if (images->size() == N) break;
  }
  printf("\n");
}

torch::Tensor CVImageToTensor(const cv::Mat& image) {
  torch::Tensor tensor_image =
      torch::from_blob(image.data, {1, 1, FLAGS_height, FLAGS_width});
  tensor_image = tensor_image.toType(torch::kFloat);
  if (FLAGS_cuda) tensor_image = tensor_image.to(torch::kCUDA);
  CHECK(std::isfinite(tensor_image.max().item<float>()));
  return tensor_image;
}

// Performn non-maximum suppression on the interest-point and descriptor outputs
// from SuperPoint.
// Inputs:
//   semi: N x 2, where N is the number of keypoints.
torch::Tensor GetKeypoints(torch::Tensor& semi_orig,
                           torch::Tensor& coarse_desc) {
  torch::Tensor semi = semi_orig.clone();
  // Expected sizes:
  // semi: N x 65 x H/8 x W/8.
  printf("\n");
  CHECK_EQ(semi.sizes(),
           torch::IntArrayRef({1, 65, FLAGS_height / 8, FLAGS_width / 8}));
  // desc: N x 256 x H/8 x W/8.
  CHECK_EQ(coarse_desc.sizes(),
           torch::IntArrayRef({1, 256, FLAGS_height / 8, FLAGS_width / 8}));

  semi = semi.squeeze();
  semi = semi.exp();
  semi = semi / (torch::sum(semi, 0) + 0.00001);

  const int kCell = 8;
  torch::Tensor nodust = semi.index({torch::indexing::Slice(0, -1),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice()});
  nodust = nodust.permute({1, 2, 0});
  nodust = nodust.reshape(
      {FLAGS_height / kCell, FLAGS_width / kCell, kCell, kCell});
  nodust = nodust.permute({0, 2, 1, 3});
  nodust = nodust.reshape({FLAGS_height, FLAGS_width});

  // Find the coordinates of the points in nodust with values greater than 0.5.
  torch::Tensor points =
      torch::nonzero(nodust > FLAGS_min_conf).toType(torch::kInt64);

  torch::Tensor nms_points;

  try {
    const auto PrintTensor = [](
        const std::string s, const torch::Tensor& tensor) {
      if (FLAGS_v < 2) return;
      std::cout << std::endl << s << ":" << tensor.sizes() << std::endl;
      std::cout << tensor.slice(0, 0, 10) << std::endl;
    };
    torch::Tensor xs =
        points.index({torch::indexing::Slice(), 1}).reshape({-1, 1});
    torch::Tensor ys =
        points.index({torch::indexing::Slice(), 0}).reshape({-1, 1});
    // Rmove entries from xs and ys that are too close to the edge.
    torch::Tensor remove =
      (xs >= FLAGS_border) &
      (xs <= FLAGS_width - FLAGS_border) &
      (ys >= FLAGS_border) &
      (ys <= FLAGS_height - FLAGS_border);
    xs = xs.masked_select(remove).reshape({-1, 1});
    ys = ys.masked_select(remove).reshape({-1, 1});
    torch::Tensor conf =  nodust.index({ys, xs});
    torch::Tensor x1 = (xs - FLAGS_nms_dist).clamp(0, FLAGS_width - 1);
    torch::Tensor x2 = (xs + FLAGS_nms_dist).clamp(0, FLAGS_width - 1);
    torch::Tensor y1 = (ys - FLAGS_nms_dist).clamp(0, FLAGS_height - 1);
    torch::Tensor y2 = (ys + FLAGS_nms_dist).clamp(0, FLAGS_height - 1);
    // PrintTensor("x1", x1);
    // PrintTensor("x2", x2);
    torch::Tensor bboxes =
        torch::cat({x1, y1, x2, y2}, 1).toType(torch::kFloat);
    // PrintTensor("bboxes", bboxes);
    // Reshape conf to a 1D tensor.
    conf = conf.reshape({-1});
    torch::Tensor nms_indexes = vision::ops::nms(bboxes, conf, 0.0);
    // PrintTensor("nms_indexes", nms_indexes);

    nms_points = torch::cat({
        xs.index({nms_indexes}),
        ys.index({nms_indexes}),
        conf.index({nms_indexes}).reshape({-1, 1})
        }, 1);
    // Print the % of points that were retained by NMS.
    float num_retained = nms_points.size(0) / (float)xs.size(0);
    printf("NMS retained %6ld / %6ld points (%d%%)\n",
           nms_points.size(0),
           xs.size(0),
           int(num_retained * 100.0));
    // TODO: Sort the points by confidence.
    // PrintTensor("nms_points", nms_points);

    // Process the descriptors.

    const int kD = coarse_desc.sizes()[1];
    torch::Tensor desc;
    if (nms_points.sizes()[1] == 0) {
      desc = torch::zeros({kD, 0});
    } else {
      // Interpolate into descriptor map using 2D point locations.
      torch::Tensor samp_pts = nms_points.index({torch::indexing::Slice(),
                                                 torch::indexing::Slice(0, 2)});
      // PrintTensor("samp_pts", samp_pts);
      if (FLAGS_cuda) {
        samp_pts = samp_pts.to(torch::kCUDA);
      }
      samp_pts[0] = (samp_pts[0] / (float(FLAGS_width) / 2.)) - 1.;
      samp_pts[1] = (samp_pts[1] / (float(FLAGS_height) / 2.)) - 1.;
      samp_pts = samp_pts.transpose(0, 1).contiguous();
      samp_pts = samp_pts.view({1, 1, -1, 2});
      samp_pts = samp_pts.toType(torch::kFloat);
      desc = torch::nn::functional::grid_sample(
          coarse_desc,
          samp_pts,
          torch::nn::functional::GridSampleFuncOptions().align_corners(false));
      desc = desc.reshape({kD, -1});
      desc = desc / torch::norm(desc, 2, 0, true);
      std::cout << "desc: " << desc.sizes() << std::endl;
    }

  } catch (const std::exception& ex) {
    LOG(ERROR) << "Could not get keypoints: " << ex.what();
  }
  std::cout << "nms_points: " << nms_points.sizes() << std::endl;
  return nms_points;
}

void DisplayResults(cv::Mat image,
                    torch::Tensor& semi,
                    torch::Tensor& desc,
                    torch::Tensor& nms_points,
                    const std::string& filename) {
  // Expected sizes:
  // semi: N x 65 x H/8 x W/8.
  CHECK_EQ(semi.sizes(),
           torch::IntArrayRef({1, 65, FLAGS_height / 8, FLAGS_width / 8}));
  // desc: N x 256 x H/8 x W/8.
  CHECK_EQ(desc.sizes(),
           torch::IntArrayRef({1, 256, FLAGS_height / 8, FLAGS_width / 8}));

  semi = semi.squeeze();
  semi = semi.exp();
  semi = semi / (torch::sum(semi, 0) + 0.00001);

  const int kCell = 8;
  torch::Tensor nodust = semi.index({torch::indexing::Slice(0, -1),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice()});
  nodust = nodust.permute({1, 2, 0});
  nodust = nodust.reshape(
      {FLAGS_height / kCell, FLAGS_width / kCell, kCell, kCell});
  nodust = nodust.permute({0, 2, 1, 3});
  nodust = nodust.reshape({FLAGS_height, FLAGS_width});

  nodust = torch::where(nodust < FLAGS_min_conf,
                        torch::full_like(nodust, FLAGS_min_conf),
                        nodust);
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

  cv::Mat display;
  try {
    cv::Mat heatmap =
        cv::Mat(FLAGS_height, FLAGS_width, CV_8UC3, nodust.data_ptr());
    cv::Mat image_8u;
    image.convertTo(image_8u, CV_8U, 255.0);
    cv::cvtColor(image_8u, image_8u, cv::COLOR_GRAY2BGR);
    // Draw the keypoints.
    int num_keypoints = nms_points.sizes()[0];
    std::cout << "nms_points: " << nms_points.sizes() << std::endl;
    for (int i = 0; i < num_keypoints; ++i) {
      const float x = nms_points[i][0].item<float>();
      const float y = nms_points[i][1].item<float>();
      cv::drawMarker(image_8u,
                     cv::Point(x, y),
                     cv::Scalar(0, 255, 255),
                     cv::MARKER_TILTED_CROSS,
                     4,
                     1);
    }
    cv::hconcat(image_8u, heatmap, display);
  } catch (const std::exception& e) {
    std::cout << "Exception: " << e.what() << std::endl;
  }
  if (!filename.empty()) {
    cv::imwrite(filename, display);
  }
  if (!FLAGS_no_display) {
    cv::imshow("Display", display);
    cv::waitKey(30);
  }
}

torch::jit::script::Module LoadSuperPointModel(
    const std::string& file, bool cuda) {
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from file.
    module = torch::jit::load(file);
    if (cuda) module.to(at::kCUDA);
    // Hot-start the model with a random image.
    torch::Tensor tensor_image = torch::rand({1, 1, FLAGS_height, FLAGS_width});
    tensor_image = tensor_image.toType(torch::kFloat);
    if (cuda) tensor_image = tensor_image.to(torch::kCUDA);
    module.forward(std::vector<torch::jit::IValue>({tensor_image}));
  } catch (const c10::Error& e) {
    LOG(FATAL) << "Error loading the model from " << file;
  }
  return module;
}

std::string StringPrintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  const int kBufferSize = 1024;
  char buffer[kBufferSize];
  vsnprintf(buffer, kBufferSize, format, args);
  va_end(args);
  return std::string(buffer);
}

SuperPointScript::Options LoadOptions() {
  SuperPointScript::Options options;
  options.width = FLAGS_width;
  options.height = FLAGS_height;
  options.nms_dist = FLAGS_nms_dist;
  options.conf_thresh = FLAGS_conf_thresh;
  options.border = FLAGS_border;
  options.cuda = FLAGS_cuda;
  return options;
}

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
  SuperPointScript::Options options = LoadOptions();
  SuperPointScript superpoint(FLAGS_model, options);
  const double t_load_start = cv::getTickCount();
  torch::jit::script::Module module =
      LoadSuperPointModel(FLAGS_model, FLAGS_cuda);
  const double t_load_end = cv::getTickCount();
  const double t_load = (t_load_end - t_load_start) / cv::getTickFrequency();
  printf("Done in %f ms.\n", t_load * 1000);

  printf("Loading images... ");
  fflush(stdout);
  std::vector<cv::Mat> images;
  LoadImages(FLAGS_input, FLAGS_num, &images);
  printf("Done.\n");
  CHECK(!images.empty());
  printf("Loaded %lu images.\n", images.size());

  const double start = cv::getTickCount();
  for (int i = 0; i < FLAGS_num; ++i) {
    c10::intrusive_ptr<c10::ivalue::Tuple> output;
    printf("\rInference progress: %d/%d", i + 1, FLAGS_num);
    fflush(stdout);
    cv::Mat image = images[i % images.size()];
    try {
      torch::Tensor tensor_image = CVImageToTensor(image);
      output = module.forward(
          std::vector<torch::jit::IValue>({tensor_image})).toTuple();
    } catch (const c10::Error& e) {
      std::cerr << "error running the model: " << e.msg() << std::endl;
      return -1;
    }
    CHECK_EQ(output->elements().size(), 2);
    torch::Tensor semi = output->elements()[0].toTensor();
    torch::Tensor desc = output->elements()[1].toTensor();
    torch::Tensor nms_points = GetKeypoints(semi, desc);
    std::cout << "nms_points: " << nms_points.sizes() << std::endl;
    std::string filename;
    if (!FLAGS_write_dir.empty()) {
      filename = StringPrintf("%s/%06d.png", FLAGS_write_dir.c_str(), i);
    }
    if (!FLAGS_no_display || !filename.empty()) {
      if (FLAGS_cuda) {
        semi = semi.to(torch::kCPU);
        desc = desc.to(torch::kCPU);
        nms_points = nms_points.to(torch::kCPU);
      }
      DisplayResults(image, semi, desc, nms_points, filename);
    }
  }
  printf("\n");
  const double end = cv::getTickCount();
  const double time = (end - start) / cv::getTickFrequency();
  printf("Inference time: %f ms.\n", time * 1000 / FLAGS_num);
  return 0;
}