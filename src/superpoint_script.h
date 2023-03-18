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

#include <string>

#include "opencv2/core/core.hpp"
#include "torch/torch.h"

namespace superpoint_script {

// A wrapper class for the SuperPoint pre-trained TorchScript model.
class SuperPointScript {
 public:
  // Configuration options for the SuperPoint model.
  struct Options {
    int width = 640;
    int height = 480;
    float nms_dist = 4;
    float conf_thresh = 0.015;
    float border = 4;
    bool cuda = false;
  };

  // Loads the TorchScript model from the given path.
  SuperPointScript(const std::string& model_path, const Options& options);

  // Runs the model on the given image and returns the detected keypoints and
  // descriptors.
  // Input: image - The input image, which must be a grayscale image of size
  //                (height, width).
  // Outputs:
  //   keypoints - An N x 3 matrix where each row is a keypoint with columns
  //               (x, y, score).
  //   descriptors - An N x 256 matrix where each row is a descriptor.
  //   confidences - A W x H matrix of confidence scores.
  // Returns true iff the model ran successfully.
  bool Run(const cv::Mat& image,
           torch::Tensor* keypoints,
           torch::Tensor* descriptors,
           torch::Tensor* confidences);

 private:
  torch::Tensor CVImageToTensor(const cv::Mat& image);
  // The TorchScript model.
  torch::jit::script::Module model_;
  Options options_;
};

}  // namespace superpoint_script