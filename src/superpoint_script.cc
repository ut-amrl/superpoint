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

#include "superpoint_script.h"

#include <string>

#include "glog/logging.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "torch/script.h"
#include "torch/torch.h"
#include "torchvision/vision.h"
#include "torchvision/ops/nms.h"

using std::string;

namespace superpoint_script {

SuperPointScript::SuperPointScript(const string& model_path,
                                   const Options& options)
    : options_(options) {
  try {
    // Deserialize the ScriptModule from model_path.
    model_ = torch::jit::load(model_path);
    if (options_.cuda) model_.to(at::kCUDA);
    // Hot-start the model with a random image.
    torch::Tensor tensor_image = torch::rand(
        {1, 1, options_.height, options_.width});
    tensor_image = tensor_image.toType(torch::kFloat);
    if (options_.cuda) tensor_image = tensor_image.to(torch::kCUDA);
    model_.forward(std::vector<torch::jit::IValue>({tensor_image}));
  } catch (const c10::Error& e) {
    LOG(FATAL) << "Error loading the model from " << model_path;
  }
}

torch::Tensor SuperPointScript::CVImageToTensor(const cv::Mat& image) {
  torch::Tensor tensor_image =
      torch::from_blob(image.data, {1, 1, options_.height, options_.width});
  tensor_image = tensor_image.toType(torch::kFloat);
  if (options_.cuda) tensor_image = tensor_image.to(torch::kCUDA);
  CHECK(std::isfinite(tensor_image.max().item<float>()));
  return tensor_image;
}

bool SuperPointScript::Run(const cv::Mat& image,
                           torch::Tensor* keypoints,
                           torch::Tensor* descriptors,
                           torch::Tensor* confidences) {
  // Convert the image to a tensor.
  torch::Tensor tensor_image = CVImageToTensor(image);
  // Run the model.
  c10::intrusive_ptr<c10::ivalue::Tuple> outputs = model_.forward(
      std::vector<torch::jit::IValue>({tensor_image})).toTuple();
  if (outputs->elements().size() != 2) {
    LOG(ERROR) << "Expected 2 outputs from the model, got "
               << outputs->elements().size();
    return false;
  }
  torch::Tensor semi = outputs->elements()[0].toTensor();
  torch::Tensor desc = outputs->elements()[1].toTensor();
  return true;
}

}  // namespace superpoint_script