/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/optim/Utils.h"

namespace fl {

double clipGradNorm(const std::vector<Variable>& parameters, double maxNorm) {
  double gradNorm = 0.0;
  for (const auto& p : parameters) {
    if (!p.isGradAvailable()) {
      continue;
    }
    const auto& grad = p.grad().array();
    // ArrayFire v3.7.1 doesn't support computing the norm of an f16 tensor.
    // Only if gradients are fp16, compute the norm on grads that are cast to
    // f32. This cast can be removed when support for fp16 inputs is added.
    // https://github.com/arrayfire/arrayfire/blob/v3.7.1/src/api/c/norm.cpp#L128
    gradNorm += af::sum<double>(grad * grad); 
    // if (grad.type() == af::dtype::f16) {
    //   float scaler = std::sqrt(grad.elements());
    //   auto tmp = af::flat(grad) / scaler;
    //   gradNorm +=
    //       af::sum(tmp * tmp, 0).as(f32).scalar<float>() * scaler * scaler;
    // } else {
    //   gradNorm += std::pow(af::norm(af::flat(grad)), 2);
    // }
    // gradNorm += std::pow(
    //     af::norm(af::flat(
    //         (grad.type() == af::dtype::f16 ? grad.as(af::dtype::f32) :
    //         grad))),
    //     2);
  }
  gradNorm = std::sqrt(gradNorm);
  double scale = (maxNorm / gradNorm);
  if (scale < 1.0) {
    for (auto& p : parameters) {
      if (!p.isGradAvailable()) {
        continue;
      }
      auto& grad = p.grad().array();
      grad = scale * grad;
      p.grad().array() = grad;
    }
  }
  return gradNorm;
}

} // namespace fl
