/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/Transformer.h"
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace {
fl::Variable transformerInitLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std, af::dtype::f32, true);
}
} // namespace

namespace fl {

Transformer::Transformer(
    int32_t modelDim,
    int32_t headDim,
    int32_t mlpDim,
    int32_t nHeads,
    int32_t bptt,
    float pDropout,
    float pLayerdrop,
    bool useMask,
    bool preLN)
    : nHeads_(nHeads),
      bptt_(bptt),
      pDropout_(pDropout),
      pLayerdrop_(pLayerdrop),
      useMask_(useMask),
      preLN_(preLN),
      w1_(std::make_shared<Linear>(transformerInitLinear(modelDim, mlpDim))),
      w2_(std::make_shared<Linear>(transformerInitLinear(mlpDim, modelDim))),
      wq_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wk_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wv_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wf_(std::make_shared<Linear>(
          transformerInitLinear(headDim * nHeads, modelDim))),
      norm1_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}))),
      norm2_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}))) {
  if (bptt > 0) {
    params_.push_back(
        uniform(2 * bptt - 1, headDim, -0.1, 0.1, af::dtype::f32, true));
  }

  add(w1_);
  add(w2_);
  add(wq_);
  add(wk_);
  add(wv_);
  add(wf_);
  add(norm1_);
  add(norm2_);
}

Variable Transformer::mlp(const Variable& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  return (*w2_)(dropout(relu((*w1_)(input)), pDropout));
}

Variable Transformer::getMask(int32_t n, bool cache) {
  auto mask = af::lower(af::constant(1.0, n, n), true);
  if (cache) {
    auto maskCache = af::upper(af::constant(1.0, n, n));
    mask = af::join(1, maskCache, mask);
  }
  return Variable(af::log(mask), false);
}

Variable Transformer::selfAttention(const std::vector<Variable>& input) {
  // previous step[optionally], input, padMask
  auto encoderInput = input.at(input.size() - 2);
  // in case of previous state input[0] has size CxT_prevxB
  int n = input[0].dims(1), bsz = input[0].dims(2);
  double pDrop = train_ ? pDropout_ : 0.0;

  auto q = transpose((*wq_)(encoderInput));
  std::vector<fl::Variable> inputWithState(input.begin(), input.end() - 1);
  auto k = transpose((*wk_)(concatenate(inputWithState, 1)));
  auto v = transpose((*wv_)(concatenate(inputWithState, 1)));

  Variable mask, posEmb;
  if (bptt_ > 0) {
    posEmb =
        tile(params_[0].as(encoderInput.type()), af::dim4(1, 1, nHeads_ * bsz));
  }
  if (useMask_ && encoderInput.dims(1) > 1) {
    // mask future if we use the previous state (then n is previous time)
    mask = getMask(n, input.size() == 3);
  }

  int offset = (input.size() == 2) ? 0 : n;

  // time x batch
  fl::Variable padMask;
  if (!input.back().isempty()) {
    auto padMaskArr = input.back().array();
    padMaskArr =
        af::resize(padMaskArr, encoderInput.dims(1), encoderInput.dims(2));
    padMask = fl::Variable(af::log(padMaskArr), false);
  }
  auto result = multiheadAttention(
      q, k, v, posEmb, mask, padMask, nHeads_, pDrop, offset);
  result = (*wf_)(transpose(result));

  return result;
}

std::vector<Variable> Transformer::forward(const std::vector<Variable>& input) {
  // previous step[optionally], input, padMask
  // padMask should be empty if previous step is provided
  // padMask is expected to have "1" on the used positions and "0" on padded
  // positions
  if (input.size() < 2) {
    throw std::invalid_argument(
        "Invalid inputs for transformer block: there should be at least input and mask");
  }
  auto x = input.at(input.size() - 2);
  if (!input.back().isempty() && x.dims(2) != input.back().dims(1)) {
    throw std::invalid_argument(
        "Invalid inputs for transformer block: input and Mask batch sizes are different");
  }

  // float f = 1.0;
  // if (train_ && (af::randu(1).scalar<float>() < pLayerdrop_)) {
  //   f = 0.0;
  // }
  // if (preLN_) {
  //   auto h = (f * (*norm1_)(selfAttention(input))).as(x.type()) + x;
  //   return {f * (*norm2_)(mlp(h)).as(h.type()) + h};
  // } else {
  //   auto h = (*norm1_)((f * selfAttention(input)).as(x.type()) + x);
  //   return {(*norm2_)((f * mlp(h)).as(h.type()) + h)};
  // }
  if (train_ && (af::randu(1).scalar<float>() < pLayerdrop_)) {
    if (preLN_) {
      return {x};
    } else {
      return {(*norm2_)((*norm1_)(x))};
    }  
  }
  else {
    if (preLN_) {
      auto h = ((*norm1_)(selfAttention(input))).as(x.type()) + x;
      return {(*norm2_)(mlp(h)).as(h.type()) + h};
    } else {
      auto h = (*norm1_)((selfAttention(input)).as(x.type()) + x);
      return {(*norm2_)((mlp(h)).as(h.type()) + h)};
    }
  }
}

// void check(fl::Variable out, std::string message) {
//   if (af::anyTrue<bool>(af::isNaN(out.array()))) {
//     auto p = af::sum<int>(af::flat(af::isNaN(out.array())));
//     std::cerr << getWorldRank() << " Transformer NAN in " << message << " "
//               << out.array().elements() << " | nan " << p << std::endl;
//   }
//   if (af::anyTrue<bool>(af::isInf(out.array()))) {
//     auto m = af::isInf(out.array());
//     auto p = af::sum<int>(af::flat(m));
//     auto pos = af::sum<int>(af::flat(out.array()(m) > 0));
//     std::cerr << getWorldRank() << " Transformer Inf in " << message << " "
//               << out.array().elements() << " | inf " << p << " | "
//               << " pos inf " << pos << " neg inf " << p - pos << std::endl;
//   }
// }

// Variable Transformer::selfAttention(const std::vector<Variable>& input) {
//   // previous step[optionally], input, padMask
//   auto encoderInput = input.at(input.size() - 2);
//   // in case of previous state input[0] has size CxT_prevxB
//   int n = input[0].dims(1), bsz = input[0].dims(2);
//   double pDrop = train_ ? pDropout_ : 0.0;

//   auto q = transpose((*wq_)(encoderInput));
//   std::vector<fl::Variable> inputWithState(input.begin(), input.end() - 1);
//   auto k = transpose((*wk_)(concatenate(inputWithState, 1)));
//   auto v = transpose((*wv_)(concatenate(inputWithState, 1)));
//   check(q, "self-attention q");
//   check(k, "self-attention k");
//   check(v, "self-attention v");

//   Variable mask, posEmb;
//   if (bptt_ > 0) {
//     posEmb =
//         tile(params_[0].as(encoderInput.type()), af::dim4(1, 1, nHeads_ * bsz));
//     check(posEmb, "self-attention pos emb");
//   }
//   if (useMask_ && encoderInput.dims(1) > 1) {
//     // mask future if we use the previous state (then n is previous time)
//     mask = getMask(n, input.size() == 3);
//   }

//   int offset = (input.size() == 2) ? 0 : n;

//   // time x batch
//   fl::Variable padMask;
//   if (!input.back().isempty()) {
//     auto padMaskArr = input.back().array();
//     padMaskArr =
//         af::resize(padMaskArr, encoderInput.dims(1), encoderInput.dims(2));
//     padMask = fl::Variable(af::log(padMaskArr), false);
//     check(padMask, "self-attention pad mask");
//   }
//   auto result = multiheadAttention(
//       q, k, v, posEmb, mask, padMask, nHeads_, pDrop, offset);
//   check(result, "self-attention after multiheadAttention");
//   result = (*wf_)(transpose(result));
//   check(result, "self-attention after multiheadAttention + linear");
//   return result;
// }

// std::vector<Variable> Transformer::forward(const std::vector<Variable>& input) {
//   // previous step[optionally], input, padMask
//   // padMask should be empty if previous step is provided
//   // padMask is expected to have "1" on the used positions and "0" on padded
//   // positions
//   if (input.size() < 2) {
//     throw std::invalid_argument(
//         "Invalid inputs for transformer block: there should be at least input and mask");
//   }
//   auto x = input.at(input.size() - 2);
//   if (!input.back().isempty() && x.dims(2) != input.back().dims(1)) {
//     throw std::invalid_argument(
//         "Invalid inputs for transformer block: input and Mask batch sizes are different");
//   }

//   float f = 1.0;
//   if (train_ && (af::randu(1).scalar<float>() < pLayerdrop_)) {
//     f = 0.0;
//   }
//   if (preLN_) {
//     auto out = selfAttention(input);
//     check(out, "self-attention");
//     out = (*norm1_)(out);
//     check(out, "self-attention + norm");
//     auto h = (f * out).as(x.type()) + x;
//     check(h, "self-attention + norm + drop + residual");
//     out = mlp(h);
//     check(out, "mlp");
//     out = (*norm2_)(out).as(h.type());
//     check(out, "mlp + norm");
//     out = f * out + h;
//     check(out, "mlp + norm + drop + residual");
//     return {out};
//   } else {
//     auto out = (f * selfAttention(input)).as(x.type());
//     check(out, "self-attention + drop");
//     out = out + x;
//     check(out, "self-attention + drop + residual");
//     auto h = (*norm1_)(out);
//     check(h, "self-attention + drop + residual + norm");
//     out = (f * mlp(h)).as(h.type());
//     check(out, "mlp + drop");
//     out = out + h;
//     check(out, "mlp + drop + residual");
//     out = (*norm2_)(out);
//     check(out, "mlp + drop + residual norm");
//     return {out};
//   }
// }


std::string Transformer::prettyString() const {
  std::ostringstream ss;
  ss << "Transformer (nHeads: " << nHeads_ << "), "
     << "(pDropout: " << pDropout_ << "), "
     << "(pLayerdrop: " << pLayerdrop_ << "), "
     << "(bptt: " << bptt_ << "), "
     << "(useMask: " << useMask_ << "), "
     << "(preLayerNorm: " << preLN_ << ")";
  return ss.str();
}

Transformer::Transformer() {}

} // namespace fl
