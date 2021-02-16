/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sox.h>
#include "flashlight/app/asr/augmentation/sox/Utils.h"

namespace fl {
namespace sox_effects_chain {

// Helper struct to safely close sox_effects_chain_t with handy methods
class SoxEffectsChain {
  const sox_encodinginfo_t in_enc_;
  const sox_encodinginfo_t out_enc_;
  sox_signalinfo_t in_sig_;
  sox_signalinfo_t interm_sig_;
  sox_signalinfo_t out_sig_;
  sox_effects_chain_t* sec_;

 public:
  explicit SoxEffectsChain(
      sox_encodinginfo_t input_encoding,
      sox_encodinginfo_t output_encoding);
  SoxEffectsChain(const SoxEffectsChain& other) = delete;
  SoxEffectsChain(const SoxEffectsChain&& other) = delete;
  SoxEffectsChain& operator=(const SoxEffectsChain& other) = delete;
  SoxEffectsChain& operator=(SoxEffectsChain&& other) = delete;
  ~SoxEffectsChain();
  void run();
  void addInputTensor(
      std::vector<float>* waveform,
      int64_t sample_rate,
      bool channels_first);
  void addInputFile(sox_format_t* sf);
  void addOutputBuffer(std::vector<sox_sample_t>* output_buffer);
  void addOutputFile(sox_format_t* sf);
  void addEffect(const std::vector<std::string> effect);
  int64_t getOutputNumChannels();
  int64_t getOutputSampleRate();

//   void addInputFileObj(
//       sox_format_t* sf,
//       char* buffer,
//       unsigned long long buffer_size,
//       py::object* fileobj);

//   void addOutputFileObj(
//       sox_format_t* sf,
//       char** buffer,
//       size_t* buffer_size,
//       py::object* fileobj);
};

} // namespace sox_effects_chain
} // namespace fl
