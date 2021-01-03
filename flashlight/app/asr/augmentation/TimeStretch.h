/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/app/asr/augmentation/SoundEffect.h"

#include <random>
#include <string>
#include <vector>

#include "flashlight/app/asr/augmentation/SoundEffectUtil.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

class TimeStretch : public SoundEffect {
 public:
  struct Config {
    /**
     * probability of applying reverb.
     */
    float proba_ = 1.0;
    double factorMin_ = 0.8; /* stretch factor. 1.0 means copy. */
    double factorMax_ = 1.25; /* stretch factor. 1.0 means copy. */
    double window_ = 20.0; /* window in ms */
    double shift_ = 0.8; /* shift ratio wrt window. <1.0 */
    double fading_ = 0.25; /* fading ratio wrt window. <0.5 */
    bool leaveLengthUnchanged_ = true;
    size_t sampleRate_ = 16000;
    unsigned int randomSeed_ = std::mt19937::default_seed;
    std::string prettyString() const;
  };

  explicit TimeStretch(const TimeStretch::Config& config);
  ~TimeStretch() override = default;
  void apply(std::vector<float>& signal) override;
  std::string prettyString() const override;

 private:
  const TimeStretch::Config conf_;
  RandomNumberGenerator rng_;
  std::unique_ptr<ListRandomizer<std::string>> listRandomizer_;
};

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
