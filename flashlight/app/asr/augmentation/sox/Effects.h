/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/app/asr/augmentation/sox/Utils.h"

namespace fl {
namespace sox_effects {

void initialize_sox_effects();

void shutdown_sox_effects();

// std::tuple<std::vector<float>, int64_t> apply_effects_tensor(
//     std::vector<float> waveform,
//     int64_t sample_rate,
//     std::vector<std::vector<std::string>> effects,
//     bool channels_first);

// std::tuple<std::vector<float>, int64_t> apply_effects_file(
//     const std::string path,
//     std::vector<std::vector<std::string>> effects,
//     std::optional<bool>& normalize,
//     std::optional<bool>& channels_first,
//     std::optional<std::string>& format);

// std::tuple<std::vector<float>, int64_t> apply_effects_fileobj(
//     py::object fileobj,
//     std::vector<std::vector<std::string>> effects,
//     std::optional<bool>& normalize,
//     std::optional<bool>& channels_first,
//     std::optional<std::string>& format);

} // namespace sox_effects
} // namespace fl
