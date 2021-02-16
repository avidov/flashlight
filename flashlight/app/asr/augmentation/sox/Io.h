/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/app/asr/augmentation/sox/Utils.h"

namespace fl {
namespace sox_io {

// std::tuple<int64_t, int64_t, int64_t, int64_t, std::string> get_info_file(
//     const std::string& path,
//     std::optional<std::string>& format);

// std::tuple<std::vector<float>, int64_t> load_audio_file(
//     const std::string& path,
//     std::optional<int64_t>& frame_offset,
//     std::optional<int64_t>& num_frames,
//     std::optional<bool>& normalize,
//     std::optional<bool>& channels_first,
//     std::optional<std::string>& format);

// void save_audio_file(
//     const std::string& path,
//     std::vector<float> tensor,
//     int64_t sample_rate,
//     bool channels_first,
//     std::optional<double> compression,
//     std::optional<std::string> format,
//     std::optional<std::string> dtype);

// std::tuple<int64_t, int64_t, int64_t, int64_t, std::string> get_info_fileobj(
//     py::object fileobj,
//     std::optional<std::string>& format);

// std::tuple<std::vector<float>, int64_t> load_audio_fileobj(
//     py::object fileobj,
//     std::optional<int64_t>& frame_offset,
//     std::optional<int64_t>& num_frames,
//     std::optional<bool>& normalize,
//     std::optional<bool>& channels_first,
//     std::optional<std::string>& format);

// void save_audio_fileobj(
//     py::object fileobj,
//     std::vector<float> tensor,
//     int64_t sample_rate,
//     bool channels_first,
//     std::optional<double> compression,
//     std::string filetype,
//     std::optional<std::string> dtype);


} // namespace sox_io
} // namespace fl

