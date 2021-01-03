/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <arrayfire.h>
#include <assert.h>
#include <sox.h>
#include <stdio.h>
#include <stdlib.h>

#include "flashlight/app/asr/augmentation/TimeStretch.h"
#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/lib/common/System.h"

using namespace ::fl::app::asr::sfx;
using ::fl::app::asr::loadSound;
using ::fl::app::asr::saveSound;
using ::fl::lib::dirCreateRecursive;
using ::fl::lib::getTmpPath;
using ::fl::lib::pathsConcat;
using ::testing::Pointwise;

const char* inputFilename =
    "/checkpoint/avidov/datasets/audio/LibriSpeech/train-other-500/1353/121397/1353-121397-0055.flac";

namespace {
// Arbitrary audioable signalSoxFmt values.
const int numSamples = 20000;
const size_t freq = 1000;
const size_t sampleRate = 16000;
const float amplitude = 1.0;

std::vector<float>
genSinWave(size_t numSamples, size_t freq, size_t sampleRate, float amplitude) {
  std::vector<float> output(numSamples, 0);
  const float waveLenSamples =
      static_cast<float>(sampleRate) / static_cast<float>(freq);
  const float ratio = (2 * M_PI) / waveLenSamples;

  for (size_t i = 0; i < numSamples; ++i) {
    output.at(i) = amplitude * std::sin(static_cast<float>(i) * ratio);
  }
  return output;
}

} // namespace

static void output_message(
    unsigned level,
    const char* filename,
    const char* fmt,
    va_list ap) {
  char const* const str[] = {"FAIL", "WARN", "INFO", "DBUG"};
  if (sox_globals.verbosity >= level) {
    char base_name[128];
    sox_basename(base_name, sizeof(base_name), filename);
    fprintf(stderr, "%s %s: ", str[std::min((int)level - 1, 3)], base_name);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
  }
}

void set_seed(const int64_t seed) {
  sox_get_globals()->ranqd1 = static_cast<sox_int32_t>(seed);
}

unsigned get_precision(const std::string filetype, const af::dtype dtype) {
  if (filetype == "mp3")
    return SOX_UNSPEC;
  if (filetype == "flac")
    return 24;
  if (filetype == "ogg" || filetype == "vorbis")
    return SOX_UNSPEC;
  if (filetype == "wav" || filetype == "amb") {
    if (dtype == af::dtype::u8)
      return 8;
    if (dtype == af::dtype::s16)
      return 16;
    if (dtype == af::dtype::s32)
      return 32;
    if (dtype == af::dtype::f32)
      return 32;
    throw std::runtime_error("Unsupported dtype.");
  }
  throw std::runtime_error("Unsupported file type: " + filetype);
}

sox_signalinfo_t get_signalinfo(
    const std::vector<float>& signal,
    size_t sampleRate,
    const af::dtype dtype,
    const std::string filetype) {
  return sox_signalinfo_t{/*rate=*/static_cast<sox_rate_t>(sampleRate),
                          /*channels=*/
                          1,
                          /*precision=*/get_precision(filetype, dtype),
                          /*length=*/static_cast<uint64_t>(signal.size())};
}

sox_encoding_t get_encoding(const std::string filetype, af::dtype dtype) {
  if (filetype == "mp3")
    return SOX_ENCODING_MP3;
  if (filetype == "flac")
    return SOX_ENCODING_FLAC;
  if (filetype == "ogg" || filetype == "vorbis")
    return SOX_ENCODING_VORBIS;
  if (filetype == "wav" || filetype == "amb") {
    if (dtype == af::dtype::u8)
      return SOX_ENCODING_UNSIGNED;
    if (dtype == af::dtype::s16)
      return SOX_ENCODING_SIGN2;
    if (dtype == af::dtype::s32)
      return SOX_ENCODING_SIGN2;
    if (dtype == af::dtype::f32)
      return SOX_ENCODING_FLOAT;
    throw std::runtime_error("Unsupported dtype.");
  }
  if (filetype == "sph")
    return SOX_ENCODING_SIGN2;
  if (filetype == "amr-nb")
    return SOX_ENCODING_AMR_NB;
  throw std::runtime_error("Unsupported file type: " + filetype);
}

sox_encodinginfo_t get_encodinginfo(
    const std::string filetype,
    const af::dtype dtype,
    const double compression) {
  const double compression_ = [&]() {
    if (filetype == "mp3")
      return compression;
    if (filetype == "flac")
      return compression;
    if (filetype == "ogg" || filetype == "vorbis")
      return compression;
    if (filetype == "wav" || filetype == "amb")
      return 0.;
    if (filetype == "sph")
      return 0.;
    if (filetype == "amr-nb")
      return 0.;
    throw std::runtime_error("Unsupported file type: " + filetype);
  }();

  return sox_encodinginfo_t{/*encoding=*/get_encoding(filetype, dtype),
                            /*bits_per_sample=*/get_precision(filetype, dtype),
                            /*compression=*/compression_,
                            /*reverse_bytes=*/sox_option_default,
                            /*reverse_nibbles=*/sox_option_default,
                            /*reverse_bits=*/sox_option_default,
                            /*opposite_endian=*/sox_false};
}

/// helper classes for passing the location of input tensor and output buffer
///
/// drain/flow callback functions require plaing C style function signature and
/// the way to pass extra data is to attach data to sox_fffect_t::priv pointer.
/// The following structs will be assigned to sox_fffect_t::priv pointer which
/// gives sox_effect_t an access to input Tensor and output buffer object.
struct TensorInputPriv {
  size_t index;
  std::vector<float> signal;
};
struct TensorOutputPriv {
  std::vector<sox_sample_t>* buffer;
};
struct FileOutputPriv {
  sox_format_t* sf;
};

// /// Callback function to feed Tensor data to SoxEffectChain.
// int tensor_input_drain(sox_effect_t* effp, sox_sample_t* obuf, size_t* osamp)
// {
//   // Retrieve the input Tensor and current index
//   auto priv = static_cast<TensorInputPriv*>(effp->priv);
//   auto index = priv->index;
//   auto signal = priv->signal;
//   auto tensor = signal->getTensor();
//   auto num_channels = effp->out_signal.channels;

//   // Adjust the number of samples to read
//   const size_t num_samples = tensor.numel();
//   if (index + *osamp > num_samples) {
//     *osamp = num_samples - index;
//   }
//   // Ensure that it's a multiple of the number of channels
//   *osamp -= *osamp % num_channels;

//   // Slice the input Tensor and unnormalize the values
//   const auto tensor_ = [&]() {
//     auto i_frame = index / num_channels;
//     auto num_frames = *osamp / num_channels;
//     auto t = (signal->getChannelsFirst())
//         ? tensor.index({Slice(), Slice(i_frame, i_frame + num_frames)}).t()
//         : tensor.index({Slice(i_frame, i_frame + num_frames), Slice()});
//     return unnormalize_wav(t.reshape({-1})).contiguous();
//   }();
//   priv->index += *osamp;

//   // Write data to SoxEffectsChain buffer.
//   auto ptr = tensor_.data_ptr<int32_t>();
//   std::copy(ptr, ptr + *osamp, obuf);

//   return (priv->index == num_samples) ? SOX_EOF : SOX_SUCCESS;
// }

// sox_effect_handler_t* get_tensor_input_handler() {
//   static sox_effect_handler_t handler{/*name=*/"input_tensor",
//                                       /*usage=*/NULL,
//                                       /*flags=*/SOX_EFF_MCHAN,
//                                       /*getopts=*/NULL,
//                                       /*start=*/NULL,
//                                       /*flow=*/NULL,
//                                       /*drain=*/tensor_input_drain,
//                                       /*stop=*/NULL,
//                                       /*kill=*/NULL,
//                                       /*priv_size=*/sizeof(TensorInputPriv)};
//   return &handler;
// }

// void addInputTensor(std::vector<float>& signal) {
//   // in_sig_ = get_signalinfo(signal, "wav");
//   in_sig_ = get_signalinfo(signal, "flac");
//   interm_sig_ = in_sig_;
//   SoxEffect e(sox_create_effect(get_tensor_input_handler()));
//   auto priv = static_cast<TensorInputPriv*>(e->priv);
//   priv->signal = signal;
//   priv->index = 0;
//   if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
//     throw std::runtime_error("Failed to add effect: input_tensor");
//   }
// }

TEST(TimeStretch, LibSox) {
  auto signal = loadSound<float>(inputFilename);
  const std::string tmpDir = getTmpPath("TimeStretchSox");
  dirCreateRecursive(tmpDir);

  float factor = 1;
  float window = 20.0;

  FL_LOG(fl::INFO) << "tmpDir=" << tmpDir;

  sox_signalinfo_t in_signal_info;
  in_signal_info.channels = 1;
  in_signal_info.length = signal.size();
  in_signal_info.precision = get_precision("flac", af::dtype::s16);
  in_signal_info.rate = sampleRate;
  in_signal_info.mult = NULL;

  sox_encodinginfo_t in_encoding_info;
  in_encoding_info.bits_per_sample = 32;
  in_encoding_info.encoding = get_encoding("flac", af::dtype::f16);
  in_encoding_info.reverse_bytes = sox_option_no;
  in_encoding_info.reverse_bits = sox_option_no;
  in_encoding_info.opposite_endian = sox_false;
  in_encoding_info.reverse_nibbles = sox_option_no;

  sox_globals.output_message_handler = output_message;
  sox_globals.verbosity = 4;
  sox_format_t *in = nullptr, *out = nullptr; /* input and output files */
  sox_effects_chain_t* chain;
  sox_effect_t* e;
  char* args[10];

  /* All libSoX applications must start by initialising the SoX library */
  assert(sox_init() == SOX_SUCCESS);

  std::vector<sox_sample_t> signalSoxFmt(signal.size());
  {
    size_t clips = 0;
    SOX_SAMPLE_LOCALS;
    for (int i = 0; i < signalSoxFmt.size(); ++i) {
      signalSoxFmt[i] = SOX_FLOAT_32BIT_TO_SAMPLE(signal[i], clips);
    }
  }

  // assert(
  //     in = sox_open_mem_read(
  //         signalSoxFmt.data(),
  //         signalSoxFmt.size(),
  //         &in_signal_info,
  //         &in_encoding_info,
  //         "raw"));
  assert(in = sox_open_read(inputFilename, NULL, NULL, NULL));
  // auto interm_signal = in->signalSoxFmt;

  std::stringstream ss;
  ss << "factor-" << factor << ".flac";
  const std::string noiseFilePath = pathsConcat(tmpDir, ss.str());
  char* buffer = nullptr;
  size_t buffer_size = 0;
  assert(
      out = sox_open_memstream_write(
          &buffer, &buffer_size, &in_signal_info, NULL, "raw", NULL));

  /* Create an effects chain; some effects need to know about the input
   * or output file encoding so we provide that information here */
  chain = sox_create_effects_chain(&in->encoding, &out->encoding);

  e = sox_create_effect(sox_find_effect("input"));
  args[0] = (char*)in;
  assert(sox_effect_options(e, 1, args) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &in_signal_info, &in->signal) == SOX_SUCCESS);
  free(e);

  // auto factorParam = std::to_string(factor);
  // e = sox_create_effect(sox_find_effect("stretch"));
  // args[0] = (char*)factorParam.c_str();
  // assert(sox_effect_options(e, 1, args) == SOX_SUCCESS);
  // assert(sox_add_effect(chain, e, &interm_signal, &interm_signal) ==
  // SOX_SUCCESS); free(e);

  e = sox_create_effect(sox_find_effect("output"));
  args[0] = (char*)out;
  assert(sox_effect_options(e, 1, args) == SOX_SUCCESS);
  assert(
      sox_add_effect(chain, e, &in_signal_info, &out->signal) == SOX_SUCCESS);
  free(e);

  /* Flow samples through the effects processing chain until EOF is reached */
  sox_flow_effects(chain, NULL, NULL);
  // out->olength = 0; // workaround

  static const size_t maxSamples = 4096;
  sox_sample_t samples[maxSamples];
  std::vector<sox_sample_t> augmentedSoxFmt;
  for (size_t r; 0 != (r = sox_read(out, samples, maxSamples));) {
    FL_LOG(fl::INFO) << "r=" << r;
    for (int i = 0; i < r; i++) {
      augmentedSoxFmt.push_back(samples[i]);
    }
  }

  std::vector<float> augmented(augmentedSoxFmt.size());
  {
    size_t clips = 0;
    SOX_SAMPLE_LOCALS;
    for (int i = 0; i < augmentedSoxFmt.size(); ++i) {
      augmented[i] = SOX_SAMPLE_TO_FLOAT_32BIT(augmentedSoxFmt[i], clips);
    }
  }

  // std::vector<float> augmented(buffer, buffer + buffer_size);
  // FL_LOG(fl::INFO) << "noiseFilePath=" << noiseFilePath;
  // saveSound(
  //     noiseFilePath,
  //     augmented,
  //     sampleRate,
  //     1,
  //     fl::app::asr::SoundFormat::FLAC,
  //     fl::app::asr::SoundSubFormat::PCM_16);

  FL_LOG(fl::INFO) << "signalSoxFmt.size()=" << signalSoxFmt.size();
  FL_LOG(fl::INFO) << "augmented.size()=" << augmented.size();
  for (int i = 0; i < 30; ++i) {
    FL_LOG(fl::INFO)
        << "i, signal,augmented,diff,  signalSoxFmt,diff, augmentedSoxFmt,="
        << i << ", " << signal[i] << ", " << augmented[i] << ", "
        << (signal[i] - augmented[i]) << "  , " << signalSoxFmt[i] << ", "
        << augmentedSoxFmt[i] << ", " << (signalSoxFmt[i] - augmentedSoxFmt[i]);
  }

  /* All done; tidy up: */
  sox_delete_effects_chain(chain);
  sox_close(out);
  sox_close(in);
  free(buffer);
  sox_quit();
}

TEST(TimeStretch, Basic) {
  const std::string tmpDir = getTmpPath("TimeStretchFlashlight");
  FL_LOG(fl::INFO) << "tmpDir=" << tmpDir;

  dirCreateRecursive(tmpDir);

  auto signalSoxFmt = loadSound<float>(inputFilename);

  for (int leaveUnchanged = 0; leaveUnchanged < 2; ++leaveUnchanged) {
    for (float factor = 0.8; factor <= 1.2; factor += 0.1) {
      TimeStretch::Config conf;
      conf.factorMin_ = factor;
      conf.factorMax_ = factor;
      conf.leaveLengthUnchanged_ = (leaveUnchanged != 0);

      TimeStretch sfx(conf);
      auto augmented = signalSoxFmt;
      sfx.apply(augmented);

      std::stringstream ss;
      ss << "factor-" << factor << "unchanged-" << leaveUnchanged << ".flac";
      const std::string noiseFilePath = pathsConcat(tmpDir, ss.str());

      FL_LOG(fl::INFO) << "noiseFilePath=" << noiseFilePath;
      saveSound(
          noiseFilePath,
          augmented,
          sampleRate,
          1,
          fl::app::asr::SoundFormat::FLAC,
          fl::app::asr::SoundSubFormat::PCM_16);

      if (leaveUnchanged) {
        EXPECT_EQ(signalSoxFmt.size(), augmented.size());
      } else {
        EXPECT_FLOAT_EQ(
            static_cast<float>(augmented.size()) / signalSoxFmt.size(), factor);
      }
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
