/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <arrayfire.h>
#include <sox.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory>

#include <glog/logging.h>

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

#define check(x)                                 \
  {                                              \
    if (!(x)) {                                  \
      fprintf(stderr, "check failed: %s\n", #x); \
      throw std::runtime_error(#x);              \
    }                                            \
  }

const char* inputFilename =
    "/checkpoint/avidov/datasets/audio/LibriSpeech/train-other-500/1353/121397/1353-121397-0055.flac";

static sox_format_t *in, *out; /* input and output files */

/* The function that will be called to input samples into the effects chain.
 * In this example, we get samples to process from a SoX-openned audio file.
 * In a different application, they might be generated or come from a different
 * part of the application. */
static int input_drain(sox_effect_t* effp, sox_sample_t* obuf, size_t* osamp) {
  (void)effp; /* This parameter is not needed in this example */

  /* ensure that *osamp is a multiple of the number of channels. */
  *osamp -= *osamp % effp->out_signal.channels;

  /* Read up to *osamp samples into obuf; store the actual number read
   * back to *osamp */
  *osamp = sox_read(in, obuf, *osamp);

  /* sox_read may return a number that is less than was requested; only if
   * 0 samples is returned does it indicate that end-of-file has been reached
   * or an error has occurred */
  if (!*osamp && in->sox_errno)
    fprintf(stderr, "%s: %s\n", in->filename, in->sox_errstr);
  return *osamp ? SOX_SUCCESS : SOX_EOF;
}

/* The function that will be called to output samples from the effects chain.
 * In this example, we store the samples in a SoX-opened audio file.
 * In a different application, they might perhaps be analysed in some way,
 * or displayed as a wave-form */
static int output_flow(
    sox_effect_t* effp LSX_UNUSED,
    sox_sample_t const* ibuf,
    sox_sample_t* obuf LSX_UNUSED,
    size_t* isamp,
    size_t* osamp) {
  /* Write out *isamp samples */
  size_t len = sox_write(out, ibuf, *isamp);

  /* len is the number of samples that were actually written out; if this is
   * different to *isamp, then something has gone wrong--most often, it's
   * out of disc space */
  if (len != *isamp) {
    fprintf(stderr, "%s: %s\n", out->filename, out->sox_errstr);
    return SOX_EOF;
  }

  /* Outputting is the last `effect' in the effect chain so always passes
   * 0 samples on to the next effect (as there isn't one!) */
  *osamp = 0;

  (void)effp; /* This parameter is not needed in this example */

  return SOX_SUCCESS; /* All samples output successfully */
}

// struct SoxInputHandler {
//   SoxInputHandler(const std::vector<float>& data)
//       : data_(data),
//         handler_{"input",
//                  NULL,
//                  SOX_EFF_MCHAN,
//                  NULL,
//                  NULL,
//                  NULL,
//                  inputDrain,
//                  NULL,
//                  NULL,
//                  0} {}

//   const std::vector<float>& data_;
//   size_t dataIdx_ = 0;
//   sox_effect_handler_t handler_;
// };

struct InputData {
  std::vector<float>* data;
  size_t index = 0;
};

int inputDrain(sox_effect_t* effp, sox_sample_t* obuf, size_t* osamp) {
  auto h = (InputData*)effp->priv;
  LOG(INFO) << "h=" << h << " h->data=" << h->data;

  int i = 0;
  for (; i < *osamp && h->index < h->data->size(); ++i, ++h->index) {
    SOX_SAMPLE_LOCALS;
    obuf[i] = SOX_FLOAT_32BIT_TO_SAMPLE(h->data->at(h->index), effp->clips);
  }
  *osamp = i;
  return *osamp ? SOX_SUCCESS : SOX_EOF;
}

static sox_effect_handler_t const* input_handler(void) {
  static sox_effect_handler_t handler = {"input",
                                         NULL,
                                         SOX_EFF_MCHAN,
                                         NULL,
                                         NULL,
                                         NULL,
                                         inputDrain,
                                         NULL,
                                         NULL,
                                         sizeof(InputData)};
  return &handler;
}

/* A `stub' effect handler to handle outputting samples from the effects
 * chain; the only function needed for this example is `flow' */
static sox_effect_handler_t const* output_handler(void) {
  static sox_effect_handler_t handler = {"output",
                                         NULL,
                                         SOX_EFF_MCHAN,
                                         NULL,
                                         NULL,
                                         output_flow,
                                         NULL,
                                         NULL,
                                         NULL,
                                         0};
  return &handler;
}

TEST(TimeStretch, LibSoxExample1) {
  const char* outputFilename = "/tmp/LibSoxExample1.flac";

  sox_effects_chain_t* chain;
  sox_effect_t* e;
  const char* vol[] = {"3dB"};
  char* args[10];

  /* All libSoX applications must start by initialising the SoX library */
  check(sox_init() == SOX_SUCCESS);

  /* Open the input file (with default parameters) */
  // check(in = sox_open_read(inputFilename, NULL, NULL, NULL));
  auto data = loadSound<float>(inputFilename);

  sox_encodinginfo_t encoding = {SOX_ENCODING_FLAC,
                                 16,
                                 0, // inf
                                 sox_option_no,
                                 sox_option_no,
                                 sox_option_no,
                                 sox_false};
  sox_signalinfo_t signal = {sampleRate, 1, 16, data.size(), NULL};

  /* Open the output file; we must specify the output signal characteristics.
   * Since we are using only simple effects, they are the same as the input
   * file characteristics */
  check(out = sox_open_write(outputFilename, &signal, NULL, NULL, NULL, NULL));

  /* Create an effects chain; some effects need to know about the input
   * or output file encoding so we provide that information here */
  chain = sox_create_effects_chain(&encoding, &encoding);

  /* The first effect in the effect chain must be something that can source
   * samples; in this case, we have defined an input handler that inputs
   * data from an audio file */
  e = sox_create_effect(input_handler());
  // https://github.com/chirlu/sox/blob/dd8b63bdc2966c931b73d5f7a17db336cbec6c21/src/effects.c#L72
  auto effectData = (InputData*)e->priv;
  LOG(INFO) << "effectData=" << effectData << " &data=" << &data;
  effectData->data = &data;
  effectData->index = 0;
  /* This becomes the first `effect' in the chain */
  check(sox_add_effect(chain, e, &signal, &signal) == SOX_SUCCESS);

  float factor = 2.5;
  auto factorParam = std::to_string(factor);
  e = sox_create_effect(sox_find_effect("stretch"));
  args[0] = (char*)factorParam.c_str();
  check(sox_effect_options(e, 1, args) == SOX_SUCCESS);
  check(sox_add_effect(chain, e, &signal, &signal) == SOX_SUCCESS);
  free(e);

  /* Create the `vol' effect, and initialise it with the desired parameters: */
  e = sox_create_effect(sox_find_effect("vol"));
  check(sox_effect_options(e, 1, (char* const*)vol) == SOX_SUCCESS);
  /* Add the effect to the end of the effects processing chain: */
  check(sox_add_effect(chain, e, &signal, &signal) == SOX_SUCCESS);
  free(e);

  /* Create the `flanger' effect, and initialise it with default parameters: */
  e = sox_create_effect(sox_find_effect("flanger"));
  check(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
  /* Add the effect to the end of the effects processing chain: */
  check(sox_add_effect(chain, e, &signal, &signal) == SOX_SUCCESS);
  free(e);

  /* The last effect in the effect chain must be something that only consumes
   * samples; in this case, we have defined an output handler that outputs
   * data to an audio file */
  e = sox_create_effect(output_handler());
  check(sox_add_effect(chain, e, &signal, &signal) == SOX_SUCCESS);
  free(e);

  /* Flow samples through the effects processing chain until EOF is reached */
  sox_flow_effects(chain, NULL, NULL);

  /* All done; tidy up: */
  sox_delete_effects_chain(chain);
  sox_close(out);
  sox_quit();
}

constexpr size_t MAX_SAMPLES = 2048;

struct Sox {
  Sox() {
    check(sox_init() == SOX_SUCCESS);
  }

  ~Sox() {
    sox_quit();
  }

  struct Buffer {
    Buffer(sox_rate_t sampleRate, const std::vector<float>& data)
        : signal{sampleRate, 1, 16, data.size(), NULL} {
      sox_format_t* out = nullptr;
      sox_sample_t samples[MAX_SAMPLES]; /* Temporary store whilst copying. */

      check(
          out = sox_open_memstream_write(
              &buffer, &buffer_size, &signal, NULL, "sox", NULL));
      // while ((number_read = sox_read(in, samples, MAX_SAMPLES))) {

      int j = 0;
      std::stringstream ss2;
      std::stringstream ss3;
      sox_uint64_t clips = 0;

      for (int i = 0; i < data.size(); i++) {
        SOX_SAMPLE_LOCALS;
        samples[0] = SOX_FLOAT_32BIT_TO_SAMPLE(data[i], clips);
        if (j < 100) {
          ++j;
          ss2 << samples[0] << ", ";
          ss3 << data[i] << ", ";
        }
        check(sox_write(out, samples, 1) == 1);
      }
      sox_close(out);
    }

    ~Buffer() {
      free(buffer);
    }

    char* buffer = nullptr;
    size_t buffer_size = 0;

    sox_encodinginfo_t encoding = {SOX_ENCODING_FLAC,
                                   16,
                                   0, // inf
                                   sox_option_no,
                                   sox_option_no,
                                   sox_option_no,
                                   sox_false};
    sox_signalinfo_t signal;
  };

  std::unique_ptr<Sox::Buffer> toSoxBuffer(
      size_t sampleRate,
      const std::vector<float>& signal) {
    return std::make_unique<Sox::Buffer>((sox_rate_t)sampleRate, signal);
  }
};

TEST(TimeStretch, LibSoxLoadSound2) {
  const char* outputFilename = "/tmp/LibSoxLoadSound2.flac";
  auto signal = loadSound<float>(inputFilename);

  Sox sox;
  auto soxBuf = sox.toSoxBuffer(sampleRate, signal);
  sox_format_t* in = nullptr;
  sox_format_t* out = nullptr;

  check(
      in = sox_open_mem_read(
          soxBuf->buffer, soxBuf->buffer_size, NULL, NULL, NULL));
  check(
      out = sox_open_write(
          outputFilename,
          &soxBuf->signal,
          &soxBuf->encoding,
          NULL,
          NULL,
          NULL));
  sox_sample_t samples[MAX_SAMPLES];
  int number_read = 0;
  while ((number_read = sox_read(in, samples, MAX_SAMPLES))) {
    std::stringstream ss;
    for (int i = 0; i < 10; ++i) {
      ss << samples[i] << ", ";
    }
    LOG(INFO) << " number_read=" << number_read << " samples=" << ss.str();
    check(sox_write(out, samples, number_read) == number_read);
  }
  sox_close(out);
  sox_close(in);
}

TEST(TimeStretch, LibSoxLoadSound3) {
  const char* outputFilename = "/tmp/LibSoxLoadSound3.flac";
  auto signal = loadSound<float>(inputFilename);
  char* args[10];

  Sox sox;
  auto soxBuf = sox.toSoxBuffer(sampleRate, signal);
  sox_format_t* in = nullptr;
  sox_format_t* out = nullptr;

  sox_effects_chain_t* chain;
  sox_effect_t* e;
  const char* vol[] = {"3dB"};

  /* Open the output file; we must specify the output signal characteristics.
   * Since we are using only simple effects, they are the same as the input
   * file characteristics */
  check(
      out = sox_open_write(
          outputFilename,
          &soxBuf->signal,
          &soxBuf->encoding,
          NULL,
          NULL,
          NULL));

  /* Create an effects chain; some effects need to know about the input
   * or output file encoding so we provide that information here */
  chain = sox_create_effects_chain(&soxBuf->encoding, &soxBuf->encoding);

  check(
      in = sox_open_mem_read(
          soxBuf->buffer, soxBuf->buffer_size, NULL, NULL, NULL));

  e = sox_create_effect(sox_find_effect("input"));
  args[0] = (char*)in;
  assert(sox_effect_options(e, 1, args) == SOX_SUCCESS);
  /* This becomes the first `effect' in the chain */
  assert(sox_add_effect(chain, e, &in->signal, &in->signal) == SOX_SUCCESS);
  free(e);

  /* Create the `vol' effect, and initialise it with the desired parameters: */
  e = sox_create_effect(sox_find_effect("vol"));
  check(sox_effect_options(e, 1, (char* const*)vol) == SOX_SUCCESS);
  /* Add the effect to the end of the effects processing chain: */
  check(
      sox_add_effect(chain, e, &soxBuf->signal, &soxBuf->signal) ==
      SOX_SUCCESS);
  free(e);

  /* Create the `flanger' effect, and initialise it with default parameters: */
  e = sox_create_effect(sox_find_effect("flanger"));
  check(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
  /* Add the effect to the end of the effects processing chain: */
  check(
      sox_add_effect(chain, e, &soxBuf->signal, &soxBuf->signal) ==
      SOX_SUCCESS);
  free(e);

  /* The last effect in the effect chain must be something that only consumes
   * samples; in this case, we have defined an output handler that outputs
   * data to an audio file */
  e = sox_create_effect(output_handler());
  check(
      sox_add_effect(chain, e, &soxBuf->signal, &soxBuf->signal) ==
      SOX_SUCCESS);
  free(e);

  /* Flow samples through the effects processing chain until EOF is reached */
  sox_flow_effects(chain, NULL, NULL);

  /* All done; tidy up: */
  sox_delete_effects_chain(chain);
  sox_close(out);
  sox_close(in);
  sox_quit();
}

TEST(TimeStretch, LibSoxLoadSound) {
  const char* outputFilename = "/tmp/LibSoxLoadSound.flac";
  auto signal = loadSound<float>(inputFilename);

  sox_format_t *in, *out; /* input and output files */
#define MAX_SAMPLES (size_t)2048
  sox_sample_t samples[MAX_SAMPLES]; /* Temporary store whilst copying. */
  char* buffer = nullptr;
  size_t buffer_size = 0;
  size_t number_read = 0;
  size_t sum_read = 0;

  sox_encodinginfo_t out_encoding = {SOX_ENCODING_FLAC,
                                     16,
                                     0, // inf
                                     sox_option_no,
                                     sox_option_no,
                                     sox_option_no,
                                     sox_false};
  sox_signalinfo_t out_signal = {16000, 1, 16, signal.size(), NULL};

  /* All libSoX applications must start by initialising the SoX library */
  check(sox_init() == SOX_SUCCESS);

  /* Open the input file (with default parameters) */
  // check(in = sox_open_read(inputFilename, NULL, NULL, NULL));
  check(
      out = sox_open_memstream_write(
          &buffer, &buffer_size, &out_signal, NULL, "sox", NULL));
  // while ((number_read = sox_read(in, samples, MAX_SAMPLES))) {

  int j = 0;
  std::stringstream ss2;
  std::stringstream ss3;
  sox_uint64_t clips = 0;

  for (int i = 0; i < signal.size(); i++) {
    SOX_SAMPLE_LOCALS;
    samples[0] = SOX_FLOAT_32BIT_TO_SAMPLE(signal[i], clips);
    if (j < 100) {
      ++j;
      ss2 << samples[0] << ", ";
      ss3 << signal[i] << ", ";
    }
    check(sox_write(out, samples, 1) == 1);
  }
  sox_close(out);

  LOG(INFO) << "buffer=" << buffer << " buffer_size=" << buffer_size
            << " number_read=" << number_read << " samples=" << ss2.str()
            << "\nsignal=" << ss3.str();

  check(in = sox_open_mem_read(buffer, buffer_size, NULL, NULL, NULL));
  check(
      out = sox_open_write(
          outputFilename, &out_signal, &out_encoding, NULL, NULL, NULL));
  while ((number_read = sox_read(in, samples, MAX_SAMPLES))) {
    std::stringstream ss;
    for (int i = 0; i < 10; ++i) {
      ss << samples[i] << ", ";
    }
    LOG(INFO) << " number_read=" << number_read << " samples=" << ss.str();
    check(sox_write(out, samples, number_read) == number_read);
  }
  sox_close(out);
  sox_close(in);
  free(buffer);

  sox_quit();
}

TEST(TimeStretch, LibSoxNewMem) {
  const char* outputFilename = "/tmp/LibSoxNewMem.flac";
  auto signal = loadSound<float>(inputFilename);

  static sox_format_t *in, *out; /* input and output files */
#define MAX_SAMPLES (size_t)2048
  sox_sample_t samples[MAX_SAMPLES]; /* Temporary store whilst copying. */
  char* buffer = nullptr;
  size_t buffer_size = 0;
  size_t number_read = 0;
  size_t sum_read = 0;

  sox_sample_t samples2[MAX_SAMPLES]; /* Temporary store whilst copying. */

  /* All libSoX applications must start by initialising the SoX library */
  check(sox_init() == SOX_SUCCESS);

  /* Open the input file (with default parameters) */
  check(in = sox_open_read(inputFilename, NULL, NULL, NULL));
  check(
      out = sox_open_memstream_write(
          &buffer, &buffer_size, &in->signal, NULL, "sox", NULL));
  while ((number_read = sox_read(in, samples, MAX_SAMPLES))) {
    sum_read += number_read;

    static bool first = true;
    if (first) {
      first = false;
      for (int i = 0; i < number_read; ++i) {
        samples2[i] = samples[i];
      }
    }
    check(sox_write(out, samples, number_read) == number_read);
  }
  std::cout << std::endl;
  sox_close(out);
  sox_close(in);

  // LOG(INFO) << "buffer=" << (void*)buffer << " buffer_size=" << buffer_size
  //           << " number_read=" << number_read << " sum_read=" << sum_read;

  // for (int i = 0; i < MAX_SAMPLES; ++i) {
  //   SOX_SAMPLE_LOCALS;
  //   /* convert the sample from SoX's internal format to a `double' for
  //    * processing in this application: */
  //   double sample = SOX_SAMPLE_TO_FLOAT_64BIT(samples2[i], );
  //   double sampleBuf = SOX_SAMPLE_TO_FLOAT_64BIT(((double*)buffer)[i], );

  //   std::cout << "sample, sampleBuf, ==, signal, ==" << sample << ", "
  //             << sampleBuf << ", " << (sample == sampleBuf) << " ," <<
  //             signal[i]
  //             << " ," << (sample == signal[i]) << std::endl;
  // }

  LOG(INFO) << "buffer=" << buffer << " buffer_size=" << buffer_size
            << " number_read=" << number_read;

  check(in = sox_open_mem_read(buffer, buffer_size, NULL, NULL, NULL));
  check(
      out =
          sox_open_write(outputFilename, &in->signal, NULL, NULL, NULL, NULL));
  while ((number_read = sox_read(in, samples, MAX_SAMPLES))) {
    check(sox_write(out, samples, number_read) == number_read);
  }
  sox_close(out);
  sox_close(in);
  free(buffer);

  sox_quit();
}

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
/// drain/flow callback functions require plaing C style function signature
/// and the way to pass extra data is to attach data to sox_fffect_t::priv
/// pointer. The following structs will be assigned to sox_fffect_t::priv
/// pointer which gives sox_effect_t an access to input Tensor and output
/// buffer object.
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
// int tensor_input_drain(sox_effect_t* effp, sox_sample_t* obuf, size_t*
// osamp)
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

// template <typename T>
// struct optional {
//   std::unique_ptr<T> value;
// };

// std::tuple<int64_t, int64_t, int64_t, int64_t, std::string> get_info_file(
//     const std::string& path,
//     std::optional<std::string>& format) {
//   SoxFormat sf(sox_open_read(
//       path.c_str(),
//       /*signal=*/nullptr,
//       /*encoding=*/nullptr,
//       /*filetype=*/format.has_value() ? format.value().c_str() : nullptr));

//   if (static_cast<sox_format_t*>(sf) == nullptr) {
//     throw std::runtime_error("Error opening audio file");
//   }

//   return std::make_tuple(
//       static_cast<int64_t>(sf->signal.rate),
//       static_cast<int64_t>(sf->signal.length / sf->signal.channels),
//       static_cast<int64_t>(sf->signal.channels),
//       static_cast<int64_t>(sf->encoding.bits_per_sample),
//       get_encoding(sf->encoding.encoding));
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
  in_signal_info.precision = 32; // get_precision("flac", af::dtype::s16);
  in_signal_info.rate = sampleRate;
  in_signal_info.mult = NULL;

  sox_encodinginfo_t in_encoding_info;
  in_encoding_info.bits_per_sample = 32;
  in_encoding_info.encoding =
      SOX_ENCODING_FLOAT; // get_encoding("flac", af::dtype::f16);
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
  check(sox_init() == SOX_SUCCESS);

  std::vector<sox_sample_t> signalSoxFmt(signal.size());
  {
    size_t clips = 0;
    SOX_SAMPLE_LOCALS;
    for (int i = 0; i < signalSoxFmt.size(); ++i) {
      signalSoxFmt[i] = SOX_FLOAT_32BIT_TO_SAMPLE(signal[i], clips);
    }
  }

  check(
      in = sox_open_mem_read(
          signalSoxFmt.data(),
          signalSoxFmt.size(),
          &in_signal_info,
          &in_encoding_info,
          "raw"));
  // check(in = sox_open_read(inputFilename, NULL, NULL, NULL));
  // auto interm_signal = in->signalSoxFmt;

  std::stringstream ss;
  ss << "factor-" << factor << ".flac";
  const std::string noiseFilePath = pathsConcat(tmpDir, ss.str());
  char* buffer = nullptr;
  size_t buffer_size = 0;
  check(
      out = sox_open_memstream_write(
          &buffer, &buffer_size, &in_signal_info, NULL, "raw", NULL));

  // /* Create an effects chain; some effects need to know about the input
  //  * or output file encoding so we provide that information here */
  // chain = sox_create_effects_chain(&in->encoding, &out->encoding);

  // e = sox_create_effect(sox_find_effect("input"));
  // args[0] = (char*)in;
  // check(sox_effect_options(e, 1, args) == SOX_SUCCESS);
  // check(sox_add_effect(chain, e, &in_signal_info, &in->signal) ==
  // SOX_SUCCESS); free(e);

  // // auto factorParam = std::to_string(factor);
  // // e = sox_create_effect(sox_find_effect("stretch"));
  // // args[0] = (char*)factorParam.c_str();
  // // check(sox_effect_options(e, 1, args) == SOX_SUCCESS);
  // // check(sox_add_effect(chain, e, &interm_signal, &interm_signal) ==
  // // SOX_SUCCESS); free(e);

  // e = sox_create_effect(sox_find_effect("output"));
  // args[0] = (char*)out;
  // check(sox_effect_options(e, 1, args) == SOX_SUCCESS);
  // check(sox_add_effect(chain, e, &in_signal_info, &out->signal) ==
  // SOX_SUCCESS); free(e);

  // /* Flow samples through the effects processing chain until EOF is reached
  // */ sox_flow_effects(chain, NULL, NULL);
  // // out->olength = 0; // workaround

  static const size_t maxSamples = 2048;
  sox_sample_t samples[maxSamples];
  size_t number_read;
  while ((number_read = sox_read(in, samples, MAX_SAMPLES)))
    check(sox_write(out, samples, number_read) == number_read);
  sox_close(out);
  sox_close(in);

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
  // sox_delete_effects_chain(chain);
  // sox_close(out);
  // sox_close(in);
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
  google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
