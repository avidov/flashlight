/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/augmentation/TimeStretch.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/fl/common/Logging.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

namespace {

/**
 * Copied from sox/src/stretch.c with minimal necessary changes.
 * https://github.com/chirlu/sox/blob/4927023d0978615c74e8a511ce981cf4c29031f1/src/stretch.c
 *
 * libSoX Basic time stretcher.
 * (c) march/april 2000 Fabien COELHO <fabien@coelho.net> for sox.
 *
 * cross fade samples so as to go slower or faster.
 *
 * The filter is based on 6 parameters:
 * - stretch factor f
 * - window size w
 * - input step i
 *   output step o=f*i
 * - steady state of window s, ss = s*w
 *
 */

typedef enum { input_state, output_state } stretch_status_t;

struct priv_t {
  /* internal stuff */
  stretch_status_t state; /* automaton status */

  size_t segment; /* buffer size */
  size_t index; /* next available element */
  std::vector<float> ibuf; /* input buffer */
  size_t ishift; /* input shift */

  size_t oindex; /* next evailable element */
  std::vector<float> obuf; /* output buffer */
  size_t oshift; /* output shift */

  size_t overlap; /* fading size */
  std::vector<float> fade_coefs; /* fading, 1.0 -> 0.0 */
};

void start(const TimeStretch::Config& conf, float factor, priv_t& p) {
  size_t i = 0;
  p.state = input_state;

  p.segment = (int)(conf.sampleRate_ * 0.001 * conf.window_);
  /* start in the middle of an input to avoid initial fading... */
  p.index = p.segment / 2;
  p.ibuf = std::vector<float>(p.segment);

  /* the shift ratio deal with the longest of ishift/oshift
     hence ishift<=segment and oshift<=segment. */
  if (factor < 1.0) {
    p.ishift = conf.shift_ * p.segment;
    p.oshift = factor * p.ishift;
  } else {
    p.oshift = conf.shift_ * p.segment;
    p.ishift = p.oshift / factor;
  }
  assert(p.ishift <= p.segment);
  assert(p.oshift <= p.segment);

  p.oindex = p.index; /* start as synchronized */
  p.obuf = std::vector<float>(p.segment);
  p.overlap = (int)(conf.fading_ * p.segment);
  p.fade_coefs = std::vector<float>(p.overlap);

  /* initialize buffers */
  for (i = 0; i < p.segment; i++) {
    p.ibuf[i] = 0;
  }

  for (i = 0; i < p.segment; i++)
    p.obuf[i] = 0.0;

  if (p.overlap > 1) {
    double slope = 1.0 / (p.overlap - 1);
    p.fade_coefs[0] = 1.0;
    for (i = 1; i < p.overlap - 1; i++)
      p.fade_coefs[i] = slope * (p.overlap - i - 1);
    p.fade_coefs[p.overlap - 1] = 0.0;
  } else if (p.overlap == 1) {
    p.fade_coefs[0] = 1.0;
  }
}

/* accumulates input ibuf to output obuf with fading fade_coefs */
void combine(priv_t& p) {
  size_t i = 0;

  /* fade in */
  for (i = 0; i < p.overlap; i++) {
    p.obuf[i] += p.fade_coefs[p.overlap - 1 - i] * p.ibuf[i];
  }

  /* steady state */
  for (; i < p.segment - p.overlap; i++) {
    p.obuf[i] += p.ibuf[i];
  }

  /* fade out */
  for (; i < p.segment; i++) {
    p.obuf[i] += p.fade_coefs[i - p.segment + p.overlap] * p.ibuf[i];
  }
}

void flow(
    priv_t& p,
    const std::vector<float>& ibuf,
    std::vector<float>& obuf,
    size_t* isamp,
    size_t* osamp) {
  size_t iindex = 0, oindex = 0;
  size_t i;

  while (iindex < *isamp && oindex < *osamp) {
    if (p.state == input_state) {
      size_t tocopy = std::min(*isamp - iindex, p.segment - p.index);

      std::memcpy(
          p.ibuf.data() + p.index,
          ibuf.data() + iindex,
          tocopy * sizeof(float));

      iindex += tocopy;
      p.index += tocopy;

      if (p.index == p.segment) {
        /* compute */
        combine(p);

        /* shift input */
        for (i = 0; i + p.ishift < p.segment; i++)
          p.ibuf[i] = p.ibuf[i + p.ishift];

        p.index -= p.ishift;

        /* switch to output state */
        p.state = output_state;
      }
    }

    if (p.state == output_state) {
      while (p.oindex < p.oshift && oindex < *osamp) {
        float f;
        f = p.obuf[p.oindex++];
        // SOX_SAMPLE_CLIP_COUNT(f, effp->clips);
        obuf[oindex++] = f;
      }

      if (p.oindex >= p.oshift && oindex < *osamp) {
        p.oindex -= p.oshift;

        /* shift internal output buffer */
        for (i = 0; i + p.oshift < p.segment; i++) {
          p.obuf[i] = p.obuf[i + p.oshift];
        }

        /* pad with 0 */
        for (; i < p.segment; i++) {
          p.obuf[i] = 0.0;
        }

        p.state = input_state;
      }
    }
  }

  *isamp = iindex;
  *osamp = oindex;
}

/*
 * Drain buffer at the end
 * maybe not correct ? end might be artificially faded?
 */
void drain(priv_t& p, std::vector<float>& obuf, size_t* osamp) {
  size_t i = 0;
  size_t oindex = 0;

  if (p.state == input_state) {
    for (i = p.index; i < p.segment; i++)
      p.ibuf[i] = 0;

    combine(p);

    p.state = output_state;
  }

  while (oindex < *osamp && p.oindex < p.index) {
    float f = p.obuf[p.oindex++];
    obuf[oindex++] = f;
  }

  *osamp = oindex;
}

} // namespace

void TimeStretch::apply(std::vector<float>& signal) {
  float factor = rng_.uniform(conf_.factorMin_, conf_.factorMax_);
  if (factor == 1 || rng_.random() >= conf_.proba_) {
    return;
  }

  priv_t p;
  start(conf_, factor, p);
  size_t isamp = signal.size();
  size_t osamp = conf_.leaveLengthUnchanged_ ? isamp : isamp * factor;
  std::vector<float> obuf(osamp);
  flow(p, signal, obuf, &isamp, &osamp);
  drain(p, obuf, &osamp);

  signal.swap(obuf);
}

std::string TimeStretch::Config::prettyString() const {
  std::stringstream ss;
  ss << "TimeStretch::Config{factorMin_=" << factorMin_
     << " factorMax_=" << factorMax_ << " window_=" << window_
     << " shift_=" << shift_ << " fading_=" << fading_ << " proba_=" << proba_
     << " sampleRate_=" << sampleRate_ << " randomSeed_=" << randomSeed_ << '}';
  return ss.str();
}

std::string TimeStretch::prettyString() const {
  std::stringstream ss;
  ss << "TimeStretch{config={" << conf_.prettyString() << '}';
  return ss.str();
};

TimeStretch::TimeStretch(const TimeStretch::Config& config)
    : conf_(config), rng_(config.randomSeed_) {}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
