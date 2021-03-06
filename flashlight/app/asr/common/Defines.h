/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/lib/text/dictionary/Defines.h"

#define FL_APP_ASR_VERSION "0.1"

namespace fl {
namespace app {
namespace asr {

// Dataset indices
// If a new field is added, `kNumDataIdx` should be modified accordingly.
constexpr size_t kInputIdx = 0;
constexpr size_t kTargetIdx = 1;
constexpr size_t kWordIdx = 2;
constexpr size_t kSampleIdx = 3;
constexpr size_t kPathIdx = 4;
constexpr size_t kDurationIdx = 5;
constexpr size_t kNumDataIdx = 6; // total number of dataset indices

// Various constants used in asr task
constexpr const char* kTrainMode = "train";
constexpr const char* kContinueMode = "continue";
constexpr const char* kForkMode = "fork";
constexpr const char* kGflags = "gflags";
constexpr const char* kCommandLine = "commandline";
constexpr const char* kUserName = "username";
constexpr const char* kHostName = "hostname";
constexpr const char* kTimestamp = "timestamp";
constexpr const char* kRunIdx = "runIdx";
constexpr const char* kRunPath = "runPath";
constexpr const char* kProgramName = "programname";
constexpr const char* kEpoch = "epoch";
constexpr const char* kUpdates = "updates";
constexpr const char* kSGDOptimizer = "sgd";
constexpr const char* kAdamOptimizer = "adam";
constexpr const char* kRMSPropOptimizer = "rmsprop";
constexpr const char* kAdadeltaOptimizer = "adadelta";
constexpr const char* kAdagradOptimizer = "adagrad";
constexpr const char* kAMSgradOptimizer = "amsgrad";
constexpr const char* kNovogradOptimizer = "novograd";
constexpr const char* kCtcCriterion = "ctc";
constexpr const char* kAsgCriterion = "asg";
constexpr const char* kSeq2SeqCriterion = "seq2seq";
constexpr const char* kTransformerCriterion = "transformer";
constexpr int kTargetPadValue = -1;

// Feature params
constexpr int kLifterParam = 22;
constexpr int kPrefetchSize = 2;

constexpr const char* kEosToken = "$";
constexpr const char* kBlankToken = "#";
constexpr const char* kSilToken = "|";

} // namespace asr
} // namespace app
} // namespace fl
