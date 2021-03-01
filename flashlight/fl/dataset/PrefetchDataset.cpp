/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <stdexcept>

#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/common/Serialization.h"
#include "flashlight/fl/dataset/PrefetchDataset.h"
#include "flashlight/fl/distributed/DistributedApi.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/memory/managers/CachingMemoryManager.h"

namespace fl {
namespace {
void logMemStats(const char* file, int line) {
  if (fl::getWorldRank() == 0) {
    std::stringstream ss;
    ss << file << ":" << line << " logMemStats():" << std::endl;
    auto* curMemMgr =
        fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
    if (curMemMgr) {
      curMemMgr->logStats(&ss);
    }
    std::cerr << ss.str();
  }
}

#define LOG_MEM_STATS() logMemStats(__FILE__, __LINE__)

} // namespace
PrefetchDataset::PrefetchDataset(
    std::shared_ptr<const Dataset> dataset,
    int64_t numThreads,
    int64_t prefetchSize)
    : dataset_(dataset),
      numThreads_(numThreads),
      prefetchSize_(prefetchSize),
      curIdx_(-1) {
  if (!dataset_) {
    throw std::invalid_argument("dataset to be prefetched is null");
  }
  if (!(numThreads_ > 0 && prefetchSize_ > 0) &&
      !(numThreads_ == 0 && prefetchSize_ == 0)) {
    throw std::invalid_argument("invalid numThreads or prefetchSize");
  }
  if (numThreads_ > 0) {
    auto deviceId = af::getDevice();
    auto rank = fl::getWorldRank();
    FL_LOG(fl::ERROR) << "deviceId=" << deviceId << " rank=" << rank
                      << ((deviceId != (rank % 8)) ? "XXXXX" : "oooo");
    LOG_MEM_STATS();
    threadPool_ = std::make_unique<ThreadPool>(
        numThreads_,
        [deviceId](int /* threadId */) { af::setDevice(deviceId); });
  }
}

std::vector<af::array> PrefetchDataset::get(int64_t idx) const {
  checkIndexBounds(idx);

  if (numThreads_ == 0) {
    return dataset_->get(idx);
  }

  // remove from cache (if necessary)
  while (!prefetchCache_.empty() && idx != curIdx_) {
    prefetchCache_.pop();
    ++curIdx_;
  }

  // add to cache (if necessary)
  while (prefetchCache_.size() < prefetchSize_) {
    auto fetchIdx = idx + prefetchCache_.size();
    if (fetchIdx >= size()) {
      break;
    }
    prefetchCache_.emplace(threadPool_->enqueue(
        [this, fetchIdx]() { return this->dataset_->get(fetchIdx); }));
  }

  auto curSample = prefetchCache_.front().get();

  prefetchCache_.pop();
  curIdx_ = idx + 1;
  return curSample;
}

int64_t PrefetchDataset::size() const {
  return dataset_->size();
}
} // namespace fl
