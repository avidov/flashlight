/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/memory/managers/CachingMemoryManager.h"
#include <arrayfire.h> // Needed for af exception

#include <limits.h>
#include <math.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <flashlight/fl/common/CudaUtils.h>
#include "flashlight/fl/common/Logging.h"

namespace fl {

namespace {

constexpr size_t kMinBlockSize =
    512; // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576; // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer =
    2097152; // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer =
    20971520; // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc =
    10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152; // round up large allocs to 2 MiB

// Environment variables names, specifying number of mega bytes as floats.
constexpr const char* kMemRecyclingSize = "FL_MEM_RECYCLING_SIZE_MB";
constexpr const char* kMemSplitSize = "FL_MEM_SPLIT_SIZE_MB";
constexpr double kMB = static_cast<double>(1UL << 20);

unsigned int log2int(unsigned int val) {
  if (val == 0)
    return UINT_MAX;
  if (val == 1)
    return 0;
  unsigned int ret = 0;
  while (val > 1) {
    val >>= 1;
    ret++;
  }
  return ret;
}

size_t roundSize(size_t size) {
  if (size < kMinBlockSize) {
    return kMinBlockSize;
  } else {
    return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
  }
}

size_t getAllocationSize(size_t size) {
  if (size <= kSmallSize) {
    return kSmallBuffer;
  } else if (size < kMinLargeAlloc) {
    return kLargeBuffer;
  } else {
    return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
  }
}

static bool BlockComparator(
    const CachingMemoryManager::Block* a,
    const CachingMemoryManager::Block* b) {
  if (a->size_ != b->size_) {
    return a->size_ < b->size_;
  }
  return (uintptr_t)a->ptr_ < (uintptr_t)b->ptr_;
}

std::string formatMemory(size_t bytes) {
  const std::vector<std::string> units = {"B", "KiB", "MiB", "GiB", "TiB"};
  size_t unitId =
      bytes == 0 ? 0 : std::floor(std::log(bytes) / std::log(1024.0));
  unitId = std::min(unitId, units.size() - 1);
  std::string bytesStr = std::to_string(bytes / std::pow(1024.0, unitId));
  bytesStr = bytesStr.substr(0, bytesStr.find(".") + 3);
  return bytesStr + " " + units[unitId];
}

std::string formatPercentOf(size_t numerator, size_t denominator) {
  if (numerator == 0) {
    return "0";
  }
  if (denominator == 0) {
    return "100";
  }
  const double precent =
      (1.0 -
       (static_cast<double>(numerator) / static_cast<double>(denominator))) *
      100.0;
  std::stringstream ss;
  // ss << std::setprecision(3) << precent << '%';
  ss << precent;
  return ss.str();
}

/**
 * Returns number of bytes as represented by the named environment variable. The
 * variable is interperested as a float string specifying value in MBs. Returns
 * defaultVal on failure to read the variable or parse its value.
 */
size_t getEnvAsBytesFromFloatMb(const char* name, size_t defaultVal) {
  const char* env = std::getenv(name);
  if (env) {
    try {
      const double mb = std::stod(env);
      return std::round(mb * kMB);
    } catch (std::exception& ex) {
      FL_LOG(fl::ERROR) << "Invalid environment variable=" << name
                        << " value=" << env;
    }
  }
  return defaultVal;
}

} // namespace

CachingMemoryManager::DeviceMemoryInfo::DeviceMemoryInfo(int id)
    : deviceId_(id),
      largeBlocks_(BlockComparator),
      smallBlocks_(BlockComparator) {}

CachingMemoryManager::CachingMemoryManager(
    int numDevices,
    std::shared_ptr<MemoryManagerDeviceInterface> deviceInterface)
    : MemoryManagerAdapter(deviceInterface) {
  recyclingSizeLimit_ =
      getEnvAsBytesFromFloatMb(kMemRecyclingSize, recyclingSizeLimit_);
  splitSizeLimit_ = getEnvAsBytesFromFloatMb(kMemSplitSize, splitSizeLimit_);

  FL_LOG(fl::INFO) << "CachingMemoryManager recyclingSizeLimit_="
                   << recyclingSizeLimit_ << " ("
                   << formatMemory(recyclingSizeLimit_)
                   << ") splitSizeLimit_=" << splitSizeLimit_ << " ("
                   << formatMemory(splitSizeLimit_) << ")";

  for (int i = 0; i < numDevices; ++i) {
    deviceMemInfos_.emplace(
        i, std::make_unique<CachingMemoryManager::DeviceMemoryInfo>(i));
  }
}

void CachingMemoryManager::initialize() {}

void CachingMemoryManager::setRecyclingSizeLimit(size_t limit) {
  recyclingSizeLimit_ = limit;
}

void CachingMemoryManager::setSplitSizeLimit(size_t limit) {
  splitSizeLimit_ = limit;
}

void CachingMemoryManager::shutdown() {
  signalMemoryCleanup();
}

void CachingMemoryManager::addMemoryManagement(int device) {
  if (deviceMemInfos_.find(device) != deviceMemInfos_.end()) {
    return;
  }
  deviceMemInfos_.emplace(
      device, std::make_unique<CachingMemoryManager::DeviceMemoryInfo>(device));
}

void CachingMemoryManager::removeMemoryManagement(int device) {
  if (deviceMemInfos_.find(device) == deviceMemInfos_.end()) {
    return;
  }
  deviceMemInfos_.erase(device);
}

void* CachingMemoryManager::alloc(
    bool userLock,
    const unsigned ndims,
    dim_t* dims,
    const unsigned elementSize) {
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);
  size_t useSize = elementSize;
  for (unsigned i = 0; i < ndims; ++i) {
    useSize *= dims[i];
  }
  if (useSize == 0) {
    return nullptr;
  }
  size_t size = roundSize(useSize);
  const bool isSmallAlloc = (size <= kSmallSize);
  CachingMemoryManager::Block searchKey(size);
  CachingMemoryManager::BlockSet& pool =
      isSmallAlloc ? memoryInfo.smallBlocks_ : memoryInfo.largeBlocks_;

  CachingMemoryManager::Block* block = nullptr;
  auto it = pool.lower_bound(&searchKey);
  // Recycle blocks if any found, and if small alloc or the block size is not
  // too large:
  if (it != pool.end() &&
      (isSmallAlloc || (*it)->size_ < recyclingSizeLimit_)) {
    block = *it;
    pool.erase(it);
    memoryInfo.stats_.cachedBytes_ -= block->size_;
  } else {
    void* ptr = nullptr;
    size_t allocSize = getAllocationSize(size);
    mallocWithRetry(allocSize, &ptr); // could throw
    block = new Block(allocSize, ptr);
    memoryInfo.stats_.allocatedBytes_ += allocSize;
  }

  // If the block is larger than the requested size to handle another
  // allocation in the same large or small BlockSet, it will be split into two.
  // Note that we don't split a small stepsize out of a large one to keep the
  // implementation simple.
  CachingMemoryManager::Block* remaining = nullptr;
  size_t diff = block->size_ - size;
  if ((diff >= (isSmallAlloc ? kMinBlockSize : kSmallSize)) &&
      (block->size_ < splitSizeLimit_) // possibly dont split large buffers to
                                       // minimize risk of fragmentation
  ) {
    remaining = block;
    block = new Block(size, block->ptr_);
    block->useSize_ = remaining->useSize_;
    remaining->useSize_ = 0;
    block->prev_ = remaining->prev_;
    if (block->prev_) {
      block->prev_->next_ = block;
    }
    block->next_ = remaining;

    remaining->prev_ = block;
    remaining->ptr_ = static_cast<char*>(remaining->ptr_) + size;
    remaining->size_ -= size;
    remaining->useSize_ = 0;
    pool.insert(remaining);
    memoryInfo.stats_.cachedBytes_ += remaining->size_;
  }

  block->managerLock_ = !userLock;
  block->userLock_ = userLock;
  block->useSize_ = useSize;
  memoryInfo.stats_.useAllocatedBytes_ += useSize;
  const size_t nBits = log2int(useSize);
  ++memoryInfo.stats_.totalUseAllocatedBytesHist_[nBits];
  ++memoryInfo.stats_.curUseAllocatedBytesHist_[nBits];

  memoryInfo.allocatedBlocks_[block->ptr_] = block;
  return static_cast<void*>(block->ptr_);
}

size_t CachingMemoryManager::allocated(void* ptr) {
  if (!ptr) {
    return 0;
  }
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);
  auto it = memoryInfo.allocatedBlocks_.find(ptr);
  if (it == memoryInfo.allocatedBlocks_.end()) {
    return 0;
  }
  return (it->second)->size_;
}

void CachingMemoryManager::unlock(void* ptr, bool userUnlock) {
  if (!ptr) {
    return;
  }
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);
  auto it = memoryInfo.allocatedBlocks_.find(ptr);
  if (it == memoryInfo.allocatedBlocks_.end()) {
    // Probably came from user, just free it
    this->deviceInterface->nativeFree(ptr);
    ++memoryInfo.stats_.totalNativeFrees_;
    memoryInfo.stats_.nativeAllocated_.erase(ptr);
    return;
  }

  CachingMemoryManager::Block* block = it->second;
  if (userUnlock) {
    block->userLock_ = false;
  } else {
    block->managerLock_ = false;
  }

  // Return early if either one is locked
  if (block->inUse()) {
    return;
  }
  memoryInfo.allocatedBlocks_.erase(it);
  freeBlock(block);
}

void CachingMemoryManager::freeBlock(CachingMemoryManager::Block* block) {
  if (block->inUse()) {
    throw std::runtime_error("trying to free a block which is in use");
  }
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);

  const bool isSmallAlloc = (block->size_ <= kSmallSize);
  CachingMemoryManager::BlockSet& pool =
      isSmallAlloc ? memoryInfo.smallBlocks_ : memoryInfo.largeBlocks_;
  tryMergeBlocks(block, block->prev_, pool);
  tryMergeBlocks(block, block->next_, pool);

  pool.insert(block);
  memoryInfo.stats_.cachedBytes_ += block->size_;
  memoryInfo.stats_.useAllocatedBytes_ -= block->useSize_;
  --memoryInfo.stats_.curUseAllocatedBytesHist_[log2int(block->useSize_)];
}

/** combine previously split blocks */
void CachingMemoryManager::tryMergeBlocks(
    CachingMemoryManager::Block* dst,
    CachingMemoryManager::Block* src,
    BlockSet& pool) {
  if (!src || src->inUse()) {
    return;
  }
  if (dst->prev_ == src) {
    dst->ptr_ = src->ptr_;
    dst->prev_ = src->prev_;
    if (dst->prev_) {
      dst->prev_->next_ = dst;
    }
  } else {
    dst->next_ = src->next_;
    if (dst->next_) {
      dst->next_->prev_ = dst;
    }
  }
  dst->size_ += src->size_;
  pool.erase(src);
  getDeviceMemoryInfo().stats_.cachedBytes_ -= src->size_;
  delete src;
}

void CachingMemoryManager::mallocWithRetry(size_t size, void** ptr) {
  // Try nativeMalloc. If nativeMalloc fails, frees all non-split cached blocks
  // and retries.
  auto& memInfo = getDeviceMemoryInfo();
  try {
    ++memInfo.stats_.totalNativeMallocs_;
    *ptr = this->deviceInterface->nativeAlloc(size);
  } catch (std::exception& exUnused) {
    try {
      signalMemoryCleanup();
      ++memInfo.stats_.totalNativeMallocs_;
      *ptr = this->deviceInterface->nativeAlloc(size);
    } catch (std::exception& ex) {
      std::stringstream ss;
      // note: af exception inherits from std exception
      ss << "Failed to allocate memory of size " << size << "("
         << formatMemory(size) << ") (Device: " << memInfo.deviceId_
         << ", Capacity: "
         << formatMemory(
                this->deviceInterface->getMaxMemorySize(memInfo.deviceId_))
         << ", Allocated: " << formatMemory(memInfo.stats_.allocatedBytes_)
         << ", Cached: " << formatMemory(memInfo.stats_.cachedBytes_)
         << ") with error '" << ex.what() << "'" << std::endl;
      logStats(&ss);
      std::cerr << ss.str() << std::endl;
      // note: converting here an af exception to std exception prevents to
      // catch the af error code at the user level. Rethrowing.
      throw;
    }
  }
  memInfo.stats_.nativeAllocated_[*ptr] = size;
}

void CachingMemoryManager::freeBlocks(
    BlockSet& blocks,
    BlockSet::iterator it,
    BlockSet::iterator end) {
  // Frees all non-split blocks between `it` and `end`
  auto& memoryInfo = getDeviceMemoryInfo();
  while (it != end) {
    Block* block = *it;
    if (!block->isSplit()) {
      this->deviceInterface->nativeFree(static_cast<void*>(block->ptr_));
      ++memoryInfo.stats_.totalNativeFrees_;
      memoryInfo.stats_.nativeAllocated_.erase(block->ptr_);

      memoryInfo.stats_.allocatedBytes_ -= block->size_;
      memoryInfo.stats_.cachedBytes_ -= block->size_;
      auto cur = it;
      ++it;
      blocks.erase(cur);
      delete block;
    } else {
      ++it;
    }
  }
}

void CachingMemoryManager::signalMemoryCleanup() {
  // Free all non-split cached blocks on device
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);

  if (isMaster()) {
    printInfo("Before signalMemoryCleanup()");
  }

  freeBlocks(
      memoryInfo.largeBlocks_,
      memoryInfo.largeBlocks_.begin(),
      memoryInfo.largeBlocks_.end());

  freeBlocks(
      memoryInfo.smallBlocks_,
      memoryInfo.smallBlocks_.begin(),
      memoryInfo.smallBlocks_.end());

  if (isMaster()) {
    printInfo("After signalMemoryCleanup()");
  }
}

float CachingMemoryManager::getMemoryPressure() {
  return 0.0; // TODO: check if this is optimal
}

bool CachingMemoryManager::jitTreeExceedsMemoryPressure(size_t /* unused */) {
  return false; // TODO: check if this is optimal
}

void CachingMemoryManager::printInfo(const char* msg, const int /* unused */) {
  auto& memInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memInfo.mutexAll_);
  std::cout << msg;
  std::cout << "\nType: CachingMemoryManager";
  std::cout << "\nDevice: " << memInfo.deviceId_ << ", Capacity: "
            << formatMemory(
                   this->deviceInterface->getMaxMemorySize(memInfo.deviceId_))
            << ", Allocated: " << formatMemory(memInfo.stats_.allocatedBytes_)
            << ", Cached: " << formatMemory(memInfo.stats_.cachedBytes_);
  std::cout << "\nTotal native calls: " << memInfo.stats_.totalNativeMallocs_
            << "(mallocs), " << memInfo.stats_.totalNativeFrees_ << "(frees)"
            << std::endl;
}

namespace {

// Subtract the given block from given memory map. This splits a block in the
// exiting map.
void subtractBlock(
    std::map<void*, size_t>& nativeMemMap,
    CachingMemoryManager::Block* block,
    size_t mask) {
  void* blockPtr = (void*)(mask & (size_t)block->ptr_);

  auto itr = nativeMemMap.lower_bound(blockPtr);
  if (itr != nativeMemMap.end()) {
    --itr;
    // If block happen to start right at the begining of a block at the memory
    // map.
    if (blockPtr == itr->first) {
      long leftOverSize = itr->second - block->size_;
      size_t newPtr = (size_t)blockPtr + block->size_;
      nativeMemMap[(void*)newPtr] = leftOverSize;
      nativeMemMap.erase(itr);
    } else {
      size_t origSize = itr->second;
      long leftOverSize =
          (long)itr->first + itr->second - (long)blockPtr - block->size_;
      itr->second = (size_t)blockPtr - (size_t)itr->first;
      if (leftOverSize > 0) {
        size_t newPTr = (size_t)blockPtr + block->size_;
        nativeMemMap[(void*)newPTr] = leftOverSize;
      }
    }
  } else {
    FL_LOG(fl::INFO) << "block_ [" << blockPtr << ','
                     << formatMemory(block->size_)
                     << "] not found in memory map";
  }
}

void addMemSizeStat(std::ostream* sink, const std::string name, size_t size) {
  *sink << name << ':' << size << ':' << formatMemory(size) << std::endl
}

} // namespace

void CachingMemoryManager::logStats(std::ostream* sink /*=&std::cout*/) {
  auto& memInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memInfo.mutexAll_);

  auto afDeviceId = af::getDevice();

  size_t largestContiguousCache = 0;
  if (memInfo.stats_.gpuMemSize_ == 0) {
    const size_t capacity =
        this->deviceInterface->getMaxMemorySize(memInfo.deviceId_);
    memInfo.stats_.gpuMemSize_ = capacity;
    // Mask off bits left of capacity.
    memInfo.stats_.gpuMemMask_ =
        ((1UL << (static_cast<size_t>(std::log2(capacity)) + 1)) - 1);
  }

  std::map<void*, size_t> nativeMemMap;
  bool refreshNativeMemMap = false;
  if (memInfo.stats_.totalNativeMallocs_ !=
      memInfo.stats_.totalNativeMallocsRecentLogging_) {
    memInfo.stats_.totalNativeMallocsRecentLogging_ =
        memInfo.stats_.totalNativeMallocs_;
    memInfo.stats_.largestContiguousNative_ = 0;
    refreshNativeMemMap = true;

    // Init native memory map with a single block the size of the entire native
    // memory.
    nativeMemMap[0] = memInfo.stats_.gpuMemSize_;
    nativeMemMap[(void*)(memInfo.stats_.gpuMemSize_ + 1)] = 0;
  }
  for (auto& ptAndBlock : memInfo.allocatedBlocks_) {
    if (refreshNativeMemMap) {
      subtractBlock(
          nativeMemMap, ptAndBlock.second, memInfo.stats_.gpuMemMask_);
    }
  }
  for (auto block : memInfo.largeBlocks_) {
    largestContiguousCache = std::max(largestContiguousCache, block->size_);
    if (refreshNativeMemMap) {
      subtractBlock(nativeMemMap, block, memInfo.stats_.gpuMemMask_);
    }
  }
  if (refreshNativeMemMap) {
    for (auto block : memInfo.smallBlocks_) {
      subtractBlock(nativeMemMap, block, memInfo.stats_.gpuMemMask_);
    }
    for (auto& cuBlock : nativeMemMap) {
      memInfo.stats_.largestContiguousNative_ =
          std::max(memInfo.stats_.largestContiguousNative_, cuBlock.second);
    }
  }

  std::stringstream ss;
  size_t free_byte = 0;
  size_t total_byte = 0;
  auto cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
  if (cudaSuccess != cuda_status) {
    ss << "Error: cudaMemGetInfo fails wih error="
       << cudaGetErrorString(cuda_status);
  } else {
    auto used_db = total_byte - free_byte;
    ss << "GPU memory usage: used=" << used_db << " (" << formatMemory(used_db)
       << ") free=" << free_byte << "(" << formatMemory(free_byte)
       << ") total=" << total_byte << "(" << formatMemory(total_byte) << ")";
  }
  ss << std::endl;

  const size_t internalFragMem =
      (memInfo.stats_.allocatedBytes_ - memInfo.stats_.cachedBytes_ -
       memInfo.stats_.useAllocatedBytes_);
  const size_t largestContiguous =
      std::max(memInfo.stats_.largestContiguousNative_, largestContiguousCache);
  addMemSizeStat(sink, "GpuMemSize", memInfo.stats_.gpuMemSize_);
  addMemSizeStat(sink, "Allocated", memInfo.stats_.allocatedBytes_);
  addMemSizeStat(sink, "Used", memInfo.stats_.useAllocatedBytes_);
  addMemSizeStat(sink, "Cached", memInfo.stats_.cachedBytes_);
  addMemSizeStat(sink, "LargestContiguousCache", largestContiguousCache);
  addMemSizeStat(sink, "LargestContiguousNative", largestContiguousNative);
  addMemSizeStat(sink, "LargestContiguous", largestContiguous);
  *sink << "Internal Fragmentation: "
        << formatPercentOf(
               memInfo.stats_.useAllocatedBytes_,
               memInfo.stats_.allocatedBytes_ - memInfo.stats_.cachedBytes_)
        << std::endl
        << "internalFragMem: " << (internalFragMem) << ":"
        << formatMemory(internalFragMem) << std::endl
        << formatMemory(largestContiguous) << std::endl
        << "NativeMallocCount:" << memInfo.stats_.totalNativeMallocs_
        << std::endl
        << "NativeFreeCout:" << memInfo.stats_.totalNativeFrees_ << std::endl
        << "Device:" << memInfo.deviceId_ << std::endl
        << "Rank: " << fl::getWorldRank() << std::endl;

  {
    *sink << "TotalUseAllocatedBytesHist:";
    for (int i = 0; i < kMaxAllocSize2Pwr; ++i) {
      *sink << memInfo.stats_.totalUseAllocatedBytesHist_[i];
      if (i < kMaxAllocSize2Pwr - 1) {
        *sink << ':';
      }
    }
    *sink << std::endl;
  }

  {
    *sink << "CurUseAllocatedBytesHist: ";
    for (int i = 0; i < kMaxAllocSize2Pwr; ++i) {
      *sink << memInfo.stats_.curUseAllocatedBytesHist_[i];
      if (i < kMaxAllocSize2Pwr - 1) {
        *sink << ':';
      }
    }
    *sink << std::endl << std::endl;
  }
}

void CachingMemoryManager::userLock(const void* ptr) {
  if (!ptr) {
    return;
  }
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);

  auto it = memoryInfo.allocatedBlocks_.find(const_cast<void*>(ptr));
  if (it == memoryInfo.allocatedBlocks_.end()) {
    // Follows the behavior of DefaultMemoryManager
    auto block = new Block(kSmallBuffer, const_cast<void*>(ptr));
    block->managerLock_ = false;
    block->userLock_ = true;
    memoryInfo.allocatedBlocks_[block->ptr_] = block;
  } else {
    it->second->userLock_ = true;
  }
}

void CachingMemoryManager::userUnlock(const void* ptr) {
  this->unlock(const_cast<void*>(ptr), true);
}

bool CachingMemoryManager::isUserLocked(const void* ptr) {
  if (!ptr) {
    return false;
  }
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);
  auto it = memoryInfo.allocatedBlocks_.find(const_cast<void*>(ptr));
  if (it == memoryInfo.allocatedBlocks_.end()) {
    return false;
  }
  return it->second->userLock_;
}

CachingMemoryManager::DeviceMemoryInfo&
CachingMemoryManager::getDeviceMemoryInfo(int device /* = -1*/) {
  if (device == -1) {
    device = this->deviceInterface->getActiveDeviceId();
  }
  auto it = deviceMemInfos_.find(device);
  if (it == deviceMemInfos_.end() || !it->second) {
    throw std::runtime_error("meminfo for the device doesn't exist");
  }
  return *(it->second);
}
} // namespace fl
