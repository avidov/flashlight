/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/opencl.h>
#include <miopen/miopen.h>
#include <iostream>

#define MIOPEN_CHECK_ERR(expr) \
  ::fl::miopen::detail::check((expr), __FILE__, __LINE__, #expr)

namespace fl {
namespace miopen {

struct ArrayInfo {
    unsigned devId;
    af_dtype type;
    af::dim4 dim_size;
    dim_t offset;
    af::dim4 dim_strides;
    bool is_sparse;
};

struct MiOpenDevicePtr {
    MiOpenDevicePtr(const af::array& in) : arr_(in.get()) {
      if (in.isempty()) {
        std::cerr << "MiOpenDevicePtr in.empty()\n";
        throw std::runtime_error("MiOpenDevicePtr in.empty()");
      }

      ptr_ = in.device<void>();

      const ArrayInfo *info =
          static_cast<ArrayInfo *>(reinterpret_cast<void *>(arr_));
      origInfo_ = *info;


    }

    ~MiOpenDevicePtr() { 
        const ArrayInfo *info =
          static_cast<ArrayInfo *>(reinterpret_cast<void *>(arr_));

      if (info->devId != origInfo_.devId) {
        std::cout << "~MiOpenDevicePtr() cur-info:{devid=" << info->devId 
            << "} orig-info={devid=" << origInfo_.devId << std::endl;
            throw std::runtime_error("~MiOpenDevicePtr() cur-info:{devid=");
      }


      af_unlock_array(arr_); 
    }

    void* get() const { return ptr_; }

    ArrayInfo origInfo_;
    const af_array arr_;
    void* ptr_;
};

const void* kOne(const af::dtype t);
const void* kZero(const af::dtype t);

std::string PrettyString(miopenConvSolution_t algorithm);
std::string PrettyString(miopenConvAlgoPerf_t algorithm);
std::string PrettyString(miopenConvAlgorithm_t algorithm);
std::string PrettyString(miopenConvBwdDataAlgorithm_t algorithm);
std::string PrettyString(miopenConvBwdWeightsAlgorithm_t algorithm);
std::string PrettyString(miopenConvFwdAlgorithm_t algorithm);
std::string PrettyString(miopenStatus_t status);

namespace detail {

void check(miopenStatus_t err, const char* file, int line, const char* cmd);

} // namespace detail
} // namespace miopen
} // namespace fl
