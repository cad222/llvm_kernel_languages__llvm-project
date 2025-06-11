//===------ LanguageLaunch.cpp - Language (CUDA/HIP) launch api -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LanguageLaunch.h"
#include "ExportedAPI.h"
#include "Types.h"
#include "OffloadAPI.h"

#include <cstdint>
#include <cstdio>
#include <algorithm>

extern "C" {



#define LLVM_STYLE_LAUNCH(SUFFIX, PER_THREAD_STREAM)                           \
  unsigned __llvmLaunchKernel##SUFFIX(const char *KernelID, dim3 GridDim,      \
                                      dim3 BlockDim, void *KernelArgsPtr,      \
                                      size_t DynamicSharedMem, void *Stream) { \
    auto *LOKA = reinterpret_cast<LLVMOffloadKernelArgsTy *>(KernelArgsPtr);   \
    ol_result_t Result =                                                       \
        llvmLaunchKernelImpl(KernelID, GridDim, BlockDim, KernelArgsPtr,       \
                             DynamicSharedMem, Stream, LOKA);                  \
    return Result ? Result->Code : 0;                                          \
  }

LLVM_STYLE_LAUNCH(, false);
LLVM_STYLE_LAUNCH(_spt, true);
LLVM_STYLE_LAUNCH(_ptsz, true);

} // extern "C"
