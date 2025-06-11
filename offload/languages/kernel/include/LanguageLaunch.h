//===------ LanguageLaunch.h - Language (CUDA/HIP) launch api ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOAD_LANGUAGELAUNCH_H
#define LLVM_OFFLOAD_LANGUAGELAUNCH_H

#include "Types.h"
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

struct LLVMOffloadKernelArgsTy {
  size_t Size;
  void *Args;
  void *_;
};

unsigned llvmPushCallConfiguration(dim3 __grid_size, dim3 __block_size,
                                   size_t __shared_memory, void *__stream);

unsigned llvmPopCallConfiguration(dim3 *__grid_size, dim3 *__block_size,
                                  size_t *__shared_memory, void *__stream);

ol_result_t llvmLaunchKernelImpl(const char *KernelID, dim3 GridDim,
                                 dim3 BlockDim, void *KernelArgsPtr,
                                 size_t DynamicSharedMem, void *Stream,
                                 LLVMOffloadKernelArgsTy *LOKA);

unsigned __llvmLaunchKernel(const char *KernelID, dim3 GridDim, dim3 BlockDim,
                            void *KernelArgsPtr, size_t DynamicSharedMem,
                            void *Stream);
unsigned __llvmLaunchKernel_spt(const char *KernelID, dim3 GridDim, dim3 BlockDim,
                                void *KernelArgsPtr, size_t DynamicSharedMem,
                                void *Stream);
unsigned __llvmLaunchKernel_ptsz(const char *KernelID, dim3 GridDim, dim3 BlockDim,
                                 void *KernelArgsPtr, size_t DynamicSharedMem,
                                 void *Stream);

#ifdef __cplusplus
}
#endif

#endif // LLVM_OFFLOAD_LANGUAGELAUNCH_H
