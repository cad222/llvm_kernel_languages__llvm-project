//===---- LanguageRegistration.cpp - Language (CUDA/HIP) registration api -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "LanguageRegistration.h"
#include "ExportedAPI.h"
#include "OffloadAPI.h"

#include <cstdint>
#include <iterator>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#define HIP_FATBIN_MAGIC_STR "__CLANG_OFFLOAD_BUNDLE__"
constexpr auto HIP_FATBIN_MAGIC_STR_LEN = sizeof(HIP_FATBIN_MAGIC_STR) - 1;

namespace {
struct FatbinWrapperTy {
  int Magic;
  int Version;
  const char *Data;
  const char *DataEnd;
};

template <typename T> T readAndAdvance(const char *&Ptr) {
  auto V = *reinterpret_cast<const T *>(Ptr);
  std::advance(Ptr, sizeof(T));
  return V;
}

void readTUFatbin(const char *Binary, const FatbinWrapperTy *FW) {
  ol_device_handle_t Device = olKGetDefaultDevice();
  size_t Size = FW->DataEnd - FW->Data;
  printf("%p : %p :: %zu \n", FW->Data, FW->DataEnd, Size);
  ol_program_handle_t Program = nullptr;
  ol_result_t Result = olCreateProgram(Device, FW->Data, Size, &Program);
  if (Result && Result->Code) {
    fprintf(stderr, "Failed to register device code (%i): %s\n", Result->Code,
            Result->Details);
    abort();
  }
  printf("Program :: %p\n", Program);
  olKRegisterProgram(Binary, Program);
}

void readHIPFatbinEntries(const char *Binary, const char *HIPFatbinPtr) {
  ol_device_handle_t Device = olKGetDefaultDevice();
  const char *DataIt = HIPFatbinPtr;
  std::advance(DataIt, HIP_FATBIN_MAGIC_STR_LEN);

  uint64_t NumBundles = readAndAdvance<uint64_t>(DataIt);
  for (uint64_t BundleId = 0; BundleId < NumBundles; ++BundleId) {
    uint64_t BundleOffset = readAndAdvance<uint64_t>(DataIt);
    uint64_t BundleSize = readAndAdvance<uint64_t>(DataIt);
    uint64_t BundleTripleSize = readAndAdvance<uint64_t>(DataIt);
    std::advance(DataIt, BundleTripleSize);

    if (!BundleSize)
      continue;

    ol_program_handle_t Program = nullptr;
    ol_result_t Result = olCreateProgram(Device, HIPFatbinPtr + BundleOffset,
                                         BundleSize, &Program);
    if (Result && Result->Code) {
      fprintf(stderr, "Failed to register device code (%i): %s\n", Result->Code,
              Result->Details);
      abort();
    }
    olKRegisterProgram(Binary, Program);
  }
}
} // namespace

extern "C" {

void llvmRegisterFunction(const char *Binary, const char *KernelID,
                          char *KernelName, const char *KernelName1, int,
                          uint3 *, uint3 *, dim3 *, dim3 *, int *) {
  ol_kernel_handle_t Kernel;
  ol_program_handle_t Program = olKGetProgram(Binary);
  ol_result_t Result = olGetKernel(Program, KernelName, &Kernel);
  if (Result && Result->Code) {
    fprintf(stderr, "Failed to register kernel (%i): %s\n", Result->Code,
            Result->Details);
    abort();
  }
  olKRegisterKernel(KernelID, Kernel);
}

const char *llvmRegisterFatBinary(const char *Binary) {
  const auto *FW = reinterpret_cast<const FatbinWrapperTy *>(Binary);
  if (FW->Magic == 0x466243b1) {
    readTUFatbin(Binary, FW);
  } else if (FW->Magic == 0x48495046) {
    if (!memcmp(FW->Data, HIP_FATBIN_MAGIC_STR, HIP_FATBIN_MAGIC_STR_LEN))
      readHIPFatbinEntries(Binary, FW->Data);
    else
      readTUFatbin(Binary, FW);
  } else {
    fprintf(stderr, "Unknown fatbin format");
  }
  return Binary;
}

void llvmUnregisterFatBinary(void *Handle) {
  // No-op
}

void llvmRegisterVar(void **, char *, char *, const char *, int, int, int, int) {
  fprintf(stderr, "RegisterVar is not implemented!\n");
}

void llvmRegisterManagedVar(void **, char *, char *, const char *, size_t,
                            unsigned) {
  fprintf(stderr, "RegisterManagedVar is not implemented!\n");
}

void llvmRegisterSurface(void **, const struct surfaceReference *,
                         const void **, const char *, int, int) {
  fprintf(stderr, "RegisterSurface is not implemented!\n");
}

void llvmRegisterTexture(void **, const struct textureReference *,
                         const void **, const char *, int, int, int) {
  fprintf(stderr, "RegisterTexture is not implemented!\n");
}

} // extern "C"

