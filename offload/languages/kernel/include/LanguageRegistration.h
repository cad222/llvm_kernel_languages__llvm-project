//===---- LanguageRegistration.h - Language (CUDA/HIP) registration api ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LANGUAGE_REGISTRATION_H
#define LLVM_LANGUAGE_REGISTRATION_H

#include <cstddef>
#include <cstdint>

// Forward declarations for types from OffloadAPI.h
struct uint3;
struct dim3;
struct surfaceReference;
struct textureReference;

#ifdef __cplusplus
extern "C" {
#endif

/// Registers a kernel function with the offloading runtime.
void llvmRegisterFunction(const char *Binary, const char *KernelID,
                          char *KernelName, const char *KernelName1, int,
                          uint3 *, uint3 *, dim3 *, dim3 *, int *);

/// Registers a fat binary with the offloading runtime.
const char *llvmRegisterFatBinary(const char *Binary);

/// Unregisters a previously registered fat binary.
void llvmUnregisterFatBinary(void *Handle);

/// Registers a global variable with the offloading runtime.
void llvmRegisterVar(void **, char *HostVar, char *DeviceVar,
                     const char *Name, int, int, int, int);

/// Registers a managed variable with the offloading runtime.
void llvmRegisterManagedVar(void **, char *HostVar, char *DeviceVar,
                            const char *Name, size_t Size, unsigned Flags);

/// Registers a surface reference.
void llvmRegisterSurface(void **, const struct surfaceReference *,
                         const void **, const char *Name, int, int);

/// Registers a texture reference.
void llvmRegisterTexture(void **, const struct textureReference *,
                         const void **, const char *Name, int, int, int);

#ifdef __cplusplus
}
#endif

#endif // LLVM_LANGUAGE_REGISTRATION_H

