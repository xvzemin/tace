// Copyright (c) 2026 Zemin Xu. SPDX-License-Identifier: MIT
// A small, model-independent CUDA compiler and launcher.
#include <cuda.h>
#include <nvrtc.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

void check(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(status, &message);
        throw std::runtime_error(message ? message : "CUDA driver error");
    }
}

py::bytes compile(const std::string& source, const std::vector<std::string>& options) {
    std::string binary;
    {
        py::gil_scoped_release release;
        nvrtcProgram program;
        auto status = nvrtcCreateProgram(&program, source.c_str(), "eqx.cu", 0, nullptr, nullptr);
        if (status != NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(status));
        std::vector<const char*> flags;
        for (const auto& option : options) flags.push_back(option.c_str());
        status = nvrtcCompileProgram(program, flags.size(), flags.data());
        if (status != NVRTC_SUCCESS) {
            size_t size;
            nvrtcGetProgramLogSize(program, &size);
            std::string log(size, '\0');
            nvrtcGetProgramLog(program, log.data());
            nvrtcDestroyProgram(&program);
            throw std::runtime_error("EQX CUDA compilation failed:\n" + log);
        }
        size_t size;
        status = nvrtcGetCUBINSize(program, &size);
        if (status == NVRTC_SUCCESS) {
            binary.resize(size);
            status = nvrtcGetCUBIN(program, binary.data());
        }
        nvrtcDestroyProgram(&program);
        if (status != NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(status));
    }
    return py::bytes(binary);
}

struct Kernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUcontext context = nullptr;
    int registers = 0;
    int local_bytes = 0;

    Kernel(py::bytes code, const std::string& name) {
        std::string binary = code;
        check(cuCtxGetCurrent(&context));
        check(cuModuleLoadData(&module, binary.data()));
        try {
            check(cuModuleGetFunction(&function, module, name.c_str()));
            check(cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function));
            check(cuFuncGetAttribute(&local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function));
        } catch (...) {
            cuModuleUnload(module);
            throw;
        }
    }

    ~Kernel() {
        if (module && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            cuModuleUnload(module);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
    }

    void launch(std::vector<uint64_t> arguments, unsigned int gx, unsigned int gy,
                unsigned int threads, uint64_t stream) {
        if (!gx || !gy) return;
        std::vector<void*> pointers;
        for (auto& value : arguments) pointers.push_back(&value);
        CUcontext current;
        check(cuCtxGetCurrent(&current));
        const bool change = current != context;
        if (change) check(cuCtxPushCurrent(context));
        const auto status = cuLaunchKernel(function, gx, gy, 1, threads, 1, 1, 0,
                            reinterpret_cast<CUstream>(stream), pointers.data(), nullptr);
        if (change) cuCtxPopCurrent(&current);
        check(status);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compile", &compile);
    m.def("version", []() {
        int major, minor;
        nvrtcVersion(&major, &minor);
        return py::make_tuple(major, minor);
    });
    py::class_<Kernel, std::shared_ptr<Kernel>>(m, "Kernel")
        .def(py::init<py::bytes, const std::string&>())
        .def_readonly("registers", &Kernel::registers)
        .def_readonly("local_bytes", &Kernel::local_bytes)
        .def("launch", &Kernel::launch);
}
