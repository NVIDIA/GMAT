import ctypes

CSwscale = ctypes.CDLL("../build/CSwscale.so")

SwscaleCuda_Nv12ToRgbpf32_Init = CSwscale.SwscaleCuda_Nv12ToRgbpf32_Init
SwscaleCuda_Nv12ToRgbpf32_Convert = CSwscale.SwscaleCuda_Nv12ToRgbpf32_Convert
SwscaleCuda_Nv12ToRgbpf32_Delete = CSwscale.SwscaleCuda_Nv12ToRgbpf32_Delete

SwscaleCuda_Nv12ToRgbpf32_Init.restype = ctypes.c_void_p

class SwscaleCuda:
    def __init__(self, w, h):
        self.ctx = SwscaleCuda_Nv12ToRgbpf32_Init(ctypes.c_int(w), ctypes.c_int(h))
        self.w = w
        self.h = h
    def __del__(self):
        SwscaleCuda_Nv12ToRgbpf32_Delete(ctypes.c_void_p(self.ctx))

    # in_nv12 and out_rgbp are pointers to CUDA memory
    def nv12_to_rgbpf32(self, in_nv12, in_stride, out_rgbp, out_stride, stream=0):
        return SwscaleCuda_Nv12ToRgbpf32_Convert(ctypes.c_void_p(self.ctx), in_nv12, in_stride,
                                                ctypes.c_ulonglong(out_rgbp), ctypes.c_int(out_stride),
                                                self.w, self.h, ctypes.c_ulonglong(stream))
