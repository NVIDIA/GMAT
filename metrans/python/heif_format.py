import ctypes
import numpy as np

CHeif = ctypes.CDLL('../build/CHeif.so')

NvEncLite_InitStill = CHeif.NvEncLite_InitStill
NvEncLite_EncodeDeviceFrame = CHeif.NvEncLite_EncodeDeviceFrame
NvHeifWriter_Init = CHeif.NvHeifWriter_Init
NvHeifWriter_WriteStillImage = CHeif.NvHeifWriter_WriteStillImage
NvHeifReader_Init = CHeif.NvHeifReader_Init
NvDecLite_Init = CHeif.NvDecLite_Init
NvDecLite_DecodeStill = CHeif.NvDecLite_DecodeStill
NvHeifReader_ReadImage = CHeif.NvHeifReader_ReadImage
NvHeifWriter_Delete = CHeif.NvHeifWriter_Delete
NvEncLite_Delete = CHeif.NvEncLite_Delete
NvHeifReader_Delete = CHeif.NvHeifReader_Delete
NvDecLite_Delete = CHeif.NvDecLite_Delete
NvHeifWriter_GetBufferData = CHeif.NvHeifWriter_GetBufferData
NvHeifWriter_GetBufferSize = CHeif.NvHeifWriter_GetBufferSize
NvHeifWriter_WriteToNp =CHeif.NvHeifWriter_WriteToNp
Create_PktVector = CHeif.Create_PktVector
Delete_PktVector = CHeif.Delete_PktVector

# NvEncLite_InitStill.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
NvEncLite_InitStill.restype = ctypes.c_void_p
NvHeifWriter_Init.restype = ctypes.c_void_p
NvHeifWriter_GetBufferData.restype = ctypes.c_void_p
NvHeifWriter_GetBufferSize.restype = ctypes.c_ulonglong
NvHeifReader_Init.restype = ctypes.c_void_p
NvDecLite_Init.restype = ctypes.c_void_p
Create_PktVector.restype = ctypes.c_void_p

class Packet:
    def __init__(self):
        self.v = Create_PktVector()
    def __del__(self):
        Delete_PktVector(ctypes.c_void_p(self.v))

class NvEncLite:
    def __init__(self, width, height, init_param="-codec hevc -preset p1 -bitrate 4M"):
        if width == 0 or height == 0:
            raise ValueError('width and height cannot be 0')
        self.h = height
        self.w = width
        self.enc = NvEncLite_InitStill(width, height)
    
    def __del__(self):
        NvEncLite_Delete(ctypes.c_void_p(self.enc))
    
    def encode_device_frame(self, dpframe, vpkt):
        return NvEncLite_EncodeDeviceFrame(ctypes.c_void_p(self.enc), ctypes.c_ulonglong(dpframe), ctypes.c_void_p(vpkt))

class NvDecLite:
    def __init__(self):
        self.dec = NvDecLite_Init()
    
    def __del__(self):
        NvDecLite_Delete(ctypes.c_void_p(self.dec))

    def decode_still(self, pkt_data, pkt_size):
        frame = ctypes.POINTER(ctypes.c_uint8)()
        width = ctypes.c_int()
        height = ctypes.c_int()
        linesize = ctypes.c_int()
        NvDecLite_DecodeStill(ctypes.c_void_p(self.dec), ctypes.byref(pkt_data), ctypes.c_int(pkt_size), ctypes.byref(frame), ctypes.byref(width), ctypes.byref(height), ctypes.byref(linesize))
        return frame, width, height, linesize

# Add external memory ctor
class NvHeifWriter:
    def __init__(self):
        self.writer = NvHeifWriter_Init()
    def __del__(self):
        NvHeifWriter_Delete(ctypes.c_void_p(self.writer))

    def write_stillimage(self, pkt):
        res = NvHeifWriter_WriteStillImage(ctypes.c_void_p(self.writer), ctypes.c_void_p(pkt))
        # img_buf = NvHeifWriter_GetBufferData(ctypes.c_void_p(self.writer))
        size = NvHeifWriter_GetBufferSize(ctypes.c_void_p(self.writer))
        img_np = np.zeros((size,), dtype=np.uint8)
        NvHeifWriter_WriteToNp(ctypes.c_void_p(self.writer), img_np.ctypes.data_as(ctypes.c_void_p))
        return img_np, size

class NvHeifReader:
    def __init__(self, img, size):
        if img is None:
            raise ValueError('input buffer cannot be empty')
        if size is None:
            raise ValueError('input buffer size cannot be 0')
        buffer = img.ctypes.data_as(ctypes.c_void_p)
        self.reader = NvHeifReader_Init(img.ctypes.data_as(ctypes.c_void_p), ctypes.c_ulonglong(size))
    def __del__(self):
        NvHeifReader_Delete(ctypes.c_void_p(self.reader))
    
    def read_image(self):
        pkt_ref = ctypes.POINTER(ctypes.c_uint8)()
        pkt_size = NvHeifReader_ReadImage(ctypes.c_void_p(self.reader), ctypes.byref(pkt_ref))
        return pkt_ref, pkt_size
