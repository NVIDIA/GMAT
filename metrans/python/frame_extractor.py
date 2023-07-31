import ctypes

libavutil = ctypes.CDLL('libavutil.so')
libavutil.av_log_set_level(24)

CFrameExtractor = ctypes.CDLL('../build/CFrameExtractor.so')

FrameExtractor_InitFromFile = CFrameExtractor.FrameExtractor_InitFromFile
FrameExtractor_InitFromBuffer = CFrameExtractor.FrameExtractor_InitFromBuffer
FrameExtractor_Delete = CFrameExtractor.FrameExtractor_Delete
FrameExtractor_SetFrameInterval = CFrameExtractor.FrameExtractor_SetFrameInterval
FrameExtractor_SetTimeInterval = CFrameExtractor.FrameExtractor_SetTimeInterval
FrameExtractor_GetWidth = CFrameExtractor.FrameExtractor_GetWidth
FrameExtractor_GetHeight = CFrameExtractor.FrameExtractor_GetHeight
FrameExtractor_GetFrameSize = CFrameExtractor.FrameExtractor_GetFrameSize
FrameExtractor_ExtractToDeviceBuffer = CFrameExtractor.FrameExtractor_ExtractToDeviceBuffer
FrameExtractor_ExtractToBuffer = CFrameExtractor.FrameExtractor_ExtractToBuffer

FrameExtractor_InitFromFile.restype = ctypes.c_void_p
FrameExtractor_InitFromBuffer.restype = ctypes.c_void_p

class FrameExtractor:
    def __init__(self, file_path=None, buffer=None):
        if file_path:
            self.h = FrameExtractor_InitFromFile(file_path.encode('utf-8'))
        elif buffer:
            self.buffer = buffer
            self.h = FrameExtractor_InitFromBuffer(buffer, len(buffer))
        else:
            raise ValueError('file_path or buffer is needed')

    def __del__(self):
        print('delete FrameExtractor')
        FrameExtractor_Delete(ctypes.c_ulonglong(self.h))

    def set_frame_interval(self, frame_interval):
        FrameExtractor_SetFrameInterval(ctypes.c_ulonglong(self.h), frame_interval);
    def set_time_interval(self, time_interval):
        FrameExtractor_SetTimeInterval(ctypes.c_ulonglong(self.h), ctypes.c_double(time_interval));

    def get_width(self):
        return FrameExtractor_GetWidth(ctypes.c_ulonglong(self.h))
    def get_height(self):
        return FrameExtractor_GetHeight(ctypes.c_ulonglong(self.h))
    def get_frame_size(self):
        return FrameExtractor_GetFrameSize(ctypes.c_ulonglong(self.h));

    def extract_to_device_buffer(self, dpBgrp, stream=0):
        return FrameExtractor_ExtractToDeviceBuffer(ctypes.c_ulonglong(self.h), ctypes.c_ulonglong(dpBgrp), ctypes.c_ulonglong(stream))
    def extract_to_buffer(self, pframe, stream=0):
        return FrameExtractor_ExtractToBuffer(ctypes.c_ulonglong(self.h), ctypes.c_ulonglong(pframe), ctypes.c_ulonglong(stream))