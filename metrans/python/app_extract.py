import torch
import torch.cuda as cuda
from frame_extractor import FrameExtractor

import ctypes
libnvToolsExt = ctypes.CDLL('libnvToolsExt.so')
nvtxRangePush = libnvToolsExt.nvtxRangePushA
nvtxRangePop = libnvToolsExt.nvtxRangePop

dev = torch.device("cuda:0")
dummy = torch.empty((1,), device=dev)

file_path = '../build/bunny.mp4'
# extractor = FrameExtractor(file_path, None)
with open(file_path, 'rb') as mp4:
    extractor = FrameExtractor(buffer=mp4.read())
bgr = torch.empty((3, extractor.get_height(), extractor.get_width()), dtype=torch.float32, device=dev)

extractor.set_frame_interval(10)
n = 0
with open('out.bgrp', 'wb') as f, cuda.stream(cuda.Stream(dev)):
    while True:
        nvtxRangePush(('Frame#' + str(n)).encode('utf8'))
        if not extractor.extract_to_device_buffer(bgr.data_ptr(), cuda.current_stream().cuda_stream):
            nvtxRangePop()
            break;
        n += 1
        t = (bgr.cpu() * 255.0).char();
        nvtxRangePop()
        f.write(t.numpy().tobytes())
