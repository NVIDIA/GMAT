import numpy as np
import torch
import torch.cuda as cuda
from frame_extractor import FrameExtractor
import heif_format
import swscale

import ctypes
libnvToolsExt = ctypes.CDLL('libnvToolsExt.so')
nvtxRangePush = libnvToolsExt.nvtxRangePushA
nvtxRangePop = libnvToolsExt.nvtxRangePop

dev = torch.device("cuda:0")
dummy = torch.empty((1,), device=dev)

file_path = '../build/bunny.mp4'
with open(file_path, 'rb') as mp4:
    extractor = FrameExtractor(buffer=mp4.read())
enc = heif_format.NvEncLite(width=extractor.get_width(), height=extractor.get_height())
dec = heif_format.NvDecLite()
nv12 = torch.empty((1, extractor.get_height() * 3 // 2, extractor.get_width()), dtype=torch.uint8, device=dev)
rgbp = torch.empty((3, extractor.get_height(), extractor.get_width()), dtype=torch.float32, device=dev)
scale = swscale.SwscaleCuda(extractor.get_width(), extractor.get_height())

extractor.set_frame_interval(10)
n = 0
with open('bunny.rgb24', 'wb') as f, cuda.stream(cuda.Stream(dev)):
    while True:
        nvtxRangePush(('Frame#' + str(n)).encode('utf8'))
        if not extractor.extract_to_buffer(nv12.data_ptr(), cuda.current_stream().cuda_stream):
            nvtxRangePop()
            break
        n += 1
        pkt = heif_format.Packet()
        enc.encode_device_frame(nv12.data_ptr(), pkt.v)
        writer = heif_format.NvHeifWriter()
        img, size = writer.write_stillimage(pkt.v)
        reader = heif_format.NvHeifReader(img, size)
        pkt_ref, pkt_size = reader.read_image()

        frame, width, height, linesize = dec.decode_still(pkt_ref, pkt_size)
        scale.nv12_to_rgbpf32(frame, linesize, rgbp.data_ptr(), rgbp.stride(1), cuda.current_stream().cuda_stream)

        rgb24 = (rgbp*255.0).permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
        if n == 1:
            seq = rgb24
        else:
            seq = np.concatenate([seq, rgb24], axis=0)
        
    seq.tofile('bunny_ext_rgb24.rgb')