import torch
import torch.cuda as cuda
import threading
from functools import reduce
import time
import sys
import ctypes
import pdb

from frame_extractor import FrameExtractor
import heif_format
import swscale

cudaFree = ctypes.CDLL('libcudart.so').cudaFree
cudaSetDevice = ctypes.CDLL('libcudart.so').cudaSetDevice

dev = torch.device("cuda:0")
# initialize cuda runtime
dummy = torch.empty((1,), device=dev)

def extract_heif_proc(file_path, l_n_frame):
    # cudaFree(0)
    cudaSetDevice(ctypes.c_int(0))

    with open(file_path, 'rb') as mp4:
        extractor = FrameExtractor(buffer=mp4.read())
    enc = heif_format.NvEncLite(width=extractor.get_width(), height=extractor.get_height())
    dec = heif_format.NvDecLite()
    nv12 = torch.empty((2, extractor.get_height(), extractor.get_width()), dtype=torch.uint8, device=dev)
    rgbp = torch.empty((3, extractor.get_height(), extractor.get_width()), dtype=torch.float32, device=dev)
    scale = swscale.SwscaleCuda(extractor.get_width(), extractor.get_height())
    
    n_frame = 0
    with cuda.stream(cuda.Stream(dev)):
        while extractor.extract_to_buffer(nv12.data_ptr(), cuda.current_stream().cuda_stream):
            n_frame += 1
            pkt = heif_format.Packet()
            enc.encode_device_frame(nv12.data_ptr(), pkt.v)
            writer = heif_format.NvHeifWriter()
            img, size = writer.write_stillimage(pkt.v)
            reader = heif_format.NvHeifReader(img, size)
            pkt_ref, pkt_size = reader.read_image()
            frame, width, height, linesize = dec.decode_still(pkt_ref, pkt_size)
            scale.nv12_to_rgbpf32(frame, linesize, rgbp.data_ptr(), rgbp.stride(1))

    l_n_frame.append(n_frame)

if __name__ == '__main__':
    file_path = '../build/bunny.mp4'
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]

    n_thread = 2
    l_n_frame = []
    l_thread = []
    for i in range(n_thread):
        th = threading.Thread(target=extract_heif_proc, args=(file_path, l_n_frame))
        l_thread.append(th)
        th.start()
    t0 = time.time()
    for th in l_thread:
        th.join()
    sum = reduce(lambda x,y:x+y, l_n_frame)
    print('sum =', sum, ', fps =', sum / (time.time() - t0))