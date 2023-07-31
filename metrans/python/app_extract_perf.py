import torch
import torch.cuda as cuda
from frame_extractor import FrameExtractor
import threading
from functools import reduce
import time
import sys
import ctypes

cudaFree = ctypes.CDLL('libcudart.so').cudaFree

dev = torch.device("cuda:0")
# initialize cuda runtime
dummy = torch.empty((1,), device=dev)

def extract_proc(file_path, l_n_frame):
    # setting cuda context for current context
    cudaFree(0)

    with open(file_path, 'rb') as mp4:
        extractor = FrameExtractor(buffer=mp4.read())

    n_frame = 0
    bgr = torch.empty((3, extractor.get_height(), extractor.get_width()), dtype=torch.float32, device=dev)
    with cuda.stream(cuda.Stream(dev)):
        while extractor.extract_to_device_buffer(bgr.data_ptr(), cuda.current_stream().cuda_stream):
            n_frame += 1
    l_n_frame.append(n_frame)

if __name__ == '__main__':
    file_path = '../build/bunny.mp4'
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]

    n_thread = 2
    l_n_frame = []
    l_thread = []
    for i in range(n_thread):
        th = threading.Thread(target=extract_proc, args=(file_path, l_n_frame))
        l_thread.append(th)
        th.start()
    t0 = time.time()
    for th in l_thread:
        th.join()
    sum = reduce(lambda x,y:x+y, l_n_frame)
    print('sum =', sum, ', fps =', sum / (time.time() - t0))
