CUDA_PATH = /usr/local/cuda
# TRT_PATH = /usr/local/tensorrt7.2-cuda11.1
TORCH_PATH ?= $(shell python -c "import torch; print(torch.__path__[0])")
TORCH_INCPATH = -I$(TORCH_PATH)/include/torch/csrc/api/include -I$(TORCH_PATH)/include

GCC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
NVCCFLAGS = -arch=sm_75 -g -std=c++14 -Xcompiler -fPIC
CCFLAGS = -g -std=c++14 -fPIC
INCPATH := -I$(TORCH_PATH)/include/torch/csrc/api/include -I$(TORCH_PATH)/include
# INCPATH ?= $(shell python3 -c "import torch.utils.cpp_extension as C; print('-I' + str.join(' -I', C.include_paths()))")
INCLUDES := -isystem $(CUDA_PATH)/include -I./include/ -I$(CUDA_PATH)/include/ $(TORCH_INCPATH) -I../cv-cuda/include
LIBPATH := $(TORCH_PATH)/lib
# LIBPATH ?= $(shell python3 -c "import torch.utils.cpp_extension as C; print(C.library_paths()[0])")
LDFLAGS := -L$(CUDA_PATH)/lib64 -Wl,-rpath=$(CUDA_PATH)/lib64 -L$(LIBPATH) -Wl,-rpath=$(LIBPATH)
LDFLAGS += -L/usr/lib/x86_64-linux-gnu
LDFLAGS += -lnvinfer -lnvonnxparser -lcudart -ldl -lpthread -lcuda -ltorchvision
LDFLAGS += -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lnvrtc -lnvToolsExt
LDFLAGS += -lX11 -lpthread -lXrandr -lXi -lz -lEGL -lGL
LDFLAGS += -lnvidia-encode -lnvcuvid -lavformat -lavutil -lavcodec -lswresample -lavfilter -lnvjpeg
LDFLAGS += -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_cudawarping -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dpm -lopencv_highgui -lopencv_videoio -lopencv_face -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_sfm -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ml -lopencv_ximgproc -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_imgproc -lopencv_cudaarithm -lopencv_core -lopencv_cudev
LDFLAGS += -lcv_cuda -lheif_writer_shared
ODIR = bin
NVCODEC = src/NvCodec/NvDecLite.o src/NvCodec/NvEncLite.o src/NvCodec/NvEncLiteUnbuffered.o src/NvCodec/ColorSpace.o src/NvCodec/NvHeifWriter.o
SRC = src/format_cuda.o sample/tddfa_kernels.o sample/cnpy.o
TDDFA = sample/3ddfa_sample.cpp
_OBJ = $(NVCODEC) $(SRC)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

SO = $(ODIR)/libffgd.so
TDDFA_BIN = $(ODIR)/tddfa
HEIF_ENC = $(ODIR)/heifEnc

.PHONY: all clean
# all: heif_enc nvjpeg_enc tddfa
all: $(HEIF_ENC) $(TDDFA_BIN)

$(TDDFA_BIN): $(TDDFA) SO
	$(GCC) -g -o $@ $< $(INCLUDES) -L$(ODIR) -lffgd $(LDFLAGS)

$(HEIF_ENC): sample/heif_enc.cpp SO
	$(GCC) -g -o $@ $< $(INCLUDES) -L$(ODIR) -lffgd $(LDFLAGS)

$(SO): $(OBJ)
	$(GCC) -shared -Wl,-soname,$(SO) -o $@ $^ $(LDFLAGS)

$(ODIR)/%.o: %.cpp $(DEPS)
	$(GCC) -c -o $@ $< $(CCFLAGS) $(INCPATH) $(INCLUDES)

$(ODIR)/%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(NVCCFLAGS) $(INCPATH) $(INCLUDES)

clean:
	rm -rf $(addprefix $(ODIR)/, $(NVCODEC))
	rm -rf $(addprefix $(ODIR)/, $(SRC))
	rm -rf $(SO)
