CUDA_PATH = /usr/local/cuda
HEIF_INC_PATH = /usr/local/include
FF_PATH = /usr/local

CC = gcc
GCC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
CFLAGS = -g -m64 -O3 -std=c11 -fPIC
CCFLAGS = -g -m64 -O3 -std=c++17 -fPIC
NVCCFLAGS = -g -m64 -O3 -std=c++11 -Xcompiler -fPIC
INCLUDES += -Iinclude -I$(CUDA_PATH)/include -I$(FF_PATH)/include
LDFLAGS := -L/usr/local/lib -L/usr/local/cuda/lib64 -L/lib/x86_64-linux-gnu -L$(FF_PATH)/lib $(LDFLAGS)
LDFLAGS +=  -lavformat -lavutil -lavcodec -lswresample -lavfilter -lswscale
LDFLAGS += -lstdc++fs -ldl -lpthread -lcuda -lcudart -l:libnvidia-encode.so.1 -l:libnvcuvid.so.1 -lnvjpeg
LDFLAGS_GL := -lGLEW -lglut -lGLU -lGL -lX11 -lXmu
LDFLAGS_HEIF := -lheif_writer_shared -lheif_shared
INCLUDES_HEIF := -I$(HEIF_INC_PATH)/heif/common
INCLUDES += $(INCLUDES_HEIF)

BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

BIN = $(addprefix $(BUILD_DIR)/, AppMux AppAudDec AppAudEnc AppAudFilt AppAudTrans AppVidDec AppVidEnc AppVidEncPerf AppVidFilt AppVidTrans AppAvTrans AppNvDecPerf AppNvEnc AppNvEncPerf AppNvDecImageProvider \
		AppNvDec AppNvDecScan AppHevcParse AppNvjpegDec AppExtract AppSelect)
# BIN_CUDA = $(addprefix $(BUILD_DIR)/, AppMeTrans AppNvTrans)
BIN_GL = $(addprefix $(BUILD_DIR)/, AppNvDecGL)
BIN_HEIF = $(addprefix $(BUILD_DIR)/, AppHeifDec AppHeifEnc AppExtractPerf)
SO = $(addprefix $(BUILD_DIR)/, CFrameExtractor.so CHeif.so CSwscale.so libmetrans.so)
OBJ = $(shell find . -name '*.o')
DEP = $(OBJ:.o=.d)

essentials: $(BIN) $(BIN_CUDA) $(SO)
all_but_gl: essentials $(BIN_HEIF)
all: all_but_gl $(BIN_GL)

$(BUILD_DIR)/AppMeTrans: $(addprefix $(OBJ_DIR)/, AppMeTrans/AppMeTrans.o AppMeTrans/TransData.o NvCodec/NvDecLite.o NvCodec/NvEncLiteUnbuffered.o NvCodec/NvEncLite.o NvCodec/Resize.o)
$(BUILD_DIR)/AppNvDecScan: $(addprefix $(OBJ_DIR)/, AppNvDecScan.o NvCodec/NvDecLite.o)
$(BUILD_DIR)/AppHevcParse: $(addprefix $(OBJ_DIR)/, AppHevcParse.o NvCodec/NvDecLite.o HevcParser/BitstreamReader.o HevcParser/Hevc.o HevcParser/HevcParser.o HevcParser/HevcParserImpl.o HevcParser/HevcUtils.o)
$(BUILD_DIR)/AppExtract: $(addprefix $(OBJ_DIR)/, AppExtract.o NvCodec/NvDecLite.o)
$(BUILD_DIR)/AppExtractPerf: $(addprefix $(OBJ_DIR)/, AppExtractPerf.o NvCodec/NvDecLite.o NvCodec/NvEncLiteUnbuffered.o NvCodec/NvEncLite.o NvCodec/NvHeifWriter.o)
$(BUILD_DIR)/AppSelect: $(addprefix $(OBJ_DIR)/, AppSelect.o NvCodec/NvDecLite.o)

$(BUILD_DIR)/AppMux: $(addprefix $(OBJ_DIR)/, AppMux.o)
$(BUILD_DIR)/AppAudDec: $(addprefix $(OBJ_DIR)/, AppAudDec.o)
$(BUILD_DIR)/AppAudEnc: $(addprefix $(OBJ_DIR)/, AppAudEnc.o)
$(BUILD_DIR)/AppAudFilt: $(addprefix $(OBJ_DIR)/, AppAudFilt.o)
$(BUILD_DIR)/AppAudTrans: $(addprefix $(OBJ_DIR)/, AppAudTrans.o)
$(BUILD_DIR)/AppVidDec: $(addprefix $(OBJ_DIR)/, AppVidDec.o)
$(BUILD_DIR)/AppVidEnc: $(addprefix $(OBJ_DIR)/, AppVidEnc.o)
$(BUILD_DIR)/AppVidEncPerf: $(addprefix $(OBJ_DIR)/, AppVidEncPerf.o)
$(BUILD_DIR)/AppVidFilt: $(addprefix $(OBJ_DIR)/, AppVidFilt.o)
$(BUILD_DIR)/AppVidTrans: $(addprefix $(OBJ_DIR)/, AppVidTrans.o)
$(BUILD_DIR)/AppAvTrans: $(addprefix $(OBJ_DIR)/, AppAvTrans.o)

$(BUILD_DIR)/AppNvDec: $(addprefix $(OBJ_DIR)/, AppNvDec.o NvCodec/NvDecLite.o)
$(BUILD_DIR)/AppNvDecPerf: $(addprefix $(OBJ_DIR)/, AppNvDecPerf.o NvCodec/NvDecLite.o)
$(BUILD_DIR)/AppNvDecImageProvider: $(addprefix $(OBJ_DIR)/, AppNvDecImageProvider.o NvCodec/NvDecLite.o NvCodec/ColorSpace.o)
$(BUILD_DIR)/AppNvEnc: $(addprefix $(OBJ_DIR)/, AppNvEnc.o NvCodec/NvEncLiteUnbuffered.o NvCodec/NvEncLite.o)
$(BUILD_DIR)/AppNvEncPerf: $(addprefix $(OBJ_DIR)/, AppNvEncPerf.o NvCodec/NvEncLiteUnbuffered.o NvCodec/NvEncLite.o)
$(BUILD_DIR)/AppNvTrans: $(addprefix $(OBJ_DIR)/, AppNvTrans.o NvCodec/BitDepth.o NvCodec/NvDecLite.o NvCodec/NvEncLiteUnbuffered.o NvCodec/NvEncLite.o NvCodec/Resize.o)
$(BUILD_DIR)/AppNvjpegDec: $(addprefix $(OBJ_DIR)/, AppNvjpegDec.o)
$(BUILD_DIR)/AppHeifDec: $(addprefix $(OBJ_DIR)/, AppHeifDec.o NvCodec/NvHeifReader.o NvCodec/NvDecLite.o)
$(BUILD_DIR)/AppHeifEnc: $(addprefix $(OBJ_DIR)/, AppHeifEnc.o NvCodec/NvHeifWriter.o NvCodec/NvEncLiteUnbuffered.o NvCodec/NvEncLite.o)

$(BUILD_DIR)/AppNvDecGL: $(addprefix $(OBJ_DIR)/, AppNvDecGL/AppNvDecGL.o NvCodec/NvDecLite.o NvCodec/ColorSpace.o)

$(BUILD_DIR)/CFrameExtractor.so: $(addprefix $(OBJ_DIR)/, CFrameExtractor.o NvCodec/NvDecLite.o NvCodec/ColorSpace.o)
$(BUILD_DIR)/libmetrans.so: $(addprefix $(OBJ_DIR)/, NvCodec/NvDecLite.o NvCodec/NvEncLiteUnbuffered.o NvCodec/NvEncLite.o NvCodec/ColorSpace.o)
$(BUILD_DIR)/CHeif.so: $(addprefix $(OBJ_DIR)/, CHeif.o NvCodec/NvDecLite.o NvCodec/NvEncLite.o NvCodec/NvEncLiteUnbuffered.o NvCodec/NvHeifReader.o NvCodec/NvHeifWriter.o)
$(BUILD_DIR)/CSwscale.so: $(addprefix $(OBJ_DIR)/, CSwscale.o)

-include $(DEP)

VPATH = include:samples:app

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(GCC) $(CCFLAGS) -MD -MP $(INCLUDES) -o $@ -c $<

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MD -MP $(INCLUDES) -o $@ -c $<

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -M -MT $@ $(INCLUDES) -o $(@:.o=.d) $<
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<

$(BIN):
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)

$(BIN_GL):
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS) $(LDFLAGS_GL)

$(BIN_CUDA):
	$(NVCC) $(NVCCFLAGS) -o $@ $+ $(LDFLAGS)

$(SO):
	$(GCC) $(CCFLAGS) -shared -o $@ $+ $(LDFLAGS) $(LDFLAGS_HEIF)

$(BIN_HEIF):
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS) $(LDFLAGS_HEIF) $(INCLUDES_HEIF)

clean:
	rm -rf $(BIN) $(BIN_CUDA) $(BIN_GL) $(OBJ_DIR) $(BIN_HEIF)

distclean: clean
	cd $(BUILD_DIR) && rm -f out.* bunny.aac bunny.nv12 bunny.iyuv bunny.h264 bunny.hevc bunny.f32 perf.h264 perf.hevc perf_*.h264

data: all_but_gl
	cd $(BUILD_DIR) && ./AppNvDec -i bunny.mp4 -o bunny.nv12
	cd $(BUILD_DIR) && ./AppNvDec -i bunny.mp4 -o bunny.iyuv -outplanar
	cd $(BUILD_DIR) && ./AppNvEnc -i bunny.iyuv -o bunny.h264
	cd $(BUILD_DIR) && ./AppNvEnc -i bunny.iyuv -o bunny.hevc -codec hevc
	cd $(BUILD_DIR) && ./AppNvEnc -i bunny.iyuv -o perf.h264 -case 2 -frame 5000
	cd $(BUILD_DIR) && ./AppNvEnc -i bunny.iyuv -o perf.hevc -case 2 -frame 5000 -codec hevc

.PHONY: essentials all_but_gl all clean distclean data
