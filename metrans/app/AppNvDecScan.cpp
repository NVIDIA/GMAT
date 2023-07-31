/*
* Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <map>
#include <iostream>
#include <algorithm>
#include <thread>
#include <mutex>
#include <cuda.h>
#include <dirent.h>
#include "AvToolkit/Demuxer.h"
#include "NvCodec/NvDecLite.h"
#include "NvCodec/NvCommon.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

using namespace std;

int DecodeFile(CUcontext cuContext, const char *szInDirPath, const char *szInFileName, const char *szOutDirPath, bool bSlow)
{
    char szInFilePath[1024] = "";
    sprintf(szInFilePath, "%s/%s", szInDirPath, szInFileName);
    Demuxer demuxer(szInFilePath);
    ofstream fOut;
    if (*szOutDirPath) {
        char szOutFilePath[1024] = "";
        sprintf(szOutFilePath, "%s/%dx%d.%s.nv12", szOutDirPath, demuxer.GetVideoStream()->codecpar->width,
                demuxer.GetVideoStream()->codecpar->height, szInFileName);
        fOut.open(szOutFilePath, ios::out | ios::binary);
    }
	cudaVideoCodec eCodec = FFmpeg2NvCodecId(demuxer.GetVideoStream()->codecpar->codec_id);
    NvDecLite dec(cuContext, false, eCodec);

    AVPacket *pkt;
    int nFrame = 0;
    do {
        demuxer.Demux(&pkt);
        if (pkt->size) {
            uint8_t b = pkt->data[2] == 1 ? pkt->data[3] : pkt->data[4];
            int nal_ref_idc = b >> 5;
            int nal_unit_type = b & 0x1f;
            if (!bSlow && !nal_ref_idc && nal_unit_type == 1) {
            	continue;
            }
        }
        uint8_t **ppFrame = NULL;
        NvFrameInfo *pInfo = NULL;
        int nFrameReturned = dec.Decode(pkt->data, pkt->size, &ppFrame, &pInfo);
        for (int i = 0; i < nFrameReturned; i++) {
            fOut.write(reinterpret_cast<char *>(ppFrame[i]), pInfo[i].nFrameSize);
        }
        if (fOut.is_open()) {
            for (int i = 0; i < nFrameReturned; i++) {
                fOut.write(reinterpret_cast<char *>(ppFrame[i]), pInfo[i].nFrameSize);
            }
        }
        nFrame += nFrameReturned;
    } while (pkt->size);

    return nFrame;
}

void ShowHelpAndExit(const char *szBadOption = NULL) {
    if (szBadOption) {
        cout << "Error parsing \"" << szBadOption << "\"" << endl;
    }
    cout << "Options:" << endl
        << "-i             Input dir path" << endl
        << "-o             Output dir path" << endl
        << "-n             Number of threads" << endl
        << "-gpu           Ordinal of GPU to use" << endl
        << "-slow          Slow scan (no skipping)" << endl
        ;
    cout << endl;
    exit(1);
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, char *szOutputFileName, int &nth, int &iGpu, bool &bSlow)
{
    ostringstream oss;
    int i;
    for (i = 1; i < argc; i++) {
        if (!_stricmp(argv[i], "-h")) {
            ShowHelpAndExit();
        }
        if (!_stricmp(argv[i], "-i")) {
            if (++i == argc) {
                ShowHelpAndExit("-i");
            }
            sprintf(szInputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-o")) {
            if (++i == argc) {
                ShowHelpAndExit("-o");
            }
            sprintf(szOutputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-n")) {
            if (++i == argc) {
                ShowHelpAndExit("-n");
            }
            nth = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-gpu")) {
            if (++i == argc) {
                ShowHelpAndExit("-gpu");
            }
            iGpu = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-slow")) {
            bSlow = true;
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
}

void DecodeProc(mutex *pMutex, map<string, int> *pName2nFrame,
        CUcontext cuContext, const char *szInDirPath, vector<string> *pvName, const char *szOutDirPath, bool bSlow) {
    while (true) {
        pMutex->lock();
        if (pvName->empty()) {
            pMutex->unlock();
            return;
        }
        string name = pvName->back();
        pvName->pop_back();
        pMutex->unlock();

        int nFrame = DecodeFile(cuContext, szInDirPath, name.c_str(), szOutDirPath, bSlow);

        pMutex->lock();
        pName2nFrame->insert(pair<string, int>(name, nFrame));
        pMutex->unlock();
    }
}

int main(int argc, char **argv) {
    char szInDirPath[256] = "tmp",
        szOutDirPath[256] = "";
    int iGpu = 0;
    bool bSlow = false;
    int nth = 1;
    ParseCommandLine(argc, argv, szInDirPath, szOutDirPath, nth, iGpu, bSlow);

    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu) {
        cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << endl;
        return 1;
    }
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    cout << "GPU in use: " << szDeviceName << endl;
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));

    DIR *dir;
    if ((dir = opendir(szInDirPath)) == NULL) {
    	cout << "Unable to open input dir: " << szInDirPath << endl;
    	return 1;
    }
    struct dirent *ent;
    map<string, int> name2nFrame;
    vector<string> vName;
	while ((ent = readdir(dir)) != NULL) {
		if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..") || strstr(ent->d_name, "msmp")) {
			continue;
		}
		vName.push_back(ent->d_name);
	}
	closedir(dir);

	mutex mutex;
	vector<thread *> vth;
	for (int i = 0; i < nth; i++) {
	    vth.push_back(new thread(DecodeProc, &mutex, &name2nFrame, cuContext, szInDirPath, &vName, szOutDirPath, bSlow));
	}
	for (thread *pth : vth) {
	    pth->join();
	    delete pth;
	}

	for (pair<string, int> kv : name2nFrame) {
		cout << kv.first << ": " << kv.second << endl;
	}

    return 0;
}
