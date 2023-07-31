#pragma once

#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace boost;

struct Options {
    int iGpu;
    string strInputFile;
    int nSession;
    int nFpsLimit;

    string strAudioFilterDesc;
    string strAudioCodec;
    int nAudioBitRate;
    int nAudioSampleRate;
    
    bool bUseSwVideoDecoder;
    string strVideoFilterDesc;
    string strVideoEncParam;

    struct Resolution {
        int nWidth, nHeight;
        string strVideoFilterDesc;
        string strVideoEncParamSuffix;

        string strOutputFormat;
        string strOutputFile;
    };
    vector<Resolution> vRes;

    void Load(string strXmlFile) {
        using boost::property_tree::ptree;
        ptree pt;
        read_xml(strXmlFile, pt);

        iGpu = pt.get<int>("Options.Gpu", 0);
        strInputFile = pt.get<string>("Options.InputFile", "");
        nSession = pt.get<int>("Options.Session", 1);
        nFpsLimit = pt.get<int>("Options.FpsLimit", 0);

        strAudioFilterDesc = pt.get<string>("Options.AudioFilterDesc", "");
        strAudioCodec = pt.get<string>("Options.AudioCodec", "");
        nAudioBitRate = pt.get<int>("Options.AudioBitRate", 0);
        nAudioSampleRate = pt.get<int>("Options.AudioSampleRate", 0);

        bUseSwVideoDecoder = pt.get<bool>("Options.UseSwVideoDecoder", false);
        strVideoFilterDesc = pt.get<string>("Options.VideoFilterDesc", "");
        strVideoEncParam = pt.get<string>("Options.VideoEncParam", "");

        if (!pt.get_child_optional("Options.Resolutions").is_initialized()) {
            return;
        }

        for (ptree::value_type &v : pt.get_child("Options.Resolutions")) {
            ptree &p = v.second;
            Resolution res = {
                p.get<int>("Width", 0),  p.get<int>("Height", 0),
                p.get<string>("VideoFilterDesc", ""), 
                p.get<string>("VideoEncParamSuffix", ""), 

                p.get<string>("OutputFormat", ""), 
                p.get<string>("OutputFile", ""), 
            };
            vRes.push_back(res);
        }
    }
};
