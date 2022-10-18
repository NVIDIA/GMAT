#ifndef WIN32
#include <dlfcn.h>
#endif

#include "NvCodec/NvHeifReader.h"

NvHeifReader::NvHeifReader(const char* inFilePath) {
        
    m_reader = HEIF::Reader::Create();
    if (m_reader->initialize(inFilePath) != HEIF::ErrorCode::OK)
    {
        HEIF::Reader::Destroy(m_reader);
        return;
    }
    
    // get information about all input file content
    m_reader->getFileInformation(m_fileInfo);
    
    // Image sequence
    if (m_fileInfo.features & HEIF::FileFeatureEnum::HasImageSequence) {
        for (const auto& trackProperties : m_fileInfo.trackInformation) {
            const auto sequenceId = trackProperties.trackId;
            LOG(INFO) << "Track ID " << sequenceId.get() << "\n";  // Context ID corresponds to the track ID

            if (trackProperties.features & HEIF::TrackFeatureEnum::IsMasterImageSequence
                || trackProperties.features & HEIF::TrackFeatureEnum::IsVideoTrack) {
                m_trackInfo = trackProperties;
                m_index = 0;
                break;
            }
        }
    }

    m_pktData = new uint8_t[1024 * 1024];
}

NvHeifReader::~NvHeifReader() {
    HEIF::Reader::Destroy(m_reader);
    delete[] m_pktData;
}

// Read the first master image from the HEIF file
bool NvHeifReader::readImage(uint8_t* &pktData, size_t &pktBytes) {
    // Verify that the file has one master image in the MetaBox
    // Image collection is not supported at this moment
    if (!(m_fileInfo.features & HEIF::FileFeatureEnum::HasSingleImage))
    {
        LOG(ERROR) << "No still image found\n";
        return false;
    }

    for (const auto& image : m_fileInfo.rootMetaBoxInformation.itemInformations) {
        if (image.features & HEIF::ItemFeatureEnum::IsMasterImage) {
            if (image.type != ("hvc1") && image.type != ("hev1")) {
                LOG(ERROR) << "Only supports hevc image\n";
                return false;
            }

            HEIF::DecoderConfiguration inputdecoderConfig{};
            m_reader->getDecoderParameterSets(image.itemId, inputdecoderConfig);

            size_t parameterSetBytes = 0;
            for (int i = 0; i < inputdecoderConfig.decoderSpecificInfo.size; i++) {
                parameterSetBytes += inputdecoderConfig.decoderSpecificInfo[i].decSpecInfoData.size;
            }

            pktBytes = parameterSetBytes + image.size;
            // pktData = new uint8_t[pktBytes];

            uint8_t *psData = m_pktData;
            for (int i = 0; i < inputdecoderConfig.decoderSpecificInfo.size; i++) {
                memcpy(psData, 
                       inputdecoderConfig.decoderSpecificInfo[i].decSpecInfoData.elements, 
                       inputdecoderConfig.decoderSpecificInfo[i].decSpecInfoData.size);
                psData += inputdecoderConfig.decoderSpecificInfo[i].decSpecInfoData.size;
            }

            m_reader->getItemData(image.itemId, m_pktData + parameterSetBytes, pktBytes);
            pktData = m_pktData;
            break;
        }
    }
    return true;
}

bool NvHeifReader::readVideoFrame(std::vector<uint8_t*> &v_pktData, std::vector<size_t> &v_pktBytes) {
    // Array<ImageId> itemIds;
    // m_reader->getMasterImages(itemIds);
    // getDecoderCodeType(const ImageId& imageId, FourCC& type) const = 0;

    // for (const auto& trackProperties : m_fileInfo.trackInformation) {
    const auto sequenceId = m_trackInfo.trackId;
    LOG(INFO) << "Track ID " << sequenceId.get();  // Context ID corresponds to the track ID

    if (m_trackInfo.features & HEIF::TrackFeatureEnum::IsMasterImageSequence
        || m_trackInfo.features & HEIF::TrackFeatureEnum::IsVideoTrack) {
        
        std::cout << "m_index: " << m_index << std::endl;
        if (m_index >= m_trackInfo.sampleProperties.size) {
            return true;
        }
        else {
            auto sampleProperties = m_trackInfo.sampleProperties[m_index];
            // A sample might have decoding dependencies. The simplest way to handle this is just to always ask and
                // decode all dependencies.
            HEIF::Array<HEIF::SequenceImageId> itemsToDecode;
            m_reader->getDecodeDependencies(sequenceId, sampleProperties.sampleId, itemsToDecode);
            for (auto dependencyId : itemsToDecode)
            {
                uint8_t* pktData = new uint8_t[1024 * 1024];
                size_t pktBytes = 1024  * 1024;
                std::cout << "Demuxing packet " << dependencyId.get() << " from heif\n";
                m_reader->getItemDataWithDecoderParameters(sequenceId, dependencyId, pktData, pktBytes);
                v_pktData.push_back(pktData);
                v_pktBytes.push_back(pktBytes);
                std::cout << "Packet number: " << v_pktData.size() << ". Packet size: " << pktBytes << std::endl;
            }

            m_index++;
            return true;
        }
    }
    return false;
    // }
}
