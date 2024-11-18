#include "UARTFrame.h"

constexpr uint64_t MAGIC_WORD = 0x0708050603040102;

UART_frame::UART_frame() {}


Frame_header::Frame_header(std::vector<uint8_t>& data) 
{
    parseFrameHeader(data);
}

void Frame_header::parseFrameHeader(std::vector<uint8_t>& data)
{
    EndianUtils EndianUtils_c;

    // Extract magic word (64-bit) from the vector
    uint64_t magicWord = EndianUtils_c.toLittleEndian64(data, 8);

    // Check if the magic word matches the expected value
    if (magicWord != MAGIC_WORD) {
        std::cerr << "Error: Invalid magic word detected! Aborting frame parsing.\n";
        return; // Early exit if the magic word is invalid
    }

    // Extract version (32-bit) from the vector
    setVersion(EndianUtils_c.toLittleEndian32(data, 4));

    // Extract packet length (32-bit) from the vector
    setPacketLength(EndianUtils_c.toLittleEndian32(data, 4));

    // Extract platform (32-bit) from the vector
    setPlatform(EndianUtils_c.toLittleEndian32(data, 4));

    // Extract frame number (32-bit) from the vector
    setFrameNumber(EndianUtils_c.toLittleEndian32(data, 4));

    // Extract time (32-bit) from the vector
    setTime(EndianUtils_c.toLittleEndian32(data, 4));

    // Extract number of detected objects (32-bit) from the vector
    setNumObjDetecter(EndianUtils_c.toLittleEndian32(data, 4));

    // Extract number of TLVs (32-bit) from the vector
    setNumTLV(EndianUtils_c.toLittleEndian32(data, 4));

    // Extract subframe number (32-bit) from the vector
    setSubframeNum(EndianUtils_c.toLittleEndian32(data, 4));
}

void Frame_header::setVersion(uint32_t var)
{
    FrameHeader_str.version_u32 = var;
}

void Frame_header::setPacketLength(uint32_t var)
{
    FrameHeader_str.totalPacketLength_u32 = var;
}

void Frame_header::setPlatform(uint32_t var)
{
    FrameHeader_str.platform_u32 = var;
}

void Frame_header::setFrameNumber(uint32_t var)
{
    FrameHeader_str.frameNumber_u32 = var;
}

void Frame_header::setTime(uint32_t var)
{
    FrameHeader_str.timeCpuCycles_u32 = var;
}

void Frame_header::setNumObjDetecter(uint32_t var)
{
    FrameHeader_str.numDetectedObj_u32 = var;
}

void Frame_header::setNumTLV(uint32_t var)
{
    FrameHeader_str.numTLVs_u32 = var;
}

void Frame_header::setSubframeNum(uint32_t var)
{
    FrameHeader_str.subFrameNumber_u32 = var;
}

uint32_t Frame_header::getVersion() const
{
    return FrameHeader_str.version_u32;
}

uint32_t Frame_header::getPacketLength() const
{
    return FrameHeader_str.totalPacketLength_u32;
}

uint32_t Frame_header::getPlatform() const
{
    return FrameHeader_str.platform_u32;
}

uint32_t Frame_header::getFrameNumber() const
{
    return FrameHeader_str.frameNumber_u32;
}

uint32_t Frame_header::getTime() const
{
    return FrameHeader_str.timeCpuCycles_u32;
}

uint32_t Frame_header::getNumObjDetecter() const
{
    return FrameHeader_str.numDetectedObj_u32;
}

uint32_t Frame_header::getNumTLV() const
{
    return FrameHeader_str.numTLVs_u32;
}

uint32_t Frame_header::getSubframeNum() const
{
    return FrameHeader_str.subFrameNumber_u32;
}


TLV_frame::TLV_frame() 
{
}

TLV_frame::TLV_frame(std::vector<uint8_t>& data, uint32_t numDetectedObj_var)
{
    TLVHeaderData TLVHeaderData_str;
    TLVPayloadData TLVPayloadData_str;
    TLVHeaderData_str = parseTLVHeader(data);
    TLVPayloadData_str = parseTLVPayload(data, TLVHeaderData_str, numDetectedObj_var);
}

TLVHeaderData TLV_frame::parseTLVHeader(std::vector<uint8_t>& data)
{
    EndianUtils EndianUtils_c;
    TLV_header TLV_header_c;
    TLVHeaderData TLVHeaderData_str;

    TLV_header_c.setType(EndianUtils_c.toLittleEndian32(data,4));
    TLV_header_c.setLength(EndianUtils_c.toLittleEndian32(data, 4));

    TLVHeaderData_str.type_u32 = TLV_header_c.getType();
    TLVHeaderData_str.length_u32 = TLV_header_c.getLength();
    return TLVHeaderData_str;
}

TLVPayloadData TLV_frame::parseTLVPayload(std::vector<uint8_t>& data, TLVHeaderData TLVHeaderData_var, uint32_t numDetectedObj_var)
{
    EndianUtils EndianUtils_c;
    TLV_payload TLV_payload_c;
    TLVPayloadData TLVPayloadData_str;
    // Implement parsing logic based on the type and length from the header
    TLVHeaderData_var.type_u32;
    TLVHeaderData_var.length_u32;

    switch (TLVHeaderData_var.type_u32) {
    case 1: // Detected points
    {
        DetectedPoints DetectedPoints_var;
        for (uint32_t i = 0; i < numDetectedObj_var; i++)
        {
            DetectedPoints_var.x_f = EndianUtils_c.toLittleEndian32(data, 4);
            DetectedPoints_var.y_f = EndianUtils_c.toLittleEndian32(data, 4);
            DetectedPoints_var.z_f = EndianUtils_c.toLittleEndian32(data, 4);
            DetectedPoints_var.doppler_f = EndianUtils_c.toLittleEndian32(data, 4);

            TLV_payload_c.setDetectedPoints(DetectedPoints_var);
        }
    }
    break;
    case 2: // Range Profile
    {

    }
    break;
    case 3: // Noise Profile
    {

    }
    break;
    case 4: // Azimuth Static Heatmap
    {

    }
    break;
    case 7: // Side Info for Detected Points
    {
        SideInfoPoint SideInfoPoint_var;
        for (uint32_t i = 0; i < numDetectedObj_var; i++)
        {
            SideInfoPoint_var.snr = EndianUtils_c.toLittleEndian32(data, 4);
            SideInfoPoint_var.snr = EndianUtils_c.toLittleEndian32(data, 4);

            TLV_payload_c.setSideInfoPoints(SideInfoPoint_var);
        }
    }
    break;
    default:
        std::cerr << "Unknown TLV type " << "\n";
        break;
    }

    return TLVPayloadData_str;
}

TLV_header::TLV_header()
{
}

void TLV_header::setType(uint32_t var)
{
    TLVHeaderData_str.type_u32 = var;
}

void TLV_header::setLength(uint32_t var)
{
    TLVHeaderData_str.length_u32 = var;
}

uint32_t TLV_header::getType() const
{
    return TLVHeaderData_str.type_u32;
}

uint32_t TLV_header::getLength() const
{
    return TLVHeaderData_str.length_u32;
}

TLV_payload::TLV_payload()
{
}

void TLV_payload::setDetectedPoints(DetectedPoints DetectedPoints_var) 
{
    detectedPoints_vect.push_back(DetectedPoints_var);
}

void TLV_payload::setRangeProfilePoints(RangeProfilePoint RangeProfilePoint_var)
{
    RangeProfilePoint_vect.push_back(RangeProfilePoint_var);
}

void TLV_payload::setNoiseProfilePoints(NoiseProfilePoint NoiseProfilePoint_var)
{
    NoiseProfilePoint_vect.push_back(NoiseProfilePoint_var);
}

void TLV_payload::setAzimuthHeatmapPoints(AzimuthHeatmapPoint AzimuthHeatmapPoint_var)
{
    AzimuthHeatmapPoint_vect.push_back(AzimuthHeatmapPoint_var);
}
void TLV_payload::setSideInfoPoints(SideInfoPoint SideInfoPoint_var)
{
    SideInfoPoint_vect.push_back(SideInfoPoint_var);
}

void TLV_payload::setSphericalCoordinates(SphericalCoordinate SphericalCoordinate_var)
{
    SphericalCoordinate_vect.push_back(SphericalCoordinate_var);
}

void TLV_payload::setTargetData(TargetData TargetData_var)
{
    TargetData_vect.push_back(TargetData_var);
}

void TLV_payload::setPointCloudUnits(PointCloudUnit PointCloudUnit_var)
{
    PointCloudUnit_vect.push_back(PointCloudUnit_var);
}

void TLV_payload::setCompressedPointCloud(CompressedPoint CompressedPoint_var)
{
    CompressedPoint_vect.push_back(CompressedPoint_var);
}

void TLV_payload::setPresenceDetection(bool var)
{
    presenceDetection_vect.push_back(var);
}

std::vector<DetectedPoints> TLV_payload::getDetectedPoints() 
{
    return detectedPoints_vect;
}

std::vector<RangeProfilePoint> TLV_payload::getRangeProfilePoints() 
{
    return RangeProfilePoint_vect;
}

std::vector<NoiseProfilePoint> TLV_payload::getNoiseProfilePoints() 
{
    return NoiseProfilePoint_vect;
}

std::vector<AzimuthHeatmapPoint> TLV_payload::getAzimuthHeatmapPoints() 
{
    return AzimuthHeatmapPoint_vect;
}

std::vector<SideInfoPoint> TLV_payload::getSideInfoPoints() 
{
    return SideInfoPoint_vect;
}

std::vector<SphericalCoordinate> TLV_payload::getSphericalCoordinates() 
{
    return SphericalCoordinate_vect;
}

std::vector<TargetData> TLV_payload::getTargetData() 
{
    return TargetData_vect;
}

std::vector<PointCloudUnit> TLV_payload::getPointCloudUnits() 
{
    return PointCloudUnit_vect;
}

std::vector<CompressedPoint> TLV_payload::getCompressedPointCloud() 
{
    return CompressedPoint_vect;
}

std::vector<bool> TLV_payload::getPresenceDetection() 
{
    return presenceDetection_vect;
}