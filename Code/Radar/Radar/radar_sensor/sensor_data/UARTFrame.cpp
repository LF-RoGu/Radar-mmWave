#include "UARTFrame.h"

constexpr uint64_t MAGIC_WORD = 0x0708050604030201;

UART_frame::UART_frame() {}

uint32_t UART_frame::toLittleEndian32(const uint8_t* data, uint8_t size) 
{
    uint32_t result = 0;
    for (uint8_t i = 0; i < size && i < 4; ++i) {
        result |= static_cast<uint32_t>(data[i]) << (8 * i);
    }
    return result;
}

uint64_t UART_frame::toLittleEndian64(const uint8_t* data, uint8_t size) 
{
    uint64_t result = 0;
    for (uint8_t i = 0; i < size && i < 8; ++i) {
        result |= static_cast<uint64_t>(data[i]) << (8 * i);
    }
    return result;
}

Frame_header::Frame_header(const std::vector<uint8_t>& data) 
{
    parseFrameHeader(data);
}

FrameHeaderData Frame_header::parseFrameHeader(const std::vector<uint8_t>& data) 
{
    FrameHeaderData headerData;
    size_t offset = 0;

    uint64_t magicWord = toLittleEndian64(&data[offset], 8);
    // Check if the magic word matches the expected value
    if (magicWord != MAGIC_WORD) {
        std::cerr << "Error: Invalid magic word detected! Aborting frame parsing.\n";
        return {}; // Return an empty FrameHeaderData or handle error appropriately
    }
    for (int i = 0; i < 4; ++i) {
        headerData.magicWord_u16[i] = (magicWord >> (16 * i)) & 0xFFFF;
    }
    offset += 8;

    headerData.version_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.totalPacketLength_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.platform_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.frameNumber_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.timeCpuCycles_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.numDetectedObj_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.numTLVs_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.subFrameNumber_u32 = toLittleEndian32(&data[offset], 4);

    return headerData;
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

TLV_header::TLV_header()
{
}

void TLV_header::parseTLVHeader(const uint8_t* data, size_t& offset)
{
    TLVHeaderData_str.type_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    TLVHeaderData_str.length_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;
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

void TLV_payload::parsePayload(const uint8_t* data, size_t& offset, const TLVHeaderData& header) {
    // Implement parsing logic based on the type and length from the header
    uint32_t type = header.type_u32;
    uint32_t length = header.length_u32;

    switch (type) {
    case 1: // Detected points
    {
        std::vector<DetectedPoints> points;
        for (size_t i = 0; i < length / sizeof(DetectedPoints); ++i) {
            DetectedPoints point;
            point.x_f = EndianUtils::readFloatFromLittleEndian(data, offset);
            point.y_f = EndianUtils::readFloatFromLittleEndian(data, offset);
            point.z_f = EndianUtils::readFloatFromLittleEndian(data, offset);
            point.doppler_f = EndianUtils::readFloatFromLittleEndian(data, offset);
            points.push_back(point);
        }
        setDetectedPoints(points);
    }
    break;
    case 2: // Range Profile
    {
        std::vector<RangeProfilePoint> points;
        for (size_t i = 0; i < length / sizeof(uint16_t); ++i) {
            RangeProfilePoint point;
            point.rangePoint = EndianUtils::readLittleEndian16(data, offset);
            offset += sizeof(uint16_t);
            points.push_back(point);
        }
        setRangeProfilePoints(points);
    }
    break;
    case 3: // Noise Profile
    {
        std::vector<NoiseProfilePoint> points;
        for (size_t i = 0; i < length / sizeof(uint16_t); ++i) {
            NoiseProfilePoint point;
            point.noisePoint = EndianUtils::readLittleEndian16(data, offset);
            offset += sizeof(uint16_t);
            points.push_back(point);
        }
        setNoiseProfilePoints(points);
    }
    break;
    case 4: // Azimuth Static Heatmap
    {
        std::vector<AzimuthHeatmapPoint> points;
        size_t numRangeBins = length / (4 * sizeof(uint16_t));
        for (size_t i = 0; i < numRangeBins; ++i) {
            AzimuthHeatmapPoint point;
            point.imag = EndianUtils::readLittleEndianInt16(data, offset);
            point.real = EndianUtils::readLittleEndianInt16(data, offset);
            points.push_back(point);
        }
        setAzimuthHeatmapPoints(points);
    }
    break;
    case 7: // Side Info for Detected Points
    {
        std::vector<SideInfoPoint> points;
        for (size_t i = 0; i < length / 4; ++i) {
            SideInfoPoint point;
            point.snr = EndianUtils::readLittleEndianInt16(data, offset);
            point.noise = EndianUtils::readLittleEndianInt16(data, offset);
            points.push_back(point);
        }
        setSideInfoPoints(points);
    }
    break;
    default:
        std::cerr << "Unknown TLV type: " << type << "\n";
        offset += length; // Skip unknown TLV
        break;
    }
}

void TLV_payload::setDetectedPoints(const std::vector<DetectedPoints>& points) {
    detectedPoints_vect = points;
}

void TLV_payload::setRangeProfilePoints(const std::vector<RangeProfilePoint>& points) {
    RangeProfilePoint_vect = points;
}

void TLV_payload::setNoiseProfilePoints(const std::vector<NoiseProfilePoint>& points) {
    NoiseProfilePoint_vect = points;
}

void TLV_payload::setAzimuthHeatmapPoints(const std::vector<AzimuthHeatmapPoint>& points) {
    AzimuthHeatmapPoint_vect = points;
}

void TLV_payload::setSideInfoPoints(const std::vector<SideInfoPoint>& points) {
    SideInfoPoint_vect = points;
}

void TLV_payload::setSphericalCoordinates(const std::vector<SphericalCoordinate>& coordinates) {
    SphericalCoordinate_vect = coordinates;
}

void TLV_payload::setTargetData(const std::vector<TargetData>& targets) {
    TargetData_vect = targets;
}

void TLV_payload::setPointCloudUnits(const std::vector<PointCloudUnit>& units) {
    PointCloudUnit_vect = units;
}

void TLV_payload::setCompressedPointCloud(const std::vector<CompressedPoint>& points) {
    CompressedPoint_vect = points;
}

void TLV_payload::setPresenceDetection(const std::vector<bool>& presence) {
    presenceDetection_vect = presence;
}

std::vector<DetectedPoints> TLV_payload::getDetectedPoints() const {
    return detectedPoints_vect;
}

std::vector<RangeProfilePoint> TLV_payload::getRangeProfilePoints() const {
    return RangeProfilePoint_vect;
}

std::vector<NoiseProfilePoint> TLV_payload::getNoiseProfilePoints() const {
    return NoiseProfilePoint_vect;
}

std::vector<AzimuthHeatmapPoint> TLV_payload::getAzimuthHeatmapPoints() const {
    return AzimuthHeatmapPoint_vect;
}

std::vector<SideInfoPoint> TLV_payload::getSideInfoPoints() const {
    return SideInfoPoint_vect;
}

std::vector<SphericalCoordinate> TLV_payload::getSphericalCoordinates() const {
    return SphericalCoordinate_vect;
}

std::vector<TargetData> TLV_payload::getTargetData() const {
    return TargetData_vect;
}

std::vector<PointCloudUnit> TLV_payload::getPointCloudUnits() const {
    return PointCloudUnit_vect;
}

std::vector<CompressedPoint> TLV_payload::getCompressedPointCloud() const {
    return CompressedPoint_vect;
}

std::vector<bool> TLV_payload::getPresenceDetection() const {
    return presenceDetection_vect;
}