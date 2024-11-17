#include "UARTFrame.h"

constexpr uint64_t MAGIC_WORD = 0x0708050603040102;

UART_frame::UART_frame() {}


Frame_header::Frame_header(const std::vector<uint8_t>& data) 
{
    parseFrameHeader(data);
}

FrameHeaderData Frame_header::parseFrameHeader(const std::vector<uint8_t>& data) 
{
    EndianUtils EndianUtils_c;
    FrameHeaderData headerData;
    size_t offset = 0;

    uint64_t magicWord = EndianUtils_c.toLittleEndian64(&data[offset], 8);
    // Check if the magic word matches the expected value
    if (magicWord != MAGIC_WORD) {
        std::cerr << "Error: Invalid magic word detected! Aborting frame parsing.\n";
        return {}; // Return an empty FrameHeaderData or handle error appropriately
    }
    for (int i = 0; i < 4; ++i) {
        headerData.magicWord_u16[i] = (magicWord >> (16 * i)) & 0xFFFF;
    }
    offset += 8;

    setVersion(EndianUtils_c.toLittleEndian32(&data[offset], 4));
    offset += 4;

    setPacketLength(EndianUtils_c.toLittleEndian32(&data[offset], 4));
    offset += 4;

    setPlatform(EndianUtils_c.toLittleEndian32(&data[offset], 4));
    offset += 4;

    setFrameNumber(EndianUtils_c.toLittleEndian32(&data[offset], 4));
    offset += 4;

    setTime(EndianUtils_c.toLittleEndian32(&data[offset], 4));
    offset += 4;

    setNumObjDetecter(EndianUtils_c.toLittleEndian32(&data[offset], 4));
    offset += 4;

    setNumTLV(EndianUtils_c.toLittleEndian32(&data[offset], 4));
    offset += 4;

    setSubframeNum(EndianUtils_c.toLittleEndian32(&data[offset], 4));

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
    EndianUtils EndianUtils_c;
    TLVHeaderData_str.type_u32 = EndianUtils_c.toLittleEndian32(&data[offset], 4);
    offset += 4;

    TLVHeaderData_str.length_u32 = EndianUtils_c.toLittleEndian32(&data[offset], 4);
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