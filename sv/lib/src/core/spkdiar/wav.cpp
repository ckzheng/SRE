#include "wav.h"
#include "fstream"
#include "iostream"
#include "base/log_message.h"

namespace alspkdiar {

WaveFile::WaveFile() {
  header_ = NULL;
  data_ = NULL;
}

WaveFile::~WaveFile() {
  idec::IDEC_DELETE(header_);
  idec::IDEC_DELETE_ARRAY(data_);
}

int WaveFile::Write(const string &filename, const Wave_header &header,
                    void *data, uint32_t length) {
  ofstream ofs(filename.c_str(), ofstream::binary);
  if (!ofs) {
    ofs.close();
    idec::IDEC_ERROR << "Open file fail in WaveFile " << filename;
  }

  // Calculate size of RIFF chunk data
  header.data->cb_size = ((length + 1) / 2) * 2;
  header.riff->cb_size = 4 + 4 + header.fmt->cb_size + 4 + 4 +
                         header.data->cb_size + 4;
  // Write RIFF
  char *chunk = (char *)header.riff;
  ofs.write(chunk, sizeof(BaseChunk));

  // Write WAVE fourcc
  ofs.write((char *)&(header.wave_fcc), 4);

  // Write fmt
  chunk = (char *)header.fmt;
  ofs.write(chunk, sizeof(BaseChunk));

  // Write fmt_data
  chunk = (char *)header.fmt_data;
  ofs.write(chunk, header.fmt->cb_size);

  // Write data
  chunk = (char *)header.data;
  ofs.write(chunk, sizeof(BaseChunk));

  // Write data
  ofs.write((char *)data, length);

  ofs.close();
  return ALS_OK;
}

int WaveFile::Read(const string &filename) {
  if (!ReadHeader(filename)) {
    idec::IDEC_ERROR << "Read head fail in WaveFile " << filename;
  }
  // PCM 数据相对文件头位置的偏移量， +8（RIFF fourcc +4，size + 4）
  uint32_t offset = header_->riff->cb_size - header_->data->cb_size + 8;
  data_ = new uint8_t[header_->data->cb_size];

  ifstream ifs(filename.c_str(), ifstream::binary);
  if (!ifs) {
    ifs.close();
    idec::IDEC_ERROR << "Open file fail in WaveFile" << filename;
  }

  ifs.seekg(offset);
  ifs.read((char *) data_, header_->data->cb_size);

  return ALS_OK;
}

// Read wav file header
int WaveFile::ReadHeader(const string &filename) {
  ifstream ifs(filename.c_str(), ifstream::binary);
  if (!ifs) {
    ifs.close();
    idec::IDEC_ERROR << "Open file fail in WaveFile" << filename;
  }

  header_ = new Wave_header();

  // Read RIFF chunk
  FOURCC fourcc;
  ifs.read((char *)&fourcc, sizeof(FOURCC));
  if (fourcc != MakeFOURCC<'R', 'I', 'F', 'F'>::value) {
    ifs.close();
    idec::IDEC_ERROR << "Wave format error, not RIFF.";
  }

  BaseChunk riff_chunk(fourcc);
  ifs.read((char *)&riff_chunk.cb_size, sizeof(uint32_t));
  header_->riff = new BaseChunk(riff_chunk);
  // Read WAVE FOURCC
  ifs.read((char *)&fourcc, sizeof(FOURCC));
  if (fourcc != MakeFOURCC<'W', 'A', 'V', 'E'>::value) {
    ifs.close();
    idec::IDEC_ERROR << "Wave format error, not WAVE.";
  }

  header_->wave_fcc = fourcc;
  // Read format chunk
  ifs.read((char *)&fourcc, sizeof(FOURCC));
  if (fourcc != MakeFOURCC<'f', 'm', 't', ' '>::value) {
    ifs.close();
    idec::IDEC_ERROR << "Wave format error, not 'fmt'.";
  }

  BaseChunk fmt_chunk(fourcc);
  ifs.read((char *)&fmt_chunk.cb_size, sizeof(uint32_t));
  header_->fmt = new BaseChunk(fmt_chunk);
  // Read format data
  WaveFormat format;
  ifs.read((char *)&format, fmt_chunk.cb_size);
  // Read data chunk
  ifs.read((char *)&fourcc, sizeof(fourcc));
  if (fourcc != MakeFOURCC<'d', 'a', 't', 'a'>::value) {
    ifs.close();
    idec::IDEC_ERROR << "[ERROR] Wave format error, not 'data'.";
  }

  BaseChunk data_chunk(fourcc);
  ifs.read((char *)&data_chunk.cb_size, sizeof(uint32_t));
  header_->data = new BaseChunk(data_chunk);
  ifs.close();
  return ALS_OK;
}
}
