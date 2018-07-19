#ifndef _WAV_H_
#define _WAV_H_

#include <string>
#include <memory>
#include "als_error.h"
#include "base/idec_common.h"

namespace alspkdiar {
using namespace std;

#define FOURCC uint32_t

#define MAKE_FOURCC(a,b,c,d) \
( ((uint32_t)d) | ( ((uint32_t)c) << 8 ) | ( ((uint32_t)b) << 16 ) | ( ((uint32_t)a) << 24 ) )

template <char ch0, char ch1, char ch2, char ch3> struct MakeFOURCC { enum { value = (ch0 << 0) + (ch1 << 8) + (ch2 << 16) + (ch3 << 24) }; };


// Format chunk data field
struct WaveFormat {
  uint16_t format_tag;      // WAVE的数据格式，PCM数据该值为1
  uint16_t channels;        // 声道数
  uint32_t sample_per_sec;  // 采样率
  uint32_t bytes_per_sec;   // 码率，channels * sample_per_sec * bits_per_sample / 8
  uint16_t block_align;     // 音频数据块，每次采样处理的数据大小，channels * bits_per_sample / 8
  uint16_t bits_per_sample; // 量化位数，8、16、32等
  //uint16_t ex_size;         // 扩展块的大小，附加块的大小

  WaveFormat() {
    format_tag = 1; // PCM format data
    //ex_size = 0; // don't use extesion field
    channels = 0;
    sample_per_sec = 0;
    bytes_per_sec = 0;
    block_align = 0;
    bits_per_sample = 0;
  }

  WaveFormat(uint16_t nb_channel, uint32_t sample_rate, uint16_t sample_bits)
    :channels(nb_channel), sample_per_sec(sample_rate),
     bits_per_sample(sample_bits) {
    format_tag = 0x01;                                           // PCM format data
    bytes_per_sec = channels * sample_per_sec * bits_per_sample / 8; // 码率
    block_align = channels * bits_per_sample / 8;
    //ex_size = 0;                                               // don't use extension field
  }
};

// The basic chunk of RIFF file format
struct BaseChunk {
  FOURCC fcc;    // FourCC id
  uint32_t cb_size; // 数据域的大小
  BaseChunk(FOURCC fourcc)
    : fcc(fourcc) {
    cb_size = 0;
  }
};

/*
数据格式为PCM的WAV文件的基本结构
--------------------------------
| BaseChunk | RIFF	|
---------------------
|	WAVE            |
---------------------
| BaseChunk | fmt  |	Header
---------------------
| WaveFormat|      |
---------------------
| BaseChunk | data |
---------------------------------
|    PCM data                   |
---------------------------------
*/

/*
数据格式为PCM的WAV文件头
--------------------------------
| BaseChunk | RIFF	|
---------------------
|	WAVE            |
---------------------
| BaseChunk | fmt  |	Header
---------------------
| WaveFormat|      |
---------------------
| BaseChunk | data |
--------------------------------
*/

struct Wave_header {
  BaseChunk *riff;
  FOURCC wave_fcc;
  BaseChunk *fmt;
  WaveFormat *fmt_data;
  BaseChunk *data;

  Wave_header(uint16_t nb_channel, uint32_t sample_rate, uint16_t sample_bits) {
    riff = new BaseChunk(MakeFOURCC<'R', 'I', 'F', 'F'>::value);
    fmt = new BaseChunk(MakeFOURCC<'f', 'm', 't', ' '>::value);
    fmt->cb_size = 16;
    fmt_data = new WaveFormat(nb_channel, sample_rate, sample_bits);
    data = new BaseChunk(MakeFOURCC<'d', 'a', 't', 'a'>::value);
    wave_fcc = MakeFOURCC<'W', 'A', 'V', 'E'>::value;
  }

  ~Wave_header() {
    idec::IDEC_DELETE(riff);
    idec::IDEC_DELETE(fmt);
    idec::IDEC_DELETE(fmt_data);
    idec::IDEC_DELETE(data);
  }

  Wave_header() {
    riff = NULL;
    fmt = NULL;
    fmt_data = NULL;
    data = NULL;
    wave_fcc = 0;
  }
};

class WaveFile {
 public:
  WaveFile();
  ~WaveFile();
  // Write wav file
  int Write(const string &filename, const Wave_header &header, void *data,
            uint32_t length);
  int Read(const string &filename);
 private:
  // Read wav file header
  int ReadHeader(const string &filename);
 public:
  Wave_header *header_;
  uint8_t *data_;
};
}

#endif
