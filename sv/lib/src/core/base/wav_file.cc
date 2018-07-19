#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <string>
#include <sstream>
#include <cstring>
#include "wav_file.h"

// define ST_NO_EXCEPTION_HANDLING switch to disable throwing std exceptions:
// #define ST_NO_EXCEPTION_HANDLING    1
#ifdef ST_NO_EXCEPTION_HANDLING
// Exceptions disabled. Throw asserts instead if enabled.
#define ST_THROW_RT_ERROR(x)    {assert((const char *)x);}
#else
// use c++ standard exceptions
#include <stdexcept>
#define ST_THROW_RT_ERROR(x)    {throw std::runtime_error(x);}
#endif
#pragma warning(disable: 4996)

namespace idec {

static const char kRiffStr[] = "RIFF";
static const char kWaveStr[] = "WAVE";
static const char kFmtStr[]  = "fmt ";
static const char kFactStr[] = "fact";
static const char kDataStr[] = "data";

//////////////////////////////////////////////////////////////////////////////
// Helper functions for swapping byte order to correctly read/write WAV files
// with big-endian CPU's: Define compile-time defInition _BIG_ENDIAN_ to
// turn-on the conversion if it appears necessary.
//
// For example, Intel x86 is little-endian and doesn't require conversion,
// while PowerPC of Mac's and many other RISC cpu's are big-endian.

#ifdef BYTE_ORDER
// In gcc compiler detect the byte order automatically
#if BYTE_ORDER == BIG_ENDIAN
// big-endian platform.
#define _BIG_ENDIAN_
#endif
#endif

#ifdef _BIG_ENDIAN_
// big-endian CPU, swap bytes in 16 & 32 bit words

// helper-function to swap byte-order of 32bit integer
static inline int _swap32(int &dwData) {
  dwData = ((dwData >> 24) & 0x000000FF) |
           ((dwData >> 8)  & 0x0000FF00) |
           ((dwData << 8)  & 0x00FF0000) |
           ((dwData << 24) & 0xFF000000);
  return dwData;
}

// helper-function to swap byte-order of 16bit integer
static inline short _swap16(short &wData) {
  wData = ((wData >> 8) & 0x00FF) |
          ((wData << 8) & 0xFF00);
  return wData;
}

// helper-function to swap byte-order of buffer of 16bit integers
static inline void _swap16Buffer(short *pData, int numWords) {
  int i;

  for (i = 0; i < numWords; i ++) {
    pData[i] = _swap16(pData[i]);
  }
}

#else   // BIG_ENDIAN
// little-endian CPU, WAV file is ok as such

// dummy helper-function
static inline int _swap32(int &dwData) {
  // do nothing
  return dwData;
}

// dummy helper-function
static inline short _swap16(short &wData) {
  // do nothing
  return wData;
}

// dummy helper-function
static inline void _swap16Buffer(short *pData, int numBytes) {
  // do nothing
}
#endif  // BIG_ENDIAN


//////////////////////////////////////////////////////////////////////////////
// Class WavFileBase
WavFileBase::WavFileBase() {
  conv_buf_ = NULL;
  conv_buf_Size = 0;
}

WavFileBase::~WavFileBase() {
  delete[] conv_buf_;
  conv_buf_Size = 0;
}

/// Get pointer to conversion buffer of at min. given size
void *WavFileBase::GetConvBuffer(int sizeBytes) {
  if (conv_buf_Size < sizeBytes) {
    delete[] conv_buf_;
    conv_buf_Size = (sizeBytes + 15) &
                    -8;   // round up to following 8-byte bounday
    conv_buf_ = new char[conv_buf_Size];
  }
  return conv_buf_;
}

//////////////////////////////////////////////////////////////////////////////
// Class WavInFile
WavInFile::WavInFile(const char *fileName) {
  // Try to open the file for reading
  fp_wav_ = fopen(fileName, "rb");
  if (fp_wav_ == NULL) {
    // didn't succeed
    std::string msg = "Error : Unable to open file \"";
    msg += fileName;
    msg += "\" for reading.";
    ST_THROW_RT_ERROR(msg.c_str());
  }
  Init();
}

WavInFile::WavInFile(FILE *file) {
  // Try to open the file for reading
  fp_wav_ = file;
  if (!file) {
    // didn't succeed
    std::string msg = "Error : Unable to access input stream for reading";
    ST_THROW_RT_ERROR(msg.c_str());
  }
  Init();
}

/// Init the WAV file stream
void WavInFile::Init() {
  int hdrsOk;
  // assume file stream is already open
  assert(fp_wav_);

  // Read the file headers
  hdrsOk = ReadWavHeaders();
  if (hdrsOk != 0) {
    // Something didn't match in the wav file headers
    std::string msg = "Input file is corrupt or not a WAV file";
    ST_THROW_RT_ERROR(msg.c_str());
  }

  /* Ignore 'fixed' field value as 32bit signed linear data can have other value than 1.
     if (header.format.fixed != 1)
     {
     std::string msg = "Input file uses unsupported encoding.";
     ST_THROW_RT_ERROR(msg.c_str());
     }
     */

  data_read_ = 0;
}

WavInFile::~WavInFile() {
  if (fp_wav_) fclose(fp_wav_);
  fp_wav_ = NULL;
}

void WavInFile::Rewind() {
  int hdrsOk;
  fseek(fp_wav_, 0, SEEK_SET);
  hdrsOk = ReadWavHeaders();
  assert(hdrsOk == 0);
  data_read_ = 0;
}

int WavInFile::CheckCharTags() const {
  // header.format.fmt should equal to 'fmt '
  if (memcmp(kFmtStr, header_.format.fmt, 4) != 0) return -1;
  // header.data.data_field should equal to 'data'
  if (memcmp(kDataStr, header_.data.data_field, 4) != 0) return -1;

  return 0;
}

int WavInFile::Read(unsigned char *buffer, int maxElems) {
  int numBytes;
  uint32_t afterDataRead;

  // ensure it's 8 bit format
  if (header_.format.bits_per_sample != 8) {
    ST_THROW_RT_ERROR("Error: WavInFile::Read works only with 8bit samples.");
  }
  assert(sizeof(char) == 1);

  numBytes = maxElems;
  afterDataRead = data_read_ + numBytes;
  if (afterDataRead > header_.data.data_len) {
    // Don't read more samples than are marked available in header
    numBytes = (int)header_.data.data_len - (int)data_read_;
    assert(numBytes >= 0);
  }

  assert(buffer);
  numBytes = (int)fread(buffer, 1, numBytes, fp_wav_);
  data_read_ += numBytes;

  return numBytes;
}

int WavInFile::Read(short *buffer, int maxElems) {
  unsigned int afterDataRead;
  int numBytes;
  int numElems;

  assert(buffer);
  switch (header_.format.bits_per_sample) {
  case 8: {
    // 8 bit format
    unsigned char *temp = (unsigned char *)GetConvBuffer(maxElems);
    int i;

    numElems = Read(temp, maxElems);
    // convert from 8 to 16 bit
    for (i = 0; i < numElems; i ++) {
      buffer[i] = (short)(((short)temp[i] - 128) * 256);
    }
    break;
  }

  case 16: {
    // 16 bit format

    assert(sizeof(short) == 2);

    numBytes = maxElems * 2;
    afterDataRead = data_read_ + numBytes;
    if (afterDataRead > header_.data.data_len) {
      // Don't read more samples than are marked available in header
      numBytes = (int)header_.data.data_len - (int)data_read_;
      assert(numBytes >= 0);
    }

    numBytes = (int)fread(buffer, 1, numBytes, fp_wav_);
    data_read_ += numBytes;
    numElems = numBytes / 2;

    // 16bit samples, swap byte order if necessary
    _swap16Buffer((short *)buffer, numElems);
    break;
  }

  default: {
    std::stringstream ss;
    ss << "\nOnly 8/16 bit sample WAV files supported in integer compilation. Can't open WAV file with ";
    ss << (int)header_.format.bits_per_sample;
    ss << " bit sample format. ";
    ST_THROW_RT_ERROR(ss.str().c_str());
  }
  }

  return numElems;
}

/// Read data in float format. Notice that when reading in float format
/// 8/16/24/32 bit sample formats are supported
int WavInFile::Read(float *buffer, int maxElems) {
  unsigned int afterDataRead;
  int numBytes;
  int numElems;
  int bytesPerSample;

  assert(buffer);

  bytesPerSample = header_.format.bits_per_sample / 8;
  if ((bytesPerSample < 1) || (bytesPerSample > 4)) {
    std::stringstream ss;
    ss << "\nOnly 8/16/24/32 bit sample WAV files supported. Can't open WAV file with ";
    ss << (int)header_.format.bits_per_sample;
    ss << " bit sample format. ";
    ST_THROW_RT_ERROR(ss.str().c_str());
  }

  numBytes = maxElems * bytesPerSample;
  afterDataRead = data_read_ + numBytes;
  if (afterDataRead > header_.data.data_len) {
    // Don't read more samples than are marked available in header
    numBytes = (int)header_.data.data_len - (int)data_read_;
    assert(numBytes >= 0);
  }

  // read raw data into temporary buffer
  char *temp = (char *)GetConvBuffer(numBytes);
  numBytes = (int)fread(temp, 1, numBytes, fp_wav_);
  data_read_ += numBytes;

  numElems = numBytes / bytesPerSample;

  // swap byte ordert & convert to float, depending on sample format
  switch (bytesPerSample) {
  case 1: {
    unsigned char *temp2 = (unsigned char *)temp;
    double conv = 1.0 / 128.0;
    for (int i = 0; i < numElems; i ++) {
      buffer[i] = (float)(temp2[i] * conv - 1.0);
    }
    break;
  }

  case 2: {
    short *temp2 = (short *)temp;
    double conv = 1.0 / 32768.0;
    for (int i = 0; i < numElems; i ++) {
      short value = temp2[i];
      buffer[i] = (float)(_swap16(value) * conv);
    }
    break;
  }

  case 3: {
    char *temp2 = reinterpret_cast<char *>(temp);
    double conv = 1.0 / 8388608.0;
    for (int i = 0; i < numElems; i ++) {
      int value = *((int *)temp2);
      value = _swap32(value) & 0x00ffffff;  // take 24 bits
      // extend minus sign bits
      value |= (value & 0x00800000) ? 0xff000000 : 0;
      buffer[i] = (float)(value * conv);
      temp2 += 3;
    }
    break;
  }

  case 4: {
    int *temp2 = (int *)temp;
    double conv = 1.0 / 2147483648.0;
    assert(sizeof(int) == 4);
    for (int i = 0; i < numElems; i ++) {
      int value = temp2[i];
      buffer[i] = (float)(_swap32(value) * conv);
    }
    break;
  }
  }

  return numElems;
}

int WavInFile::eof() const {
  // return true if all data has been read or file eof has reached
  return (data_read_ == header_.data.data_len || feof(fp_wav_));
}

// test if character code is between a white space ' ' and little 'z'
static int isAlpha(char c) {
  return (c >= ' ' && c <= 'z') ? 1 : 0;
}

// test if all characters are between a white space ' ' and little 'z'
static int isAlphaStr(const char *str) {
  char c;
  c = str[0];
  while (c) {
    if (isAlpha(c) == 0) return 0;
    str ++;
    c = str[0];
  }

  return 1;
}


int WavInFile::ReadRiffBlock() {
  if (fread(&(header_.riff), sizeof(WavRiff), 1, fp_wav_) != 1) return -1;

  // swap 32bit data byte order if necessary
  _swap32((int &)header_.riff.package_len);

  // header.riff.riff_char should equal to 'RIFF');
  if (memcmp(kRiffStr, header_.riff.riff_char, 4) != 0) return -1;
  // header.riff.wave should equal to 'WAVE'
  if (memcmp(kWaveStr, header_.riff.wave, 4) != 0) return -1;

  return 0;
}

int WavInFile::ReadHeaderBlock() {
  char label[5];
  std::string sLabel;

  // lead label string
  if (fread(label, 1, 4, fp_wav_) !=4) return -1;
  label[4] = 0;

  if (isAlphaStr(label) == 0) return -1;    // not a valid label

  // Decode blocks according to their label
  if (strcmp(label, kFmtStr) == 0) {
    int nLen, nDump;

    // 'fmt ' block
    memcpy(header_.format.fmt, kFmtStr, 4);

    // read length of the format field
    if (fread(&nLen, sizeof(int), 1, fp_wav_) != 1) return -1;
    // swap byte order if necessary
    _swap32(nLen); // int format_len;
    header_.format.format_len = nLen;

    // calculate how much length differs from expected
    nDump = nLen - ((int)sizeof(header_.format) - 8);

    // if format_len is larger than expected, read only as much data as we've space for
    if (nDump > 0) {
      nLen = sizeof(header_.format) - 8;
    }

    // read data
    if (fread(&(header_.format.fixed), nLen, 1, fp_wav_) != 1) return -1;

    // swap byte order if necessary
    _swap16(header_.format.fixed);            // short int fixed;
    _swap16(header_.format.channel_number);   // short int channel_number;
    _swap32((int &)header_.format.sample_rate);      // int sample_rate;
    _swap32((int &)header_.format.byte_rate);        // int byte_rate;
    _swap16(header_.format.byte_per_sample);  // short int byte_per_sample;
    _swap16(header_.format.bits_per_sample);  // short int bits_per_sample;

    // if format_len is larger than expected, skip the extra data
    if (nDump > 0) {
      fseek(fp_wav_, nDump, SEEK_CUR);
    }
    return 0;
  } else if (strcmp(label, kFactStr) == 0) {
    int nLen, nDump;

    // 'fact' block
    memcpy(header_.fact.fact_field, kFactStr, 4);

    // read length of the fact field
    if (fread(&nLen, sizeof(int), 1, fp_wav_) != 1) return -1;
    // swap byte order if necessary
    _swap32(nLen); // int fact_len;
    header_.fact.fact_len = nLen;

    // calculate how much length differs from expected
    nDump = nLen - ((int)sizeof(header_.fact) - 8);

    // if format_len is larger than expected, read only as much data as we've space for
    if (nDump > 0) {
      nLen = sizeof(header_.fact) - 8;
    }

    // read data
    if (fread(&(header_.fact.fact_sample_len), nLen, 1, fp_wav_) != 1) return -1;

    // swap byte order if necessary
    _swap32((int &)header_.fact.fact_sample_len);    // int sample_length;

    // if fact_len is larger than expected, skip the extra data
    if (nDump > 0) {
      fseek(fp_wav_, nDump, SEEK_CUR);
    }

    return 0;
  } else if (strcmp(label, kDataStr) == 0) {
    // 'data' block
    memcpy(header_.data.data_field, kDataStr, 4);
    if (fread(&(header_.data.data_len), sizeof(uint32_t), 1,
              fp_wav_) != 1) return -1;

    // swap byte order if necessary
    _swap32((int &)header_.data.data_len);

    return 1;
  } else {
    uint32_t len, i;
    uint32_t temp;
    // unknown block

    // read length
    if (fread(&len, sizeof(len), 1, fp_wav_) != 1) return -1;
    // scan through the block
    for (i = 0; i < len; i ++) {
      if (fread(&temp, 1, 1, fp_wav_) != 1) return -1;
      if (feof(fp_wav_)) return -1;   // unexpected eof
    }
  }
  return 0;
}

int WavInFile::ReadWavHeaders() {
  int res;

  memset(&header_, 0, sizeof(header_));

  res = ReadRiffBlock();
  if (res) return 1;
  // read header blocks until data block is found
  do {
    // read header blocks
    res = ReadHeaderBlock();
    if (res < 0) return 1;  // error in file structure
  } while (res == 0);
  // check that all required tags are legal
  return CheckCharTags();
}

uint32_t WavInFile::num_channels() const {
  return header_.format.channel_number;
}

uint32_t WavInFile::num_bits() const {
  return header_.format.bits_per_sample;
}

uint32_t WavInFile::bytes_per_sample() const {
  return num_channels() * num_bits() / 8;
}

uint32_t WavInFile::sample_rate() const {
  return header_.format.sample_rate;
}

uint32_t WavInFile::datasize_in_bytes() const {
  return header_.data.data_len;
}

uint32_t WavInFile::num_samples() const {
  if (header_.format.byte_per_sample == 0) return 0;
  if (header_.format.fixed > 1) return header_.fact.fact_sample_len;
  return header_.data.data_len / (unsigned short)header_.format.byte_per_sample;
}

uint32_t WavInFile::length_ms() const {
  double numSamples;
  double sampleRate;
  numSamples = (double)num_samples();
  sampleRate = (double)sample_rate();
  return (uint32_t)(1000.0 * numSamples / sampleRate + 0.5);
}

// Returns how many milliseconds of audio have so far been read from the file
uint32_t WavInFile::elapsed_ms() const {
  return (uint32_t)(1000.0 * (double)data_read_ / (double)
                    header_.format.byte_rate);
}

//////////////////////////////////////////////////////////////////////////////
// Class WavOutFile
WavOutFile::WavOutFile(const char *fileName, int sampleRate, int bits,
                       int channels) {
  bytes_written_ = 0;
  fp_wav_ = fopen(fileName, "wb");
  if (fp_wav_ == NULL) {
    std::string msg = "Error : Unable to open file \"";
    msg += fileName;
    msg += "\" for writing.";
    //pmsg = msg.c_str;
    ST_THROW_RT_ERROR(msg.c_str());
  }

  FillInHeader(sampleRate, bits, channels);
  WriteHeader();
}

WavOutFile::WavOutFile(FILE *file, int sampleRate, int bits, int channels) {
  bytes_written_ = 0;
  fp_wav_ = file;
  if (fp_wav_ == NULL) {
    std::string msg = "Error : Unable to access output file stream.";
    ST_THROW_RT_ERROR(msg.c_str());
  }
  FillInHeader(sampleRate, bits, channels);
  WriteHeader();
}

WavOutFile::~WavOutFile() {
  FinishHeader();
  if (fp_wav_) {
    fclose(fp_wav_);
    fp_wav_ = NULL;
  }
}

void WavOutFile::FillInHeader(uint32_t sampleRate, uint32_t bits,
                              uint32_t channels) {
  // fill in the 'riff' part..
  // copy string 'RIFF' to riff_char
  memcpy(&(header_.riff.riff_char), kRiffStr, 4);
  // package_len unknown so far
  header_.riff.package_len = 0;
  // copy string 'WAVE' to wave
  memcpy(&(header_.riff.wave), kWaveStr, 4);

  // fill in the 'format' part..
  // copy string 'fmt ' to fmt
  memcpy(&(header_.format.fmt), kFmtStr, 4);

  header_.format.format_len = 0x10;
  header_.format.fixed = 1;
  header_.format.channel_number = (short)channels;
  header_.format.sample_rate = (int)sampleRate;
  header_.format.bits_per_sample = (short)bits;
  header_.format.byte_per_sample = (short)(bits * channels / 8);
  header_.format.byte_rate = header_.format.byte_per_sample * (int)sampleRate;
  header_.format.sample_rate = (int)sampleRate;

  // fill in the 'fact' part...
  memcpy(&(header_.fact.fact_field), kFactStr, 4);
  header_.fact.fact_len = 4;
  header_.fact.fact_sample_len = 0;

  // fill in the 'data' part..
  // copy string 'data' to data_field
  memcpy(&(header_.data.data_field), kDataStr, 4);
  // data_len unknown so far
  header_.data.data_len = 0;
}

void WavOutFile::FinishHeader() {
  // supplement the file length into the header structure
  header_.riff.package_len = bytes_written_ + sizeof(WavHeader) - sizeof(
                               WavRiff) + 4;
  header_.data.data_len = bytes_written_;
  header_.fact.fact_sample_len = bytes_written_ / header_.format.byte_per_sample;
  WriteHeader();
}

void WavOutFile::WriteHeader() {
  WavHeader hdrTemp;
  int res;

  // swap byte order if necessary
  hdrTemp = header_;
  _swap32((int &)hdrTemp.riff.package_len);
  _swap32((int &)hdrTemp.format.format_len);
  _swap16((short &)hdrTemp.format.fixed);
  _swap16((short &)hdrTemp.format.channel_number);
  _swap32((int &)hdrTemp.format.sample_rate);
  _swap32((int &)hdrTemp.format.byte_rate);
  _swap16((short &)hdrTemp.format.byte_per_sample);
  _swap16((short &)hdrTemp.format.bits_per_sample);
  _swap32((int &)hdrTemp.data.data_len);
  _swap32((int &)hdrTemp.fact.fact_len);
  _swap32((int &)hdrTemp.fact.fact_sample_len);

  // write the supplemented header in the beginning of the file
  fseek(fp_wav_, 0, SEEK_SET);
  res = (int)fwrite(&hdrTemp, sizeof(hdrTemp), 1, fp_wav_);
  if (res != 1) {
    ST_THROW_RT_ERROR("Error while writing to a wav file.");
  }

  // jump back to the end of the file
  fseek(fp_wav_, 0, SEEK_END);
}

void WavOutFile::Write(const unsigned char *buffer, int numElems) {
  int res;

  if (header_.format.bits_per_sample != 8) {
    ST_THROW_RT_ERROR("Error: WavOutFile::write(const char*, int) accepts only 8bit samples.");
  }
  assert(sizeof(char) == 1);

  res = (int)fwrite(buffer, 1, numElems, fp_wav_);
  if (res != numElems) {
    ST_THROW_RT_ERROR("Error while writing to a wav file.");
  }

  bytes_written_ += numElems;
}

void WavOutFile::Write(const short *buffer, int numElems) {
  int res;

  // 16 bit samples
  if (numElems < 1) return;   // nothing to do

  switch (header_.format.bits_per_sample) {
  case 8: {
    int i;
    unsigned char *temp = (unsigned char *)GetConvBuffer(numElems);
    // convert from 16bit format to 8bit format
    for (i = 0; i < numElems; i ++) {
      temp[i] = (unsigned char)(buffer[i] / 256 + 128);
    }
    // write in 8bit format
    Write(temp, numElems);
    break;
  }

  case 16: {
    // 16bit format

    // use temp buffer to swap byte order if necessary
    short *pTemp = (short *)GetConvBuffer(numElems * sizeof(short));
    memcpy(pTemp, buffer, numElems * 2);
    _swap16Buffer(pTemp, numElems);

    res = (int)fwrite(pTemp, 2, numElems, fp_wav_);

    if (res != numElems) {
      ST_THROW_RT_ERROR("Error while writing to a wav file.");
    }
    bytes_written_ += 2 * numElems;
    break;
  }

  default: {
    std::stringstream ss;
    ss << "\nOnly 8/16 bit sample WAV files supported in integer compilation. Can't open WAV file with ";
    ss << (int)header_.format.bits_per_sample;
    ss << " bit sample format. ";
    ST_THROW_RT_ERROR(ss.str().c_str());
  }
  }
}


/// Convert from float to integer and saturate
inline int saturate(float fvalue, float minval, float maxval) {
  if (fvalue > maxval) {
    fvalue = maxval;
  } else if (fvalue < minval) {
    fvalue = minval;
  }
  return (int)fvalue;
}

void WavOutFile::Write(const float *buffer, int numElems) {
  int numBytes;
  int bytesPerSample;

  if (numElems == 0) return;

  bytesPerSample = header_.format.bits_per_sample / 8;
  numBytes = numElems * bytesPerSample;
  short *temp = (short *)GetConvBuffer(numBytes);

  switch (bytesPerSample) {
  case 1: {
    unsigned char *temp2 = (unsigned char *)temp;
    for (int i = 0; i < numElems; i ++) {
      temp2[i] = (unsigned char)saturate(buffer[i] * 128.0f + 128.0f, 0.0f, 255.0f);
    }
    break;
  }

  case 2: {
    short *temp2 = (short *)temp;
    for (int i = 0; i < numElems; i ++) {
      short value = (short)saturate(buffer[i] * 32768.0f, -32768.0f, 32767.0f);
      temp2[i] = _swap16(value);
    }
    break;
  }

  case 3: {
    char *temp2 = (char *)temp;
    for (int i = 0; i < numElems; i ++) {
      int value = saturate(buffer[i] * 8388608.0f, -8388608.0f, 8388607.0f);
      *((int *)temp2) = _swap32(value);
      temp2 += 3;
    }
    break;
  }

  case 4: {
    int *temp2 = (int *)temp;
    for (int i = 0; i < numElems; i ++) {
      int value = saturate(buffer[i] * 2147483648.0f, -2147483648.0f, 2147483647.0f);
      temp2[i] = _swap32(value);
    }
    break;
  }

  default:
    assert(false);
  }

  int res = (int)fwrite(temp, 1, numBytes, fp_wav_);

  if (res != numBytes) {
    ST_THROW_RT_ERROR("Error while writing to a wav file.");
  }
  bytes_written_ += numBytes;
}

};  // namespace idec

