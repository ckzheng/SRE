#ifndef ASR_DECODER_SRC_CORE_BASE_WAV_FILE_H_
#define ASR_DECODER_SRC_CORE_BASE_WAV_FILE_H_

#include <stdio.h>

namespace idec {

#ifndef uint32_t
typedef unsigned int uint32_t;
#endif

// WAV audio file 'riff' section header
typedef struct {
  char riff_char[4];
  int  package_len;
  char wave[4];
} WavRiff;

// WAV audio file 'format' section header
typedef struct {
  char fmt[4];
  int format_len;
  short fixed;
  short channel_number;
  int sample_rate;
  int byte_rate;
  short byte_per_sample;
  short bits_per_sample;
} WavFormat;

// WAV audio file 'fact' section header
typedef struct {
  char fact_field[4];
  int fact_len;
  uint32_t fact_sample_len;
} WavFact;

// WAV audio file 'data' section header
typedef struct {
  char data_field[4];
  uint32_t data_len;
} WavData;

// WAV audio file header
typedef struct {
  WavRiff riff;
  WavFormat format;
  WavFact fact;
  WavData data;
} WavHeader;

// Base class for processing WAV audio files.
class WavFileBase {
 protected:
  WavFileBase();
  virtual ~WavFileBase();
  // Get pointer to conversion buffer of at min. given size
  void *GetConvBuffer(int size_byte);

 private:
  // Conversion working buffer;
  char *conv_buf_;
  int conv_buf_Size;
};

// Class for reading WAV audio files.
class WavInFile : protected WavFileBase {
 public:
  // Constructor: Opens the given WAV file. If the file can't be opened,
  // throws 'runtime_error' exception.
  explicit WavInFile(const char *file_name);
  explicit WavInFile(FILE *file);
  ~WavInFile();  // Destructor: Closes the file.
  void Rewind();  // Rewind to beginning of the file
  uint32_t sample_rate() const;  // Get sample rate.
  uint32_t num_bits() const;  // Get number of bits per sample, i.e. 8 or 16.

  // Get sample data size in bytes. Ahem, this should return same information
  // as 'bytes_per_sample'...
  uint32_t datasize_in_bytes() const;
  uint32_t num_samples() const;  // Get total number of samples in file.

  // Get number of bytes per audio sample (e.g. 16bit stereo = 4 bytes/sample)
  uint32_t bytes_per_sample() const;

  // Get number of audio channels in the file (1=mono, 2=stereo)
  uint32_t num_channels() const;

  // Get the audio file length in milliseconds
  uint32_t length_ms() const;

  // Returns how many milliseconds of audio have so far been read from the file
  // \return elapsed duration in milliseconds
  uint32_t elapsed_ms() const;

  // Reads audio samples from the WAV file.
  // This routine works only for 8 bit samples.
  // Reads given number of elements from the file or if end-of-file reached,
  // as many elements as are left in the file.
  //
  // \return Number of 8-bit integers read from the file.
  int Read(unsigned char *buffer, int max_elems);

  // Reads audio samples from the WAV file to 16 bit integer format.
  // Reads given number of elements from the file or if end-of-file reached,
  // as many elements as are left in the file.
  // \return Number of 16-bit integers read from the file.
  // @1: Pointer to buffer where to read data.
  // @2: Size of 'buffer' array (number of array elements).
  int Read(short *buffer, int max_elems);

  // Reads audio samples from the WAV file to floating point format, converting
  // sample values to range [-1,1[. Reads given number of elements from the file
  // or if end-of-file reached, as many elements as are left in the file.
  // Notice that reading in float format supports 8/16/24/32bit sample formats.
  //
  // \return Number of elements read from the file.
  int Read(float *buffer, int max_elems);

  // Check end-of-file.
  //
  // \return Nonzero if end-of-file reached.
  int eof() const;

 private:
  FILE *fp_wav_;  // File pointer.

  // Counter of how many bytes of sample data have been read from the file.
  long data_read_;

  // WAV header information
  WavHeader header_;

  // Init the WAV file stream
  void Init();

  // Read WAV file headers.
  // \return zero if all ok, nonzero if file format is invalid.
  int ReadWavHeaders();

  // Checks WAV file header tags.
  // \return zero if all ok, nonzero if file format is invalid.
  int CheckCharTags() const;

  // Reads a single WAV file header block.
  // \return zero if all ok, nonzero if file format is invalid.
  int ReadHeaderBlock();

  // Reads WAV file 'riff' block
  int ReadRiffBlock();
};

// Class for writing WAV audio files.
class WavOutFile : protected WavFileBase {
 public:
  // Constructor: Creates a new WAV file. Throws a 'runtime_error' exception
  // if file creation fails.
  WavOutFile(const char *file_name, int sample_rate, int bits, int channels);
  WavOutFile(FILE *file, int sample_rate, int bits, int channels);

  // Destructor: Finalizes & closes the WAV file.
  ~WavOutFile();

  // Write data to WAV file. This function works only with 8bit samples.
  // Throws a 'runtime_error' exception if writing to file fails.
  void Write(const unsigned char *buffer, int num_elems);

  // Write data to WAV file. Throws a 'runtime_error' exception if writing to
  // file fails.
  void Write(const short *buffer, int num_elems);

  // Write data to WAV file in floating point format, saturating sample values
  // to range [-1..+1[. Throws a 'runtime_error' exception if writing to file
  // fails.
  // @1: Pointer to sample data buffer.
  // @2: How many array items are to be written to file.
  void Write(const float *buffer, int num_elems);

 private:
  // Pointer to the WAV file
  FILE *fp_wav_;

  // WAV file header data.
  WavHeader header_;

  // Counter of how many bytes have been written to the file so far.
  int bytes_written_;

  // Fills in WAV file header information.
  void FillInHeader(const uint32_t sample_rate, const uint32_t bits,
                    const uint32_t channels);

  // Finishes the WAV file header by supplementing information of amount of
  // data written to file etc
  void FinishHeader();

  // Writes the WAV file header.
  void WriteHeader();
};

};  // namespace idec

#endif  // ASR_DECODER_SRC_CORE_BASE_WAV_FILE_H_

