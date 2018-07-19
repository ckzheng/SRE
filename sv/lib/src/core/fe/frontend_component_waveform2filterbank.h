#ifndef FE_RONTEND_COMPONENT_WAVEFORM2FILTERBANK_H_
#define FE_RONTEND_COMPONENT_WAVEFORM2FILTERBANK_H_
#include "fe/frontend_component.h"
#include "util/random.h"

namespace idec {

class FrontendComponent_Waveform2Filterbank : public
  FrontendComponentInterface {

 private:
#ifdef _MSC_VER
#  define IDEC_ISNAN _isnan
#  define IDEC_ISINF(x) (!_isnan(x) && _isnan(x-x))
#  define IDEC_ISFINITE _finite
#else
#  define IDEC_ISNAN std::isnan
#  define IDEC_ISINF std::isinf
#  define IDEC_ISFINITE(x) std::isfinite(x)
#endif


  /*****************wav2mfcc parameters struct define  begin*******************/
  struct MelBanksOptions {
    int num_bins;  // e.g. 25; number of triangular bins
    float low_freq;  // e.g. 20; lower frequency cutoff
    float high_freq;  // an upper frequency cutoff; 0 -> no cutoff, negative
    // ->added to the Nyquist frequency to get the cutoff.
    float vtln_low;  // vtln lower cutoff of warping function.
    float vtln_high;  // vtln upper cutoff of warping function: if negative, added
    // to the Nyquist frequency to get the cutoff.
    bool debug_mel;
    // htk_mode is a "hidden" config, it does not show up on command line.
    // Enables more exact compatibibility with HTK, for testing purposes.  Affects
    // mel-energy flooring and reproduces a bug in HTK.
    bool htk_mode;

    explicit  MelBanksOptions() :
      num_bins(23),
      low_freq(20),
      high_freq(0),
      vtln_low(100),
      vtln_high(-500),
      debug_mel(false),
      htk_mode(false) {
    }

    void Register(OptionsItf *po, const std::string name) {
      po->Register(name + "::num-mel-bins", &num_bins,
                   "Number of triangular mel-frequency bins");
      po->Register(name + "::low-freq", &low_freq,
                   "Low cutoff frequency for mel bins");
      po->Register(name + "::high-freq", &high_freq,
                   "High cutoff frequency for mel bins (if < 0, offset from Nyquist)");
      po->Register(name + "::vtln-low", &vtln_low,
                   "Low inflection point in piecewise linear VTLN warping function");
      po->Register(name + "::vtln-high", &vtln_high,
                   "High inflection point in piecewise linear VTLN warping function"
                   " (if negative, offset from high-mel-freq");
      po->Register(name + "::debug-mel", &debug_mel,
                   "Print out debugging information for mel bin computation");
    }


  };


  struct FrameExtractionOptions {
    float samp_freq;
    float frame_shift_ms;      // in milliseconds.
    float frame_length_ms;     // in milliseconds.
    float dither;              // Amount of dithering, 0.0 means no dither.
    bool  deterministic_dither;// make the dither "deterministic" random sequence
    float preemph_coeff;       // Preemphasis coefficient.
    bool remove_dc_offset;     // Subtract mean of wave before FFT.
    std::string window_type;   // e.g. Hamming window
    bool round_to_power_of_two;
    bool snip_edges;
    // Maybe "hamming", "rectangular", "povey", "hanning"
    // "povey" is a window I made to be similar to Hamming but to go to zero at the
    // edges, it's pow((0.5 - 0.5*cos(n/N*2*pi)), 0.85)
    // I just don't think the Hamming window makes sense as a windowing function.
    FrameExtractionOptions() :
      samp_freq(8000),
      frame_shift_ms(10.0f),
      frame_length_ms(25.0f),
      dither(1.0f),
      deterministic_dither(true),
      preemph_coeff(0.97f),
      remove_dc_offset(true),
      window_type("povey"),
      round_to_power_of_two(true),
      snip_edges(true) {
    }


    void Register(OptionsItf *po, const std::string name) {
      po->Register(name + "::sample-frequency", &samp_freq,
                   "Waveform data sample frequency (must match the waveform file, "
                   "if specified there)");
      po->Register(name + "::frame-length", &frame_length_ms,
                   "Frame length in milliseconds");
      po->Register(name + "::frame-shift", &frame_shift_ms,
                   "Frame shift in milliseconds");
      po->Register(name + "::preemphasis-coefficient", &preemph_coeff,
                   "Coefficient for use in signal preemphasis");
      po->Register(name + "::remove-dc-offset", &remove_dc_offset,
                   "Subtract mean from waveform on each frame");
      po->Register(name + "::dither", &dither,
                   "Dithering constant (0.0 means no dither)");
      po->Register(name + "::deterministic-dither", &deterministic_dither,
                   "Dithering should be deterministic_dither");
      po->Register(name + "::window-type", &window_type, "Type of window "
                   "(\"hamming\"|\"hanning\"|\"povey\"|\"rectangular\")");
      po->Register(name + "::round-to-power-of-two", &round_to_power_of_two,
                   "If true, round window size to power of two.");
      po->Register(name + "::snip-edges", &snip_edges,
                   "If true, end effects will be handled by outputting only frames that "
                   "completely fit in the file, and the number of frames depends on the "
                   "frame-length.  If false, the number of frames depends only on the "
                   "frame-shift, and we reflect the data at the ends.");
    }

    int WindowShift() const {
      return static_cast<int>(samp_freq * 0.001 * frame_shift_ms);
    }

    int WindowSize() const {
      return static_cast<int>(samp_freq * 0.001 * frame_length_ms);
    }

  };


  struct MfccOptions {
    FrameExtractionOptions frame_opts;
    MelBanksOptions mel_opts;
    int num_ceps_;  // e.g. 13: num cepstral coeffs, counting zero.
    bool use_energy;  // use energy; else C0
    float energy_floor;
    bool raw_energy;  // If true, compute energy before preemphasis and windowing
    float cepstral_lifter_;  // Scaling factor on cepstra for HTK compatibility.
    // if 0.0, no liftering is done.
    bool htk_compat;  // if true, put energy/C0 last and introduce a factor of
    // sqrt(2) on C0 to be the same as HTK.

    MfccOptions() : //mel_opts(23),
      // defaults the #mel-banks to 23 for the MFCC computations.
      // this seems to be common for 16khz-sampled data,
      // but for 8khz-sampled data, 15 may be better.
      num_ceps_(13),
      use_energy(true),
      energy_floor(0.0f),  // not in log scale: a small value e.g. 1.0e-10
      raw_energy(true),
      cepstral_lifter_(22.0f),
      htk_compat(false) {
    }


    void Register(OptionsItf *po, const std::string name) {
      frame_opts.Register(po, name);
      mel_opts.Register(po, name);
      po->Register(name + "::energy-floor", &energy_floor,
                   "Floor on energy (absolute, not relative) in MFCC computation");
      po->Register(name + "::raw-energy", &raw_energy,
                   "If true, compute energy before preemphasis and windowing");

      po->Register(name + "::htk-compat", &htk_compat,
                   "If true, put energy or C0 last and use a factor of sqrt(2) on "
                   "C0.  Warning: not sufficient to get HTK compatible features "
                   "(need to change other parameters).");
    }

  };
  /**************wav2mfcc parameters struct define  end*********************/

  class MelBanks {
   public:

    static inline int RoundUpToNearestPowerOfTwo_for_mel(int n) {
      IDEC_ASSERT(n > 0);
      n--;
      n |= n >> 1;
      n |= n >> 2;
      n |= n >> 4;
      n |= n >> 8;
      n |= n >> 16;
      return n + 1;
    }

    //template<typename Real> static inline  Real vec_dot_for_mel(const std::vector<Real> &a, const std::vector<Real> &b)
    //{
    //    //size_t dim = a.size();
    //    if (a.size() != b.size())
    //    {
    //        IDEC_ERROR << "dim of two input not eq\n";
    //    }
    //    Real out = 0;
    //    for (int i = 0; i < (int)a.size(); i++)
    //    {
    //        out += a[i] * b[i];
    //    }
    //    return out;
    //}

    static inline float InverseMelScale(float mel_freq) {
      return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
    }

    static inline float MelScale(float freq) {
      return 1127.0f * logf(1.0f + freq / 700.0f);
    }

    static float VtlnWarpFreq(float vtln_low_cutoff,
                              float vtln_high_cutoff,  // discontinuities in warp func
                              float low_freq,
                              float high_freq,  // upper+lower frequency cutoffs in
                              // the mel computation
                              float vtln_warp__factor,
                              float freq);

    static float VtlnWarpMelFreq(float vtln_low_cutoff,
                                 float vtln_high_cutoff,
                                 float low_freq,
                                 float high_freq,
                                 float vtln_warp__factor,
                                 float mel_freq);


    MelBanks(const MelBanksOptions &opts,
             const FrameExtractionOptions &frame_opts,
             float vtln_warp__factor);

    /// Compute Mel energies (note: not log enerties).
    /// At input, "fft_energies" contains the FFT energies (not log).
    void Compute(const std::vector<float> &fft_energies,
                 std::vector<float> &mel_energies__out) const;

    size_t NumBins() const { return bins_.size(); }

    // returns vector of central freq of each bin; needed by plp code.
    const std::vector<float> &GetCenterFreqs() const { return center_freqs_; }

   private:
    // center frequencies of bins, numbered from 0 ... num_bins-1.
    // Needed by GetCenterFreqs().
    std::vector<float> center_freqs_;

    // the "bins_" vector is a vector, one for each bin, of a pair:
    // (the first nonzero fft-bin), (the vector of weights).
    std::vector<std::pair<int32, std::vector<float> > > bins_;

    bool debug_;
    bool htk_mode_;
    //KALDI_DISALLOW_COPY_AND_ASSIGN(MelBanks);
  };

  template<typename Real>
  class SplitRadixComplexFft {
   public:
    typedef int Integer;
    typedef int MatrixIndexT;

    // N is the number of complex points (must be a power of two, or this
    // will crash).  Note that the constructor does some work so it's best to
    // initialize the object once and do the computation many times.
    SplitRadixComplexFft(Integer N);

    // Does the FFT computation, given pointers to the real and
    // imaginary parts.  If "forward", do the forward FFT; else
    // do the inverse FFT (without the 1/N factor).
    // xr and xi are pointers to zero-based arrays of size N,
    // containing the real and imaginary parts
    // respectively.
    void Compute(Real *xr, Real *xi, bool forward) const;

    // This version of Compute takes a single array of size N*2,
    // containing [ r0 im0 r1 im1 ... ].  Otherwise its behavior is  the
    // same as the version above.
    void Compute(Real *x, bool forward);


    // This version of Compute is const; it operates on an array of size N*2
    // containing [ r0 im0 r1 im1 ... ], but it uses the argument "temp_buffer" as
    // temporary storage instead of a class-member variable.  It will allocate it if
    // needed.
    void Compute(Real *x, bool forward, std::vector<Real> *temp_buffer) const;

    ~SplitRadixComplexFft();

   protected:
    // temp_buffer_ is allocated only if someone calls Compute with only one Real*
    // argument and we need a temporary buffer while creating interleaved data.
    std::vector<Real> temp_buffer_;
    //private:
    void ComputeTables();
    void ComputeRecursive(Real *xr, Real *xi, Integer logn) const;
    void BitReversePermute(Real *x, Integer logn) const;

    Integer N_;
    Integer logn_;  // log(N)

    Integer *brseed_;
    // brseed is Evans' seed table, ref:  (Ref: D. M. W.
    // Evans, "An improved digit-reversal permutation algorithm ...",
    // IEEE Trans. ASSP, Aug. 1987, pp. 1120-1125).
    Real **tab_;       // Tables of butterfly coefficients.

    //KALDI_DISALLOW_COPY_AND_ASSIGN(SplitRadixComplexFft);
  };

  template<typename Real>

  class SplitRadixRealFft : private SplitRadixComplexFft < Real > {
   public:
    typedef int Integer;
    typedef int MatrixIndexT;

    SplitRadixRealFft(MatrixIndexT N)
      :  // will fail unless N>=4 and N is a power of 2.
         SplitRadixComplexFft<Real>(N / 2), N_(N) {}

    /// If forward == true, this function transforms from a sequence of N real points to its complex fourier
    /// transform; otherwise it goes in the reverse direction.  If you call it
    /// in the forward and then reverse direction and multiply by 1.0/N, you
    /// will get back the original data.
    /// The interpretation of the complex-FFT data is as follows: the array
    /// is a sequence of complex numbers C_n of length N/2 with (real, im) format,
    /// i.e. [real0, real_{N/2}, real1, im1, real2, im2, real3, im3, ...].
    void Compute(Real *x, bool forward);


    /// This is as the other Compute() function, but it is a const version that
    /// uses a user-supplied buffer.
    void Compute(Real *x, bool forward, std::vector<Real> *temp_buffer) const;

    static inline void ComplexAddProduct(const Real &a_re, const Real &a_im,
                                         const Real &b_re, const Real &b_im,
                                         Real *c_re, Real *c_im) {
      *c_re += b_re*a_re - b_im*a_im;
      *c_im += b_re*a_im + b_im*a_re;
    }




    static inline void ComplexImExp(Real x, Real *a_re, Real *a_im) {
      *a_re = std::cos(x);
      *a_im = std::sin(x);
    }


    //! ComplexMul implements, inline, the complex multiplication b *= a.
    static inline void ComplexMul(const Real &a_re, const Real &a_im,
                                  Real *b_re, Real *b_im) {
      Real tmp_re = (*b_re * a_re) - (*b_im * a_im);
      *b_im = *b_re * a_im + *b_im * a_re;
      *b_re = tmp_re;
    }

   private:
    //KALDI_DISALLOW_COPY_AND_ASSIGN(SplitRadixRealFft);
    int N_;
  };

 public:

  MfccOptions mfcc_opts_;

 protected:

  SplitRadixRealFft<float> *srfft_;
  std::map<float, MelBanks *> mel_banks_;
  MelBanks *this_mel_banks_;
  int padded_window_size_;
  float vtln_warp_;
  std::vector<float> temp_buffer_;  // used by srfft.


  std::vector<float> windowCoff_;
  std::vector<float> window_;    // windowed waveform.
  std::vector<float> power_spectrum_;// todo
  std::vector<float> mel_energies_;

  float log_energy_floor_;

 public:

  bool use_energy_;
  int  rand_seed_;
  RandomGenerator rand_generator_;


  // State for thread-safe random number generator
  struct RandomState {
    RandomState();
    unsigned seed;
  };



  /* inner function define here */
  static inline int RoundUpToNearestPowerOfTwo(int n) {
    IDEC_ASSERT(n > 0);
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
  }


  //static int Rand(struct RandomState* state = NULL);
  inline static  int Rand(struct RandomState *state = NULL) {
#if defined _MSC_VER || defined __BIONIC__
    // On Windows and Android, just call Rand()
    return rand();
#else
    if (state) {
      return rand_r(&(state->seed));
    } else {
      int rs = pthread_mutex_lock(&_RandMutex);
      IDEC_ASSERT(rs == 0);
      int val = rand();
      rs = pthread_mutex_unlock(&_RandMutex);
      IDEC_ASSERT(rs == 0);
      return val;
    }
#endif
  }

  inline float Log(float x) { return logf(x); }

  template<typename Real> static inline  Real VectorDotProduct(const Real *a,
      const size_t &len_a, const Real *b, const size_t &len_b) {
    //size_t dim = a.size();
    if (len_a != len_b) {
      IDEC_ERROR << "dim of two input not eq\n";
    }
    Real out = 0;
    for (size_t i = 0; i < len_a; i++) {
      out += a[i] * b[i];
    }
    return out;
  }

  template<typename Real> static inline  Real VectorDotProduct(
    const std::vector<Real> &a, const std::vector<Real> &b) {
    //size_t dim = a.size();
    if (a.size() != b.size()) {
      IDEC_ERROR << "dim of two input not eq\n";
    }
    Real out = 0;
    for (int i = 0; i < (int)a.size(); i++) {
      out += a[i] * b[i];
    }
    return out;
  }

  /// Returns a random number strictly between 0 and 1.
  inline float RandUniform(struct RandomState *state = NULL) {
    if (mfcc_opts_.frame_opts.deterministic_dither) {
      return static_cast<float>((float)((rand_generator_.Rand() + 1.0) /
                                        (rand_generator_.RandMax() + 2.0)));
    } else {
      return static_cast<float>((float)((Rand(state) + 1.0) / (RAND_MAX + 2.0)));
    }
  }


  inline float RandGauss(struct RandomState *state = NULL) {
    return static_cast<float>((float)(sqrtf(-2 * logf(RandUniform(state)))
                                      * cosf((float)(2 * M_PI*RandUniform(state)))));
  }


  void Dither(Real *waveform, Real dither_value);
  float Average(Real *waveform);
  void Add(Real *waveform, Real c);
  void Preemphasize(Real *waveform, float preemph_coeff);
  void MulElements(Real *waveform, std::vector<float> &windowCoff_);
  void FeatureWindowFunction(std::vector<float> &window);
  int  WindowSize();
  int PaddedWindowSize(bool round_to_power_of_two);
  void CopyToWindow(Real *waveform);
  void ExtractWindow(float *tmp_buf, float *log_energy_pre_window);
  void ComputePowerSpectrum(float  *waveform, int dim);
  int applyFloor(std::vector<float> &mel_energies_, float min, int dim_);
  void applyLog(std::vector<float> &mel_energies_, int dim_);
  void MfccCompute(float *wav, MelBanks &mel_banks, float *mel_energies__out);
  MelBanks *GetMelBanks(float vtln_warp_);


  FrontendComponent_Waveform2Filterbank(ParseOptions &po,
                                        const std::string name = "Waveform2Filterbank") : FrontendComponentInterface(
                                            po, name) {
    // set default configs
    use_energy_ = false;
    vtln_warp_ = 1.0;
    srfft_ = NULL;
    this_mel_banks_ = NULL;
    rand_seed_ = 17; // arbitrary set constant

    // register optional configs
    mfcc_opts_.Register(&po, name);
  }

  ~FrontendComponent_Waveform2Filterbank() {
    if (srfft_ != NULL)
      delete srfft_;
    if (this_mel_banks_ != NULL)
      delete this_mel_banks_;
  }

  virtual void Init() {
    FrontendComponentInterface::Init();

    // prepare output-related staff
    output_dim_ = mfcc_opts_.mel_opts.num_bins;
    if (use_energy_) {
      output_dim_++;
    }
    output_buff_.Resize(output_dim_, 1);

    // set outside
    mfcc_opts_.use_energy = use_energy_;

    //int padded_window_size = mfcc_opts_.frame_opts.PaddedWindowSize();
    padded_window_size_ = PaddedWindowSize(
                            mfcc_opts_.frame_opts.round_to_power_of_two);
    if ((padded_window_size_ & (padded_window_size_ - 1)) ==
        0) { // Is a power of two...
      srfft_ = new SplitRadixRealFft<float>(padded_window_size_);
    }

    /*initial mel coff */
    this_mel_banks_ = GetMelBanks(vtln_warp_);

    /*window coffee  initial accroding to config type*/
    windowCoff_.resize(input_dim_);
    FeatureWindowFunction(windowCoff_);

    // set window size
    //window_.resize(PaddedWindowSize(mfcc_opts_.frame_opts.round_to_power_of_two) );
    //window_.resize(padded_window_size_);
    //window_.assign(padded_window_size_, 0);

    // set power_spectrum_ size
    power_spectrum_.resize(padded_window_size_ / 2 + 1);

    //set mel buf size
    mel_energies_.resize(mfcc_opts_.mel_opts.num_bins);

    // according to config outside ,alter the option inside
    //if (useEnergy_)
    //mfcc_opts_.use_energy = true;

    //calc energy floor
    if (mfcc_opts_.energy_floor > 0.0)
      log_energy_floor_ = log(mfcc_opts_.energy_floor);

  }


  // this is set for each utterance to make sure the feature is deterministic
  virtual void Reset() {
    FrontendComponentInterface::Reset();
    rand_generator_.SetSeed(rand_seed_);
  }

  virtual bool Process() {
    if (input_buf_.empty())
      return(false);

    xnnFloatRuntimeMatrixCircularBuffer &input_buff = input_buf_[0];
    while (!input_buff.Empty()) {
      MfccCompute(input_buff.Col(0), *this_mel_banks_, output_buff_.Col(0));

      if (!SendOneFrameToSucceedingComponents())
        return(false);

      input_buff.PopfrontOneColumn();
    }


    return(true);
  }
};

}
#endif


