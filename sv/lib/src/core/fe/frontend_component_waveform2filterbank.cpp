// Copyright 2015 Alibaba-inc  [zhijie.yzj]
#include "am/xnn_runtime.h"
#include "am/xnn_kaldi_utility.h"
#include "base/log_message.h"
#include "util/parse-options.h"
#include "fe/frontend_component_waveform2filterbank.h"
namespace idec {


FrontendComponent_Waveform2Filterbank::RandomState::RandomState() {
  // we initialize it as Rand() + 27437 instead of just Rand(), because on some
  // systems, e.g. at the very least Mac OSX Yosemite and later, it seems to be
  // the case that rand_r when initialized with rand() will give you the exact
  // same sequence of numbers that rand() will give if you keep calling rand()
  // after that initial call.  This can cause problems with repeated sequences.
  // For example if you initialize two RandomState structs one after the other
  // without calling rand() in between, they would give you the same sequence
  // offset by one (if we didn't have the "+ 27437" in the code).  27437 is just
  // a randomly chosen prime number.
  seed = Rand() + 27437;
}


void FrontendComponent_Waveform2Filterbank::Dither(Real *waveform,
    Real dither_value) {
  for (int  i = 0; i < input_dim_; i++) {
    waveform[i] += RandGauss() * dither_value;
  }

}

float FrontendComponent_Waveform2Filterbank::Average(Real *waveform) {
  float sum = 0.0f;
  for (int  i = 0; i < input_dim_; i++) {
    sum += waveform[i];
  }
  sum /= input_dim_;
  return sum;
}

void FrontendComponent_Waveform2Filterbank::Add(Real *waveform, Real c) {
  for (int i = 0; i < input_dim_; i++) {
    waveform[i] += c;
  }
}


void FrontendComponent_Waveform2Filterbank::Preemphasize(Real *waveform,
    float preemph_coeff) {
  if (preemph_coeff == 0.0f) return;
  IDEC_ASSERT(preemph_coeff >= 0.0f && preemph_coeff <= 1.0f);

  for (size_t  i = input_dim_ -1; i > 0; i--) {
    waveform[i] -= preemph_coeff * waveform[i-1];
  }
  waveform[0] -= preemph_coeff * waveform[0];
}


void FrontendComponent_Waveform2Filterbank::MulElements(Real *waveform,
    std::vector<float> &windowCoff) {
  for (int i = 0; i < input_dim_; i++) {
    waveform[i] *= windowCoff[i];
  }

}

void FrontendComponent_Waveform2Filterbank::FeatureWindowFunction(
  std::vector<float> &window) {
  std::string window_type = mfcc_opts_.frame_opts.window_type;
  size_t length = window.size();

  for (size_t i = 0; i < length; i++) {
    float i_fl = static_cast<float>(i);
    if (window_type == "hanning") {
      window[i] =(float)( 0.5  - 0.5*cos(M_2PI * i_fl / (length-1)));
    } else if (window_type == "hamming") {
      window[i] =(float)( 0.54 - 0.46*cos(M_2PI * i_fl / (length-1)));
    } else if (window_type == "povey") {
      // like hamming but goes to zero at edges.
      window[i] = (float)pow(0.5 - 0.5*cos(M_2PI * i_fl / (length-1)), 0.85);
    }

    else if (window_type == "rectangular") {
      window[i] = 1.0f;
    } else {
      IDEC_ERROR << "Invalid window type " << window_type;
    }
  }
}

int  FrontendComponent_Waveform2Filterbank::WindowSize() {
  //return static_cast<size_t>(samp_freq * 0.001 * frame_length_ms);

  //return static_cast<size_t>(8000 * 0.001 * 25);
  return static_cast<int>(mfcc_opts_.frame_opts.samp_freq* 0.001 *
                          mfcc_opts_.frame_opts.frame_length_ms);

}


int FrontendComponent_Waveform2Filterbank::applyFloor(std::vector<float>
    &mel_energies,  float min, int dim_) {
  int num_floored = 0;
  for (int i = 0; i < dim_; i++) {
    if (mel_energies[i] < min) {
      mel_energies[i] = min;
      num_floored++;
    }
  }
  return num_floored;
}




int FrontendComponent_Waveform2Filterbank::PaddedWindowSize(
  bool round_to_power_of_two) {
  return (round_to_power_of_two ? RoundUpToNearestPowerOfTwo(WindowSize()) :
          WindowSize());
}

void FrontendComponent_Waveform2Filterbank::CopyToWindow(Real *waveform) {
  // [zhijie.yz] directly copy to window_
  window_.assign(padded_window_size_, 0);
#ifdef _MSC_VER
  memcpy_s(&window_[0], sizeof(Real)*input_dim_, waveform,
           sizeof(Real)*input_dim_);
#else
  memcpy(&window_[0], waveform, sizeof(Real)*input_dim_);
#endif

  //    int windowLength   = inputDim_;
  //    for (int i = 0; i < windowLength; i++ )
  //       {
  //window_[i] = waveform[i];
  //       }
  //

  //for (int i = windowLength; i < window_.size(); i++ )
  //{
  //    window_[i]  =  0;
  //}
}


void FrontendComponent_Waveform2Filterbank::ExtractWindow(float *tmp_buf,
    float *log_energy_pre_window ) {
  if (mfcc_opts_.frame_opts.dither != 0.0) {
    Dither(tmp_buf,  mfcc_opts_.frame_opts.dither);
  }

  if (mfcc_opts_.frame_opts.remove_dc_offset) {
    float avg = Average(tmp_buf);
    Add(tmp_buf, -avg);
  }


  if (log_energy_pre_window != NULL) {

    /*float energy = std::max(VecVec(window_part, window_part),
                            std::numeric_limits<float>::min());
        *log_energy_pre_window = log(energy);
    */

    //modified by zhuozhu.zz
    //float min_energy = std::numeric_limits<float>::min();
    float min_energy = std::numeric_limits<float>::epsilon();

    // TODO :VecVec shixian
    float energy = 0.0;
    for (int i = 0; i < input_dim_; i++) {
      energy += tmp_buf[i] * tmp_buf[i];
    }

    if (energy < min_energy) {
      energy = min_energy;
    }

    *log_energy_pre_window = log(energy);
  }


  if (mfcc_opts_.frame_opts.preemph_coeff != 0.0) {
    Preemphasize(tmp_buf, mfcc_opts_.frame_opts.preemph_coeff);
  }


  // add window
  MulElements(tmp_buf, windowCoff_);


  //  set zero to the end
  //int frame_length_padded =  PaddedWindowSize(mfcc_opts_.frame_opts.round_to_power_of_two) ;
  //if (inputDim_ != padded_window_size_)
  //{
  //    SetZero(tmp_buf);
  //}
  // [zhijie.yzj] always need to copy tmp_buf to window_
  CopyToWindow(tmp_buf);
}


void FrontendComponent_Waveform2Filterbank::ComputePowerSpectrum(
  float  *waveform, int dim) {

  // no, letting it be non-power-of-two for now.
  // IDEC_ASSERT(dim > 0 && (dim & (dim-1) == 0));  // make sure a power of two.. actually my FFT code
  // does not require this (dan) but this is better in case we use different code [dan].

  // RealFft(waveform, true);  // true == forward (not inverse) FFT; makes no difference here,
  // as we just want power spectrum.

  // now we have in waveform, first half of complex spectrum
  // it's stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
  int half_dim = dim/2;
  float first_energy = waveform[0] * waveform[0];
  float  last_energy = waveform[1] * waveform[1];  // handle this special case
  for (int i = 1; i < half_dim; i++) {
    float real = waveform[i*2];
    float im = waveform[i*2 + 1];
    waveform[i] = real*real + im*im;
  }
  waveform[0] = first_energy;
  waveform[half_dim] = last_energy;  // Will actually never be used, and anyway
  // if the signal has been bandlimited sensibly this should be zero.
}


FrontendComponent_Waveform2Filterbank::MelBanks
*FrontendComponent_Waveform2Filterbank::GetMelBanks(float vtln_warp) {
  MelBanks *this_mel_banks = NULL;
  std::map<float, MelBanks *>::iterator iter = mel_banks_.find(vtln_warp);
  if (iter == mel_banks_.end()) {
    this_mel_banks = new MelBanks(mfcc_opts_.mel_opts,
                                  mfcc_opts_.frame_opts,
                                  vtln_warp);
    mel_banks_[vtln_warp] = this_mel_banks;
  } else {
    this_mel_banks = iter->second;
  }
  return this_mel_banks;
}


void FrontendComponent_Waveform2Filterbank::MfccCompute(float *wav,
    MelBanks &mel_banks, float *mel_energies_out) {


  float log_energy;
  ExtractWindow(wav, (mfcc_opts_.use_energy
                      && mfcc_opts_.raw_energy ? &log_energy : NULL));

  if (mfcc_opts_.use_energy && !mfcc_opts_.raw_energy) {
    //log_energy = log(std::max(VecVec(window, window), std::numeric_limits<float>::min()));
    //float tmp_energy = vec_dot(window, window);
    float tmp_energy = VectorDotProduct(window_, window_);

    // modified by zhuozhu.zz
    //if (tmp_energy < std::numeric_limits<float>::min())
    //{
    //    tmp_energy = std::numeric_limits<float>::min();
    //}
    //log_energy = log(tmp_energy);

    if (tmp_energy < std::numeric_limits<float>::epsilon()) {
      tmp_energy = std::numeric_limits<float>::epsilon();
    }
    log_energy = log(tmp_energy);
  }

  /*fft calc*/
  float *tmp_buf = &window_[0];
  if (srfft_ != NULL)  // Compute FFT using the split-radix algorithm.
    srfft_->Compute(tmp_buf, true, &temp_buffer_);
  else  // An alternative algorithm that works for non-powers-of-two.
    // RealFft(&window, true);
    IDEC_ERROR <<
               "for now not support alternative algorithm that works for non-powers-of-two ";

  /*PowerSpectrum calc*/
  ComputePowerSpectrum(tmp_buf, (int)window_.size());
  //for (int i = 0; i < power_spectrum_.size(); i++)
  //{
  //      power_spectrum_[i] = tmp_buf[i];
  //}
  // [zhijie.yzj] use assign directly
  power_spectrum_.assign(window_.begin(),
                         window_.begin() + power_spectrum_.size());

  /* mel calc */
  mel_banks.Compute(power_spectrum_, mel_energies_);

  //mel_energies_out = (float *)&mel_energies[0];




  // apply Floor
  // float min_float = std::numeric_limits<float>::min();
  float min_float = std::numeric_limits<float>::epsilon();
  applyFloor(mel_energies_, min_float, mfcc_opts_.mel_opts.num_bins);

  // apply log
  applyLog(mel_energies_, mfcc_opts_.mel_opts.num_bins);

  // copy to output
#ifdef _MSC_VER
  memcpy_s(mel_energies_out, mel_energies_.size() * sizeof(float),
           &mel_energies_[0], mel_energies_.size() * sizeof(float));
#else
  memcpy(mel_energies_out, &mel_energies_[0],
         mel_energies_.size() * sizeof(float));
#endif

  // append energy when neccessary
  if (use_energy_) {
    if (mfcc_opts_.energy_floor > 0.0 && log_energy < log_energy_floor_)
      log_energy = log_energy_floor_;
    mel_energies_out[output_dim_ - 1] = log_energy;
  }
}



/* funcions define of mel class*/
FrontendComponent_Waveform2Filterbank::MelBanks::MelBanks(
  const MelBanksOptions &opts,
  const FrameExtractionOptions &frame_opts,
  float vtln_warp_factor) :
  htk_mode_(opts.htk_mode) {
  int num_bins = opts.num_bins;
  if (num_bins < 3) IDEC_ERROR << "Must have at least 3 mel bins";
  float sample_freq = frame_opts.samp_freq;
  int window_length = static_cast<int>
                      (frame_opts.samp_freq*0.001*frame_opts.frame_length_ms);
  int window_length_padded =
    (frame_opts.round_to_power_of_two ?
     RoundUpToNearestPowerOfTwo_for_mel(window_length) :
     window_length);
  IDEC_ASSERT(window_length_padded % 2 == 0);
  int num_fft_bins = window_length_padded/2;
  float nyquist = (float) 0.5 * sample_freq;

  float low_freq = opts.low_freq, high_freq;
  if (opts.high_freq > 0.0)
    high_freq = opts.high_freq;
  else
    high_freq = nyquist + opts.high_freq;

  if (low_freq < 0.0 || low_freq >= nyquist
      || high_freq <= 0.0 || high_freq > nyquist
      || high_freq <= low_freq)
    IDEC_ERROR << "Bad values in options: low-freq " << low_freq
               << " and high-freq " << high_freq << " vs. nyquist "
               << nyquist;

  float fft_bin_width = sample_freq / window_length_padded;
  // fft-bin width [think of it as Nyquist-freq / half-window-length]

  float mel_low_freq = MelScale(low_freq);
  float mel_high_freq = MelScale(high_freq);

  debug_ = opts.debug_mel;

  // divide by num_bins+1 in next line because of end-effects where the bins
  // spread out to the sides.
  float mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1);

  float vtln_low = opts.vtln_low,
        vtln_high = opts.vtln_high;
  if (vtln_high < 0.0) vtln_high += nyquist;

  if (vtln_warp_factor != 1.0 &&
      (vtln_low < 0.0 || vtln_low <= low_freq
       || vtln_low >= high_freq
       || vtln_high <= 0.0 || vtln_high >= high_freq
       || vtln_high <= vtln_low))
    IDEC_ERROR << "Bad values in options: vtln-low " << vtln_low
               << " and vtln-high " << vtln_high << ", versus "
               << "low-freq " << low_freq << " and high-freq "
               << high_freq;

  bins_.resize(num_bins);
  center_freqs_.resize(num_bins);


  for (int bin = 0; bin < num_bins; bin++) {
    float left_mel = mel_low_freq + bin * mel_freq_delta,
          center_mel = mel_low_freq + (bin + 1) * mel_freq_delta,
          right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

    if (vtln_warp_factor != 1.0) {
      left_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                 vtln_warp_factor, left_mel);
      center_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                   vtln_warp_factor, center_mel);
      right_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                  vtln_warp_factor, right_mel);
    }
    center_freqs_[bin] = InverseMelScale(center_mel);
    // this_bin will be a vector of coefficients that is only
    // nonzero where this mel bin is active.
    std::vector<float> this_bin;
    this_bin.resize(num_fft_bins);

    int first_index = -1, last_index = -1;
    for (int i = 0; i < num_fft_bins; i++) {
      float freq = (fft_bin_width * i);  // center freq of this fft bin.
      float mel = MelScale(freq);
      if (mel > left_mel && mel < right_mel) {
        float weight;
        if (mel <= center_mel)
          weight = (mel - left_mel) / (center_mel - left_mel);
        else
          weight = (right_mel-mel) / (right_mel-center_mel);
        //this_bin(i) = weight;
        this_bin[i] = weight;

        if (first_index == -1)
          first_index = i;
        last_index = i;
      }
    }
    IDEC_ASSERT(first_index != -1 && last_index >= first_index
                && "You may have set --num-mel-bins too large.");

    bins_[bin].first = first_index;

    //int size = last_index + 1 - first_index;
    //bins_[bin].second.Resize(size);
    // bins_[bin].second.CopyFromVec(this_bin.Range(first_index, size));

    int size = last_index + 1 - first_index;
    bins_[bin].second.resize(size);
    for (int k = 0; k < size; k++) {
      bins_[bin].second[k] = this_bin[k+first_index];
    }

    // Replicate a bug in HTK, for testing purposes.
    if (opts.htk_mode && bin == 0 && mel_low_freq != 0.0)
      bins_[bin].second[0] = 0.0;

  }
}


float FrontendComponent_Waveform2Filterbank::MelBanks::VtlnWarpFreq(
  float vtln_low_cutoff,  // upper+lower frequency cutoffs for VTLN.
  float vtln_high_cutoff,
  float low_freq,  // upper+lower frequency cutoffs in mel computation
  float high_freq,
  float vtln_warp_factor,
  float freq) {
  /// This computes a VTLN warping function that is not the same as HTK's one,
  /// but has similar inputs (this function has the advantage of never producing
  /// empty bins).

  /// This function computes a warp function F(freq), defined between low_freq and
  /// high_freq inclusive, with the following properties:
  ///  F(low_freq) == low_freq
  ///  F(high_freq) == high_freq
  /// The function is continuous and piecewise linear with two inflection
  ///   points.
  /// The lower inflection point (measured in terms of the unwarped
  ///  frequency) is at frequency l, determined as described below.
  /// The higher inflection point is at a frequency h, determined as
  ///   described below.
  /// If l <= f <= h, then F(f) = f/vtln_warp_factor.
  /// If the higher inflection point (measured in terms of the unwarped
  ///   frequency) is at h, then max(h, F(h)) == vtln_high_cutoff.
  ///   Since (by the last point) F(h) == h/vtln_warp_factor, then
  ///   max(h, h/vtln_warp_factor) == vtln_high_cutoff, so
  ///   h = vtln_high_cutoff / max(1, 1/vtln_warp_factor).
  ///     = vtln_high_cutoff * min(1, vtln_warp_factor).
  /// If the lower inflection point (measured in terms of the unwarped
  ///   frequency) is at l, then min(l, F(l)) == vtln_low_cutoff
  ///   This implies that l = vtln_low_cutoff / min(1, 1/vtln_warp_factor)
  ///                       = vtln_low_cutoff * max(1, vtln_warp_factor)


  if (freq < low_freq
      || freq > high_freq) return freq;  // in case this gets called
  // for out-of-range frequencies, just return the freq.

  IDEC_ASSERT(vtln_low_cutoff > low_freq &&
              "be sure to set the --vtln-low option higher than --low-freq");
  IDEC_ASSERT(vtln_high_cutoff < high_freq &&
              "be sure to set the --vtln-high option lower than --high-freq [or negative]");
  float one = 1.0;
  float l = vtln_low_cutoff * std::max(one, vtln_warp_factor);
  float h = vtln_high_cutoff * std::min(one, vtln_warp_factor);
  float scale =(float) 1.0 / vtln_warp_factor;
  float Fl = scale * l;  // F(l);
  float Fh = scale * h;  // F(h);
  IDEC_ASSERT(l > low_freq && h < high_freq);
  // slope of left part of the 3-piece linear function
  float scale_left = (Fl - low_freq) / (l - low_freq);
  // [slope of center part is just "scale"]

  // slope of right part of the 3-piece linear function
  float scale_right = (high_freq - Fh) / (high_freq - h);

  if (freq < l) {
    return low_freq + scale_left * (freq - low_freq);
  } else if (freq < h) {
    return scale * freq;
  } else {  // freq >= h
    return high_freq + scale_right * (freq - high_freq);
  }
}


float FrontendComponent_Waveform2Filterbank::MelBanks::VtlnWarpMelFreq(
  float vtln_low_cutoff,  // upper+lower frequency cutoffs for VTLN.
  float vtln_high_cutoff,
  float low_freq,  // upper+lower frequency cutoffs in mel computation
  float high_freq,
  float vtln_warp_factor,
  float mel_freq) {
  return MelScale(VtlnWarpFreq(vtln_low_cutoff, vtln_high_cutoff,
                               low_freq, high_freq,
                               vtln_warp_factor, InverseMelScale(mel_freq)));
}




// "power_spectrum" contains fft energies.
void FrontendComponent_Waveform2Filterbank::MelBanks::Compute(
  const std::vector<float> &power_spectrum,
  std::vector<float> &mel_energies_out) const {

  int num_bins = (int)bins_.size();
  if ((int)mel_energies_out.size() != num_bins)
    mel_energies_out.resize(num_bins);

  for (int i = 0; i < num_bins; i++) {
    int offset = bins_[i].first;
    const std::vector<float> &v(bins_[i].second);

    //  energy = VecVec(v, power_spectrum.Range(offset, v.Dim()));
    //std::vector<float> vec_pow_spectrum(power_spectrum.begin()+offset, power_spectrum.begin()+offset+v.size());
    //vec_pow_spectrum.resize(v.size());
    //for (int k = 0; k < v.size(); k++)
    //{
    //    vec_pow_spectrum[k] = power_spectrum[offset+k];
    //}
    //float energy  = vec_dot(v,vec_pow_spectrum);
    float energy = VectorDotProduct<float>(v.data(), v.size(),
                                           power_spectrum.data() + offset, v.size());




    // HTK-like flooring- for testing purposes (we prefer dither)
    if (htk_mode_ && energy < 1.0) energy = 1.0;
    mel_energies_out[i] = energy;

    // The following assert was added due to a problem with OpenBlas that
    // we had at one point (it was a bug in that library).  Just to detect
    // it early.
    IDEC_ASSERT(!IDEC_ISNAN(mel_energies_out[i]));
  }

}



void FrontendComponent_Waveform2Filterbank::applyLog(std::vector<float>
    &mel_energies,   int dim_) {
  for (int i = 0; i < dim_; i++) {
    if (mel_energies[i] < 0.0)
      IDEC_ERROR << "Trying to take log of a negative number.";
    mel_energies[i] = Log(mel_energies[i]);
  }
}



/**********************
//FFT functions define here
**********************/

template<typename Real>
FrontendComponent_Waveform2Filterbank::SplitRadixComplexFft<Real>::SplitRadixComplexFft(
  MatrixIndexT N) {
  if ((N & (N - 1)) != 0 || N <= 1)
    IDEC_ERROR << "SplitRadixComplexFft called with invalid number of points "
               << N;
  N_ = N;
  logn_ = 0;
  while (N > 1) {
    N >>= 1;
    logn_++;
  }
  ComputeTables();
}

template<typename Real>
void FrontendComponent_Waveform2Filterbank::SplitRadixComplexFft<Real>::ComputeTables() {
  MatrixIndexT    imax, lg2, i, j;
  MatrixIndexT     m, m2, m4, m8, nel, n;
  Real    *cn, *spcn, *smcn, *c3n, *spc3n, *smc3n;
  Real    ang, c, s;

  lg2 = logn_ >> 1;
  if (logn_ & 1) lg2++;
#ifdef _MSC_VER
  brseed_ = new MatrixIndexT[1i64 << lg2];
#else
  brseed_ = new MatrixIndexT[1 << lg2];
#endif
  brseed_[0] = 0;
  brseed_[1] = 1;
  for (j = 2; j <= lg2; j++) {
    imax = 1 << (j - 1);
    for (i = 0; i < imax; i++) {
      brseed_[i] <<= 1;
      brseed_[i + imax] = brseed_[i] + 1;
    }
  }

  if (logn_ < 4) {
    tab_ = NULL;
  } else {
    tab_ = new Real*[logn_ - 3];
    for (i = logn_; i >= 4; i--) {
      /* Compute a few constants */
      m = 1 << i;
      m2 = m / 2;
      m4 = m2 / 2;
      m8 = m4 / 2;

      /* Allocate memory for tables */
      nel = m4 - 2;

      tab_[i - 4] = new Real[6 * nel];

      /* Initialize pointers */
      cn = tab_[i - 4];
      spcn = cn + nel;
      smcn = spcn + nel;
      c3n = smcn + nel;
      spc3n = c3n + nel;
      smc3n = spc3n + nel;

      /* Compute tables */
      for (n = 1; n < m4; n++) {
        if (n == m8) continue;
        ang = (Real) (n * M_2PI / m);
        c = std::cos(ang);
        s = std::sin(ang);
        *cn++ = c;
        *spcn++ = -(s + c);
        *smcn++ = s - c;
        ang = (float)(3 * n * M_2PI / m);
        c = std::cos(ang);
        s = std::sin(ang);
        *c3n++ = c;
        *spc3n++ = -(s + c);
        *smc3n++ = s - c;
      }
    }
  }
}

template<typename Real>
FrontendComponent_Waveform2Filterbank::SplitRadixComplexFft<Real>::~SplitRadixComplexFft() {
  delete[] brseed_;
  if (tab_ != NULL) {
    for (MatrixIndexT i = 0; i < logn_ - 3; i++)
      delete[] tab_[i];
    delete[] tab_;
  }
}

template<typename Real>
void FrontendComponent_Waveform2Filterbank::SplitRadixComplexFft<Real>::Compute(
  Real *xr, Real *xi, bool forward) const {
  if (!forward) {  // reverse real and imaginary parts for complex FFT.
    Real *tmp = xr;
    xr = xi;
    xi = tmp;
  }
  ComputeRecursive(xr, xi, logn_);
  if (logn_ > 1) {
    BitReversePermute(xr, logn_);
    BitReversePermute(xi, logn_);
  }
}

template<typename Real>
void FrontendComponent_Waveform2Filterbank::SplitRadixComplexFft<Real>::Compute(
  Real *x, bool forward,
  std::vector<Real> *temp_buffer) const {
  IDEC_ASSERT(temp_buffer != NULL);
  if ((Integer)temp_buffer->size() != N_)
    temp_buffer->resize(N_);
  Real *temp_ptr = &((*temp_buffer)[0]);
  for (MatrixIndexT i = 0; i < N_; i++) {
    x[i] = x[i * 2];  // put the real part in the first half of x.
    temp_ptr[i] = x[i * 2 + 1];  // put the imaginary part in temp_buffer.
  }
  // copy the imaginary part back to the second half of x.
  memcpy(static_cast<void *>(x + N_),
         static_cast<void *>(temp_ptr),
         sizeof(Real) * N_);

  Compute(x, x + N_, forward);
  // Now change the format back to interleaved.
  memcpy(static_cast<void *>(temp_ptr),
         static_cast<void *>(x + N_),
         sizeof(Real) * N_);
  for (MatrixIndexT i = N_ - 1; i > 0; i--) {  // don't include 0,
    // in case MatrixIndexT is unsigned, the loop would not terminate.
    // Treat it as a special case.
    x[i * 2] = x[i];
    x[i * 2 + 1] = temp_ptr[i];
  }
  x[1] = temp_ptr[0];  // special case of i = 0.
}

template<typename Real>
void FrontendComponent_Waveform2Filterbank::SplitRadixComplexFft<Real>::Compute(
  Real *x, bool forward) {
  this->Compute(x, forward, &temp_buffer_);
}

template<typename Real>
void FrontendComponent_Waveform2Filterbank::SplitRadixComplexFft<Real>::BitReversePermute(
  Real *x, MatrixIndexT logn) const {
  MatrixIndexT      i, j, lg2, n;
  MatrixIndexT      off, fj, gno, *brp;
  Real    tmp, *xp, *xq;

  lg2 = logn >> 1;
  n = 1 << lg2;
  if (logn & 1) lg2++;

  /* Unshuffling loop */
  for (off = 1; off < n; off++) {
    fj = n * brseed_[off];
    i = off;
    j = fj;
    tmp = x[i];
    x[i] = x[j];
    x[j] = tmp;
    xp = &x[i];
    brp = &(brseed_[1]);
    for (gno = 1; gno < brseed_[off]; gno++) {
      xp += n;
      j = fj + *brp++;
      xq = x + j;
      tmp = *xp;
      *xp = *xq;
      *xq = tmp;
    }
  }
}


template<typename Real>
void FrontendComponent_Waveform2Filterbank::SplitRadixComplexFft<Real>::ComputeRecursive(
  Real *xr, Real *xi, MatrixIndexT logn) const {

  MatrixIndexT    m, m2, m4, m8, nel, n;
  Real    *xr1, *xr2, *xi1, *xi2;
  Real    *cn, *spcn, *smcn, *c3n, *spc3n, *smc3n;
  Real    tmp1, tmp2;
  Real   sqhalf = M_SQRT1_2;
  xr1 = xr2 = xi1 = xi2 = NULL;
  cn  = spcn = smcn = c3n = spc3n = smc3n= NULL;

  /* Check range of logn */
  if (logn < 0)
    IDEC_ERROR << "Error: logn is out of bounds in SRFFT";

  /* Compute trivial cases */
  if (logn < 3) {
    if (logn == 2) {  /* length m = 4 */
      xr2 = xr + 2;
      xi2 = xi + 2;
      tmp1 = *xr + *xr2;
      *xr2 = *xr - *xr2;
      *xr = tmp1;
      tmp1 = *xi + *xi2;
      *xi2 = *xi - *xi2;
      *xi = tmp1;
      xr1 = xr + 1;
      xi1 = xi + 1;
      xr2++;
      xi2++;
      tmp1 = *xr1 + *xr2;
      *xr2 = *xr1 - *xr2;
      *xr1 = tmp1;
      tmp1 = *xi1 + *xi2;
      *xi2 = *xi1 - *xi2;
      *xi1 = tmp1;
      xr2 = xr + 1;
      xi2 = xi + 1;
      tmp1 = *xr + *xr2;
      *xr2 = *xr - *xr2;
      *xr = tmp1;
      tmp1 = *xi + *xi2;
      *xi2 = *xi - *xi2;
      *xi = tmp1;
      xr1 = xr + 2;
      xi1 = xi + 2;
      xr2 = xr + 3;
      xi2 = xi + 3;
      tmp1 = *xr1 + *xi2;
      tmp2 = *xi1 + *xr2;
      *xi1 = *xi1 - *xr2;
      *xr2 = *xr1 - *xi2;
      *xr1 = tmp1;
      *xi2 = tmp2;
      return;
    } else if (logn == 1) { /* length m = 2 */
      xr2 = xr + 1;
      xi2 = xi + 1;
      tmp1 = *xr + *xr2;
      *xr2 = *xr - *xr2;
      *xr = tmp1;
      tmp1 = *xi + *xi2;
      *xi2 = *xi - *xi2;
      *xi = tmp1;
      return;
    } else if (logn == 0) return; /* length m = 1 */
  }

  /* Compute a few constants */
  m = 1 << logn;
  m2 = m / 2;
  m4 = m2 / 2;
  m8 = m4 / 2;


  /* Step 1 */
  xr1 = xr;
  xr2 = xr1 + m2;
  xi1 = xi;
  xi2 = xi1 + m2;
  for (n = 0; n < m2; n++) {
    tmp1 = *xr1 + *xr2;
    *xr2 = *xr1 - *xr2;
    xr2++;
    *xr1++ = tmp1;
    tmp2 = *xi1 + *xi2;
    *xi2 = *xi1 - *xi2;
    xi2++;
    *xi1++ = tmp2;
  }

  /* Step 2 */
  xr1 = xr + m2;
  xr2 = xr1 + m4;
  xi1 = xi + m2;
  xi2 = xi1 + m4;
  for (n = 0; n < m4; n++) {
    tmp1 = *xr1 + *xi2;
    tmp2 = *xi1 + *xr2;
    *xi1 = *xi1 - *xr2;
    xi1++;
    *xr2++ = *xr1 - *xi2;
    *xr1++ = tmp1;
    *xi2++ = tmp2;
    // xr1++; xr2++; xi1++; xi2++;
  }

  /* Steps 3 & 4 */
  xr1 = xr + m2;
  xr2 = xr1 + m4;
  xi1 = xi + m2;
  xi2 = xi1 + m4;
  if (logn >= 4) {
    nel = m4 - 2;
    cn = tab_[logn - 4];
    spcn = cn + nel;
    smcn = spcn + nel;
    c3n = smcn + nel;
    spc3n = c3n + nel;
    smc3n = spc3n + nel;
  }
  xr1++;
  xr2++;
  xi1++;
  xi2++;
  // xr1++; xi1++;
  for (n = 1; n < m4; n++) {
    if (n == m8) {
      tmp1 = sqhalf * (*xr1 + *xi1);
      *xi1 = sqhalf * (*xi1 - *xr1);
      *xr1 = tmp1;
      tmp2 = sqhalf * (*xi2 - *xr2);
      *xi2 = -sqhalf * (*xr2 + *xi2);
      *xr2 = tmp2;
    } else {
      tmp2 = *cn++ * (*xr1 + *xi1);
      tmp1 = *spcn++ * *xr1 + tmp2;
      *xr1 = *smcn++ * *xi1 + tmp2;
      *xi1 = tmp1;
      tmp2 = *c3n++ * (*xr2 + *xi2);
      tmp1 = *spc3n++ * *xr2 + tmp2;
      *xr2 = *smc3n++ * *xi2 + tmp2;
      *xi2 = tmp1;
    }
    xr1++;
    xr2++;
    xi1++;
    xi2++;
  }

  /* Call ssrec again with half DFT length */
  ComputeRecursive(xr, xi, logn - 1);

  /* Call ssrec again twice with one quarter DFT length.
  Constants have to be recomputed, because they are static! */
  // m = 1 << logn; m2 = m / 2;
  ComputeRecursive(xr + m2, xi + m2, logn - 2);
  // m = 1 << logn;
  m4 = 3 * (m / 4);
  ComputeRecursive(xr + m4, xi + m4, logn - 2);
}


template<typename Real>
void FrontendComponent_Waveform2Filterbank::SplitRadixRealFft<Real>::Compute(
  Real *data, bool forward) {
  Compute(data, forward, &this->temp_buffer_);
}


// This code is mostly the same as the RealFft function.  It would be
// possible to replace it with more efficient code from Rico's book.
template<typename Real>
void FrontendComponent_Waveform2Filterbank::SplitRadixRealFft<Real>::Compute(
  Real *data, bool forward,
  std::vector<Real> *temp_buffer) const {
  MatrixIndexT N = N_, N2 = N / 2;
  IDEC_ASSERT(N % 2 == 0);
  if (forward) // call to base class
    SplitRadixComplexFft<Real>::Compute(data, true, temp_buffer);

  Real rootN_re, rootN_im;  // exp(-2pi/N), forward; exp(2pi/N), backward
  int forward_sign = forward ? -1 : 1;
  ComplexImExp(static_cast<Real>(M_2PI / N *forward_sign), &rootN_re, &rootN_im);
  Real kN_re = (float)-forward_sign,
       kN_im = 0.0;  // exp(-2pik/N), forward; exp(-2pik/N), backward
  // kN starts out as 1.0 for forward algorithm but -1.0 for backward.
  for (MatrixIndexT k = 1; 2 * k <= N2; k++) {
    ComplexMul(rootN_re, rootN_im, &kN_re, &kN_im);

    Real Ck_re, Ck_im, Dk_re, Dk_im;
    // C_k = 1/2 (B_k + B_{N/2 - k}^*) :
    Ck_re = 0.5f * (data[2 * k] + data[N - 2 * k]);
    Ck_im = 0.5f * (data[2 * k + 1] - data[N - 2 * k + 1]);
    // re(D_k)= 1/2 (im(B_k) + im(B_{N/2-k})):
    Dk_re = 0.5f * (data[2 * k + 1] + data[N - 2 * k + 1]);
    // im(D_k) = -1/2 (re(B_k) - re(B_{N/2-k}))
    Dk_im = -0.5f * (data[2 * k] - data[N - 2 * k]);
    // A_k = C_k + 1^(k/N) D_k:
    data[2 * k] = Ck_re;  // A_k <-- C_k
    data[2 * k + 1] = Ck_im;
    // now A_k += D_k 1^(k/N)
    ComplexAddProduct(Dk_re, Dk_im, kN_re, kN_im, &(data[2 * k]),
                      &(data[2 * k + 1]));

    MatrixIndexT kdash = N2 - k;
    if (kdash != k) {
      // Next we handle the index k' = N/2 - k.  This is necessary
      // to do now, to avoid invalidating data that we will later need.
      // The quantities C_{k'} and D_{k'} are just the conjugates of C_k
      // and D_k, so the equations are simple modifications of the above,
      // replacing Ck_im and Dk_im with their negatives.
      data[2 * kdash] = Ck_re;  // A_k' <-- C_k'
      data[2 * kdash + 1] = -Ck_im;
      // now A_k' += D_k' 1^(k'/N)
      // We use 1^(k'/N) = 1^((N/2 - k) / N) = 1^(1/2) 1^(-k/N) = -1 * (1^(k/N))^*
      // so it's the same as 1^(k/N) but with the real part negated.
      ComplexAddProduct(Dk_re, -Dk_im, -kN_re, kN_im, &(data[2 * kdash]),
                        &(data[2 * kdash + 1]));
    }
  }

  {
    // Now handle k = 0.
    // In simple terms: after the complex fft, data[0] becomes the sum of real
    // parts input[0], input[2]... and data[1] becomes the sum of imaginary
    // pats input[1], input[3]...
    // "zeroth" [A_0] is just the sum of input[0]+input[1]+input[2]..
    // and "n2th" [A_{N/2}] is input[0]-input[1]+input[2]... .
    Real zeroth = data[0] + data[1],
         n2th = data[0] - data[1];
    data[0] = zeroth;
    data[1] = n2th;
    if (!forward) {
      data[0] /= 2;
      data[1] /= 2;
    }
  }
  if (!forward) {  // call to base class
    SplitRadixComplexFft<Real>::Compute(data, false, temp_buffer);
    for (MatrixIndexT i = 0; i < N; i++)
      data[i] *= 2.0;
    // This is so we get a factor of N increase, rather than N/2 which we would
    // otherwise get from [ComplexFft, forward] + [ComplexFft, backward] in dimension N/2.
    // It's for consistency with our normal FFT convensions.
  }
}

template class
FrontendComponent_Waveform2Filterbank::SplitRadixComplexFft<float>;
template class
FrontendComponent_Waveform2Filterbank::SplitRadixComplexFft<double>;
template class FrontendComponent_Waveform2Filterbank::SplitRadixRealFft<float>;
template class
FrontendComponent_Waveform2Filterbank::SplitRadixRealFft<double>;

}
