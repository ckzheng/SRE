#ifndef FE_RONTEND_COMPONENT_WAVEFORM2PITCH_H_
#define FE_RONTEND_COMPONENT_WAVEFORM2PITCH_H_
#include "fe/frontend_component.h"

namespace idec {

class FrontendComponent_Waveform2Pitch : public FrontendComponentInterface {

 private:

  struct DeltaFeaturesOptions {
    int order;
    int window;  // e.g. 2; controls window size (window size is 2*window + 1)
    // the behavior at the edges is to replicate the first or last frame.
    // this is not configurable.

    DeltaFeaturesOptions(int order = 2, int32 window = 2) :
      order(order), window(window) {
    }
    void Register(OptionsItf *po, const std::string name) {
      po->Register(name + "::delta-order", &order, "Order of delta computation");
      po->Register(name + "::delta-window", &window,
                   "Parameter controlling window for delta computation (actual window"
                   " size for each delta order is 1 + 2*delta-window-size)");
    }
  };

  struct PitchExtractionOptions {
    // FrameExtractionOptions frame_opts;
    float samp_freq;          // sample frequency in hertz
    float frame_shift_ms;     // in milliseconds.
    float frame_length_ms;    // in milliseconds.
    float preemph_coeff;      // Preemphasis coefficient. [use is deprecated.]
    float min_f0;             // min f0 to search (Hz)
    float max_f0;             // max f0 to search (Hz)
    float soft_min_f0;        // Minimum f0, applied in soft way, must not
    // exceed min-f0
    float penalty_factor;     // cost factor for FO change
    float lowpass_cutoff;     // cutoff frequency for Low pass filter
    float resample_freq;      // Integer that determines filter width when
    // upsampling NCCF
    float delta_pitch;        // the pitch tolerance in pruning lags
    float nccf_ballast;       // Increasing this factor reduces NCCF for
    // quiet frames, helping ensure pitch
    // continuity in unvoiced region
    int lowpass_filter_width;   // Integer that determines filter width of
    // lowpass filter
    int upsample_filter_width;  // Integer that determines filter width when
    // upsampling NCCF

    // Below are newer config variables, not present in the original paper,
    // that relate to the online pitch extraction algorithm.

    // The maximum number of frames of latency that we allow the pitch-processing
    // to introduce, for online operation. If you set this to a large value,
    // there would be no inaccuracy from the Viterbi traceback (but it might make
    // you wait to see the pitch). This is not very relevant for the online
    // operation: normalization-right-context is more relevant, you
    // can just leave this value at zero.
    int max_frames_latency;

    // Only relevant for the function ComputeKaldiPitch which is called by
    // compute-kaldi-pitch-feats. If nonzero, we provide the input as chunks of
    // this size. This affects the energy normalization which has a small effect
    // on the resulting features, especially at the beginning of a file. For best
    // compatibility with online operation (e.g. if you plan to train models for
    // the online-deocding setup), you might want to set this to a small value,
    // like one frame.
    int frames_per_chunk;

    // Only relevant for the function ComputeKaldiPitch which is called by
    // compute-kaldi-pitch-feats, and only relevant if frames_per_chunk is
    // nonzero. If true, it will query the features as soon as they are
    // available, which simulates the first-pass features you would get in online
    // decoding. If false, the features you will get will be the same as those
    // available at the end of the utterance, after InputFinished() has been
    // called: e.g. during lattice rescoring.
    bool simulate_first_pass_online;

    // Only relevant for online operation or when emulating online operation
    // (e.g. when setting frames_per_chunk). This is the frame-index on which we
    // recompute the NCCF (e.g. frame-index 500 = after 5 seconds); if the
    // segment ends before this we do it when the segment ends. We do this by
    // re-computing the signal average energy, which affects the NCCF via the
    // "ballast term", scaling the resampled NCCF by a factor derived from the
    // average change in the "ballast term", and re-doing the backtrace
    // computation. Making this infinity would be the most exact, but would
    // introduce unwanted latency at the end of long utterances, for little
    // benefit.
    int recompute_frame;

    // This is a "hidden config" used only for testing the online pitch
    // extraction. If true, we compute the signal root-mean-squared for the
    // ballast term, only up to the current frame, rather than the end of the
    // current chunk of signal. This makes the output insensitive to the
    // chunking, which is useful for testing purposes.
    bool nccf_ballast_online;
    bool snip_edges;
    PitchExtractionOptions() :
      samp_freq(16000),
      frame_shift_ms(10.0f),
      frame_length_ms(25.0f),
      preemph_coeff(0.0f),
      min_f0(50),
      max_f0(400),
      soft_min_f0(10.0f),
      penalty_factor(0.1f),
      lowpass_cutoff(1000),
      resample_freq(4000),
      delta_pitch(0.005f),
      nccf_ballast(7000),
      lowpass_filter_width(1),
      upsample_filter_width(5),
      max_frames_latency(0),
      frames_per_chunk(0),
      simulate_first_pass_online(false),
      recompute_frame(500),
      nccf_ballast_online(false),
      snip_edges(true) {
    }

    void Register(OptionsItf *po, const std::string name) {
      po->Register(name + "::sample-frequency", &samp_freq,
                   "Waveform data sample frequency (must match the waveform "
                   "file, if specified there)");
      po->Register(name + "::frame-length", &frame_length_ms, "Frame length in "
                   "milliseconds");
      po->Register(name + "::frame-shift", &frame_shift_ms,
                   "Frame shift in milliseconds");
      po->Register(name + "::preemphasis-coefficient", &preemph_coeff,
                   "Coefficient for use in signal preemphasis (deprecated)");
      po->Register(name + "::min-f0", &min_f0,
                   "min. F0 to search for (Hz)");
      po->Register(name + "::max-f0", &max_f0,
                   "max. F0 to search for (Hz)");
      po->Register(name + "::soft-min-f0", &soft_min_f0,
                   "Minimum f0, applied in soft way, must not exceed min-f0");
      po->Register(name + "::penalty-factor", &penalty_factor,
                   "cost factor for FO change.");
      po->Register(name + "::lowpass-cutoff", &lowpass_cutoff,
                   "cutoff frequency for LowPass filter (Hz) ");
      po->Register(name + "::resample-frequency", &resample_freq,
                   "Frequency that we down-sample the signal to.  Must be "
                   "more than twice lowpass-cutoff");
      po->Register(name + "::delta-pitch", &delta_pitch,
                   "Smallest relative change in pitch that our algorithm "
                   "measures");
      po->Register(name + "::nccf-ballast", &nccf_ballast,
                   "Increasing this factor reduces NCCF for quiet frames");
      po->Register(name + "::nccf-ballast-online", &nccf_ballast_online,
                   "This is useful mainly for debug; it affects how the NCCF "
                   "ballast is computed.");
      po->Register(name + "::lowpass-filter-width", &lowpass_filter_width,
                   "Integer that determines filter width of "
                   "lowpass filter, more gives sharper filter");
      po->Register(name + "::upsample-filter-width", &upsample_filter_width,
                   "Integer that determines filter width when upsampling NCCF");
      po->Register(name + "::frames-per-chunk", &frames_per_chunk,
                   "Only relevant for "
                   "offline pitch extraction (e.g. compute-kaldi-pitch-feats), "
                   "you can set it to a small nonzero value, such as 10, for "
                   "better feature compatibility with online decoding (affects "
                   "energy normalization in the algorithm)");
      po->Register(name + "::simulate-first-pass-online",
                   &simulate_first_pass_online,
                   "If true, compute-kaldi-pitch-feats will output features "
                   "that correspond to what an online decoder would see in the "
                   "first pass of decoding-- not the final version of the "
                   "features, which is the default.  Relevant if "
                   "--frames-per-chunk > 0");
      po->Register(name + "::recompute-frame", &recompute_frame, "Only relevant for "
                   "online pitch extraction, or for compatibility with online "
                   "pitch extraction.  A non-critical parameter; the frame at "
                   "which we recompute some of the forward pointers, after "
                   "revising our estimate of the signal energy.  Relevant if"
                   "--frames-per-chunk > 0");
      po->Register(name + "::max-frames-latency", &max_frames_latency,
                   "Maximum number "
                   "of frames of latency that we allow pitch tracking to "
                   "introduce into the feature processing (affects output only "
                   "if --frames-per-chunk > 0 and "
                   "--simulate-first-pass-online=true");
      po->Register(name + "::snip-edges", &snip_edges,
                   "If this is set to false, the "
                   "incomplete frames near the ending edge won't be snipped, so "
                   "that the number of frames is the file size divided by the "
                   "frame-shift. This makes different types of features give the "
                   "same number of frames.");

    }
    /// Returns the window-size in samples, after resampling.  This is the
    /// "basic window size", not the full window size after extending by max-lag.
    int NccfWindowSize() const {
      return static_cast<int32>(resample_freq * 0.001 * frame_length_ms);
    }
    /// Returns the window-shift in samples, after resampling.
    int NccfWindowShift() const {
      return static_cast<int32>(resample_freq * 0.001 * frame_shift_ms);
    }
  };

  struct ProcessPitchOptions {
    float pitch_scale;  // the final normalized-log-pitch feature is scaled
    // with this value
    float pov_scale;    // the final POV feature is scaled with this value
    float pov_offset;   // An offset that can be added to the final POV
    // feature (useful for online-decoding, where we don't
    // do CMN to the pitch-derived features.

    float delta_pitch_scale;
    float delta_pitch_noise_stddev;  // stddev of noise we add to delta-pitch
    int normalization_left_context;  // left-context used for sliding-window
    // normalization
    int normalization_right_context; // this should be reduced in online
    // decoding to reduce latency

    int delta_window;
    int delay;

    bool add_pov_feature;
    bool add_normalized_log_pitch;
    bool add_delta_pitch;
    bool add_raw_log_pitch;

    ProcessPitchOptions() :
      pitch_scale(2.0f),
      pov_scale(2.0f),
      pov_offset(0.0f),
      delta_pitch_scale(10.0f),
      delta_pitch_noise_stddev(0.005f),
      normalization_left_context(75),
      normalization_right_context(75),
      delta_window(2),
      delay(0),
      add_pov_feature(true),
      add_normalized_log_pitch(true),
      add_delta_pitch(true),
      add_raw_log_pitch(false) {
    }


    void Register(ParseOptions *po, const std::string name) {
      po->Register(name + "::pitch-scale", &pitch_scale,
                   "Scaling factor for the final normalized log-pitch value");
      po->Register(name + "::pov-scale", &pov_scale,
                   "Scaling factor for final POV (probability of voicing) "
                   "feature");
      po->Register(name + "::pov-offset", &pov_offset,
                   "This can be used to add an offset to the POV feature. "
                   "Intended for use in online decoding as a substitute for "
                   " CMN.");
      po->Register(name + "::delta-pitch-scale", &delta_pitch_scale,
                   "Term to scale the final delta log-pitch feature");
      po->Register(name + "::delta-pitch-noise-stddev", &delta_pitch_noise_stddev,
                   "Standard deviation for noise we add to the delta log-pitch "
                   "(before scaling); should be about the same as delta-pitch "
                   "option to pitch creation.  The purpose is to get rid of "
                   "peaks in the delta-pitch caused by discretization of pitch "
                   "values.");
      po->Register(name + "::normalization-left-context",
                   &normalization_left_context,
                   "Left-context (in frames) for moving window normalization");
      po->Register(name + "::normalization-right-context",
                   &normalization_right_context,
                   "Right-context (in frames) for moving window normalization");
      po->Register(name + "::delta-window", &delta_window,
                   "Number of frames on each side of central frame, to use for "
                   "delta window.");
      po->Register(name + "::delay", &delay,
                   "Number of frames by which the pitch information is delayed.");
      po->Register(name + "::add-pov-feature", &add_pov_feature,
                   "If true, the warped NCCF is added to output features");
      po->Register(name + "::add-normalized-log-pitch", &add_normalized_log_pitch,
                   "If true, the log-pitch with POV-weighted mean subtraction "
                   "over 1.5 second window is added to output features");
      po->Register(name + "::add-delta-pitch", &add_delta_pitch,
                   "If true, time derivative of log-pitch is added to output "
                   "features");
      po->Register(name + "::add-raw-log-pitch", &add_raw_log_pitch,
                   "If true, log(pitch) is added to output features");
    }
  };


  class ArbitraryResample {
   public:
    ArbitraryResample(int num_samples_in,
                      float samp_rate_hz,
                      float filter_cutoff_hz,
                      const std::vector<float> &sample_points_secs,
                      int num_zeros);

    int NumSamplesIn() const { return num_samples_in_; }

    size_t NumSamplesOut() const { return weights_.size(); }

    /// This function does the resampling.
    /// input.NumRows() and output.NumRows() should be equal
    /// and nonzero.
    /// input.NumCols() should equal NumSamplesIn()
    /// and output.NumCols() should equal NumSamplesOut().

    void Resample(const xnnFloatRuntimeMatrix &input,
                  xnnFloatRuntimeMatrix *output) const;

    /// This version of the Resample function processes just
    /// one vector.
    void Resample(const std::vector<float> &input,
                  std::vector<float> *output) const;
   private:
    void SetIndexes(const std::vector<float> &sample_points);

    void SetWeights(const std::vector<float> &sample_points);

    float FilterFunc(float t) const;

    int num_samples_in_;
    float samp_rate_in_;
    float filter_cutoff_;
    int num_zeros_;

    std::vector<int> first_index_;  // The first input-sample index that we sum
    // over, for this output-sample index.
    std::vector<std::vector<float> > weights_;
  };



  class LinearResample {
   public:


    //template<class I> static I  Gcd(I m, I n);
    //template<class I> static I  Lcm(I m, I n);

    template<class I> inline static  I Gcd(I m, I n) {
      if (m == 0 || n == 0) {
        if (m == 0 && n == 0) {  // gcd not defined, as all integers are divisors.
          IDEC_ERROR << "Undefined GCD since m = 0, n = 0.";
        }
        return (m == 0 ? (n > 0 ? n : -n) : (m > 0 ? m : -m));
        // return absolute value of whichever is nonzero
      }
      // could use compile-time assertion
      // but involves messing with complex template stuff.
      IDEC_ASSERT(std::numeric_limits<I>::is_integer);
      while (1) {
        m %= n;
        if (m == 0) return (n > 0 ? n : -n);
        n %= m;
        if (n == 0) return (m > 0 ? m : -m);
      }
    }

    /// Returns the least common multiple of two integers.  Will
    /// crash unless the inputs are positive.
    template<class I>  inline static  I Lcm(I m, I n) {
      IDEC_ASSERT(m > 0 && n > 0);
      I gcd = Gcd(m, n);
      return gcd * (m / gcd) * (n / gcd);
    }

    /// Constructor.  We make the input and output sample rates integers, because
    /// we are going to need to find a common divisor.  This should just remind
    /// you that they need to be integers.  The filter cutoff needs to be less
    /// than samp_rate_in_hz/2 and less than samp_rate_out_hz/2.  num_zeros
    /// controls the sharpness of the filter, more == sharper but less efficient.
    /// We suggest around 4 to 10 for normal use.
    LinearResample(int samp_rate_in_hz,
                   int samp_rate_out_hz,
                   float filter_cutoff_hz,
                   int num_zeros);


    void Resample(const std::vector<float> &input,
                  bool flush,
                  std::vector<float> *output);

    /// Calling the function Reset() resets the state of the object prior to
    /// processing a new signal; it is only necessary if you have called
    /// Resample(x, y, false) for some signal, leading to a remainder of the
    /// signal being called, but then abandon processing the signal before calling
    /// Resample(x, y, true) for the last piece.  Call it unnecessarily between
    /// signals will not do any harm.
    void Reset();

   private:
    /// This function outputs the number of output samples we will output
    /// for a signal with "input_num_samp" input samples.  If flush == true,
    /// we return the largest n such that
    /// (n/samp_rate_out_) is in the interval [ 0, input_num_samp/samp_rate_in_ ),
    /// and note that the interval is half-open.  If flush == false,
    /// define window_width as num_zeros / (2.0 * filter_cutoff_);
    /// we return the largest n such that (n/samp_rate_out_) is in the interval
    /// [ 0, input_num_samp/samp_rate_in_ - window_width ).
    int64 GetNumOutputSamples(int64 input_num_samp, bool flush) const;


    /// Given an output-sample index, this function outputs to *first_samp_in the
    /// first input-sample index that we have a weight on (may be negative),
    /// and to *samp_out_wrapped the index into weights_ where we can get the
    /// corresponding weights on the input.
    inline void GetIndexes(int64 samp_out,
                           int64 *first_samp_in,
                           int *samp_out_wrapped) const;

    void SetRemainder(const std::vector<float> &input);

    void SetIndexesAndWeights();

    float FilterFunc(float) const;

    // The following variables are provided by the user.
    int samp_rate_in_;
    int samp_rate_out_;
    float filter_cutoff_;
    int num_zeros_;

    int input_samples_in_unit_;   ///< The number of input samples in the
    ///< smallest repeating unit: num_samp_in_ =
    ///< samp_rate_in_hz / Gcd(samp_rate_in_hz,
    ///< samp_rate_out_hz)
    int output_samples_in_unit_;  ///< The number of output samples in the
    ///< smallest repeating unit: num_samp_out_ =
    ///< samp_rate_out_hz / Gcd(samp_rate_in_hz,
    ///< samp_rate_out_hz)


    /// The first input-sample index that we sum over, for this output-sample
    /// index.  May be negative; any truncation at the beginning is handled
    /// separately.  This is just for the first few output samples, but we can
    /// extrapolate the correct input-sample index for arbitrary output samples.
    std::vector<int> first_index_;

    /// Weights on the input samples, for this output-sample index.
    std::vector<std::vector<float> > weights_;

    // the following variables keep track of where we are in a particular signal,
    // if it is being provided over multiple calls to Resample().

    int64 input_sample_offset_;  ///< The number of input samples we have
    ///< already received for this signal
    ///< (including anything in remainder_)
    int64 output_sample_offset_;  ///< The number of samples we have already
    ///< output for this signal.
    std::vector<float> input_remainder_;  ///< A small trailing part of the
    ///< previously seen input signal.
  };


  // class PitchFrameInfo is used inside class OnlinePitchFeatureImpl.
  // It stores the information we need to keep around for a single frame
  // of the pitch computation.
  class PitchFrameInfo {
   public:

    //bool pitch_use_naive_search; // This is used in unit-tests

    void Cleanup(PitchFrameInfo *prev_frame);


    void SetBestState(int best_state,
                      std::vector<std::pair<int, float> >::iterator lag_nccf_iter);


    int ComputeLatency(int max_latency);

    /// This function updates
    bool UpdatePreviousBestState(PitchFrameInfo *prev_frame);

    /// This constructor is used for frame -1; it sets the costs to be all zeros
    /// the pov_nccf's to zero and the backpointers to -1.
    explicit PitchFrameInfo(int num_states);

    /// This constructor is used for subsequent frames (not -1).
    PitchFrameInfo(PitchFrameInfo *prev);

    /// Record the nccf_pov value.
    ///  @param  nccf_pov     The nccf as computed for the POV computation (without ballast).
    void SetNccfPov(const std::vector<float> &nccf_pov);

    void ComputeBacktraces(const PitchExtractionOptions &opts,
                           const std::vector<float> &nccf_pitch,
                           const std::vector<float> &lags,
                           const std::vector<float> &prev_forward_cost,
                           std::vector<std::pair<int, int> > *index_info,
                           std::vector<float> *this_forward_cost);


    void ComputeLocalCost(const std::vector<float> &nccf_pitch,
                          const std::vector<float> &lags,
                          const PitchExtractionOptions &opts,
                          std::vector<float> *local_cost);
   private:
    // struct StateInfo is the information we keep for a single one of the
    // log-spaced lags, for a single frame.  This is a state in the Viterbi
    // computation.
    struct StateInfo {
      /// The state index on the previous frame that is the best preceding state
      /// for this state.
      int backpointer;
      /// the version of the NCCF we keep for the POV computation (without the
      /// ballast term).
      float pov_nccf;
      StateInfo() : backpointer(0), pov_nccf(0.0f) {}
    };
    std::vector<StateInfo> state_info_;
    /// the state index of the first entry in "state_info"; this will initially be
    /// zero, but after cleanup might be nonzero.
    int state_offset_;

    /// The current best state in the backtrace from the end.
    int cur_best_state_;

    /// The structure for the previous frame.
    PitchFrameInfo *prev_info_;



  };


  struct NccfInfo {

    std::vector<float> nccf_pitch_resampled;  // resampled nccf_pitch
    float avg_norm_prod; // average value of e1 * e2.
    float mean_square_energy;  // mean_square energy we used when computing the
    // original ballast term for
    // "nccf_pitch_resampled".

    NccfInfo(float avg_norm_prod,
             float mean_square_energy) :
      avg_norm_prod(avg_norm_prod),
      mean_square_energy(mean_square_energy) {
    }
  };

  // We could inherit from OnlineBaseFeature as we have the same interface,
  // but this will unnecessary force a lot of our functions to be virtual.
  class OnlinePitchFeatureImpl {
   public:
    explicit OnlinePitchFeatureImpl(const PitchExtractionOptions &opts);

    int Dim() const { return 2; }

    int NumFramesReady() const;

    bool IsLastFrame(int frame) const;

    void GetFrame(int frame, std::vector<float> *feat);

    void AcceptWaveform(float sampling_rate,
                        const std::vector<float> &waveform);

    void InputFinished();

    ~OnlinePitchFeatureImpl();


    // Copy-constructor, can be used to obtain a new copy of this object,
    // any state from this utterance.
    OnlinePitchFeatureImpl(const OnlinePitchFeatureImpl &other);

    void SelectLags(const PitchExtractionOptions &opts, std::vector<float> &lags);

    void ComputeCorrelation(const std::vector<float> &wave,
                            int first_lag, int last_lag,
                            int nccf_window_size,
                            std::vector<float> *inner_prod,
                            std::vector<float> *norm_prod);

    /// return abs(a - b) <= relative_tolerance * (abs(a)+abs(b)).
    static inline bool ApproxEqual(float a, float b,
                                   float relative_tolerance = 0.001) {
      // a==b handles infinities.
      if (a == b) return true;
      float diff = std::abs(a - b);
      if (diff == std::numeric_limits<float>::infinity()
          || diff != diff) return false; // diff is +inf or nan.
      return (diff <= relative_tolerance*(std::abs(a) + std::abs(b)));
    }

    void ComputeNccf(const std::vector<float> &inner_prod,
                     const std::vector<float> &norm_prod,
                     float nccf_ballast,
                     // std::vector<float> &nccf_vec);
                     float *nccf_vec);

   public:

    /// This function works out from the signal how many frames are currently
    /// available to process (this is called from inside AcceptWaveform()).
    /// Note: the number of frames differs slightly from the number the
    /// old pitch code gave.
    /// Note: the number this returns depends on whether input_finished_ == true;
    /// if it is, it will "force out" a final frame or two.
    int32 NumFramesAvailable(int64 num_downsampled_samples, bool snip_edges) const;

    /// This function extracts from the signal the samples numbered from
    /// "sample_index" (numbered in the full downsampled signal, not just this
    /// part), and of length equal to window->Dim().  It uses the data members
    /// downsampled_samples_discarded_ and downsampled_signal_remainder_, as well
    /// as the more recent part of the downsampled wave "downsampled_wave_part"
    /// which is provided.
    ///
    /// @param downsampled_wave_part  One chunk of the downsampled wave,
    ///                      starting from sample-index downsampled_samples_discarded_.
    /// @param sample_index  The desired starting sample index (measured from
    ///                      the start of the whole signal, not just this part).
    /// @param window  The part of the signal is output to here.
    void ExtractFrame(const std::vector<float> &downsampled_wave_part,
                      int64 frame_index,
                      std::vector<float> *window);


    /// This function is called after we reach frame "recompute_frame", or when
    /// InputFinished() is called, whichever comes sooner.  It recomputes the
    /// backtraces for frames zero through recompute_frame, if needed because the
    /// average energy of the signal has changed, affecting the nccf ballast term.
    /// It works out the average signal energy from
    /// downsampled_samples_processed_, signal_sum_ and signal_sumsq_ (which, if
    /// you see the calling code, might include more frames than just
    /// "recompute_frame", it might include up to the end of the current chunk).
    void RecomputeBacktraces();


    /// This function updates downsampled_signal_remainder_,
    /// downsampled_samples_processed_, signal_sum_ and signal_sumsq_; it's called
    /// from AcceptWaveform().
    void UpdateRemainder(const std::vector<float> &downsampled_wave_part);


    // The following variables don't change throughout the lifetime
    // of this object.
    PitchExtractionOptions opts_;

    // the first lag of the downsampled signal at which we measure NCCF
    int nccf_first_lag_;
    // the last lag of the downsampled signal at which we measure NCCF
    int nccf_last_lag_;

    // The log-spaced lags at which we will resample the NCCF
    std::vector<float> lags_;

    // This object is used to resample from evenly spaced to log-evenly-spaced
    // nccf values.  It's a pointer for convenience of initialization, so we don't
    // have to use the initializer from the constructor.
    ArbitraryResample *nccf_resampler_;

    // The following objects may change during the lifetime of this object.

    // This object is used to resample the signal.
    LinearResample *signal_resampler_;

    // frame_info_ is indexed by [frame-index + 1].  frame_info_[0] is an object
    // that corresponds to frame -1, which is not a real frame.
    std::vector<PitchFrameInfo *> frame_info_;


    // nccf_info_ is indexed by frame-index, from frame 0 to at most
    // opts_.recompute_frame - 1.  It contains some information we'll
    // need to recompute the tracebacks after getting a better estimate
    // of the average energy of the signal.
    std::vector<NccfInfo *> nccf_info_;

    // Current number of frames which we can't output because Viterbi has not
    // converged for them, or opts_.max_frames_latency if we have reached that
    // limit.
    int frames_latency_;

    // The forward-cost at the current frame (the last frame in frame_info_);
    // this has the same dimension as lags_.  We normalize each time so
    // the lowest cost is zero, for numerical accuracy and so we can use float.
    std::vector<float> forward_cost_;

    // stores the constant part of forward_cost_.
    double forward_cost_remainder_;

    // The resampled-lag index and the NCCF (as computed for POV, without ballast
    // term) for each frame, as determined by Viterbi traceback from the best
    // final state.
    std::vector<std::pair<int, float> > lag_nccf_;

    bool input_finished_;

    /// sum-squared of previously processed parts of signal; used to get NCCF
    /// ballast term.  Denominator is downsampled_samples_processed_.
    double signal_sumsq_;

    /// sum of previously processed parts of signal; used to do mean-subtraction
    /// when getting sum-squared, along with signal_sumsq_.
    double signal_sum_;

    /// downsampled_samples_processed is the number of samples (after
    /// downsampling) that we got in previous calls to AcceptWaveform().
    int64 downsampled_samples_processed_;
    /// This is a small remainder of the previous downsampled signal;
    /// it's used by ExtractFrame for frames near the boundary of two
    /// waveforms supplied to AcceptWaveform().
    std::vector<float> downsampled_signal_remainder_;


    std::vector<float> sub_window_;
  };


  class OnlineFeatureInterface {
   public:
    virtual int Dim() const = 0; /// returns the feature dimension.

    /// Returns the total number of frames, since the start of the utterance, that
    /// are now available.  In an online-decoding context, this will likely
    /// increase with time as more data becomes available.
    virtual int NumFramesReady() const = 0;

    /// Returns true if this is the last frame.  Frame indices are zero-based, so the
    /// first frame is zero.  IsLastFrame(-1) will return false, unless the file
    /// is empty (which is a case that I'm not sure all the code will handle, so
    /// be careful).  This function may return false for some frame if
    /// we haven't yet decided to terminate decoding, but later true if we decide
    /// to terminate decoding.  This function exists mainly to correctly handle
    /// end effects in feature extraction, and is not a mechanism to determine how
    /// many frames are in the decodable object (as it used to be, and for backward
    /// compatibility, still is, in the Decodable interface).
    virtual bool IsLastFrame(int frame) const = 0;

    /// Gets the feature vector for this frame.  Before calling this for a given
    /// frame, it is assumed that you called NumFramesReady() and it returned a
    /// number greater than "frame".  Otherwise this call will likely crash with
    /// an assert failure.  This function is not declared const, in case there is
    /// some kind of caching going on, but most of the time it shouldn't modify
    /// the class.
    virtual void GetFrame(int frame, std::vector<float> &feat) = 0;

    /// Virtual destructor.  Note: constructors that take another member of
    /// type OnlineFeatureInterface are not expected to take ownership of
    /// that pointer; the caller needs to keep track of that manually.
    virtual ~OnlineFeatureInterface() {}
  };

  /// Add a virtual class for "source" features such as MFCC or PLP or pitch
  /// features.
  class OnlineBaseFeature : public OnlineFeatureInterface {
   public:
    /// This would be called from the application, when you get more wave data.
    /// Note: the sampling_rate is typically only provided so the code can assert
    /// that it matches the sampling rate expected in the options.
    virtual void AcceptWaveform(float sampling_rate,
                                const std::vector<float> &waveform) = 0;

    /// InputFinished() tells the class you won't be providing any
    /// more waveform.  This will help flush out the last few frames
    /// of delta or LDA features (it will typically affect the return value
    /// of IsLastFrame.
    virtual void InputFinished() = 0;
  };

  // Note: to start on a new waveform, just construct a new version
  // of this object.
  class OnlinePitchFeature : public OnlineBaseFeature {
   public:
    explicit OnlinePitchFeature(const PitchExtractionOptions &opts);

    virtual int Dim() const { return 2; /* (NCCF, pitch) */ }

    virtual int NumFramesReady() const;

    virtual bool IsLastFrame(int frame) const;

    /// Outputs the two-dimensional feature consisting of (pitch, NCCF).  You
    /// should probably post-process this using class OnlineProcessPitch.
    virtual void GetFrame(int frame, std::vector<float> &feat);

    virtual void AcceptWaveform(float sampling_rate,
                                const std::vector<float> &waveform);

    virtual void InputFinished();

    virtual ~OnlinePitchFeature();

   private:
    OnlinePitchFeatureImpl *impl_;
  };



  /// This online-feature class implements post processing of pitch features.
  /// Inputs are original 2 dims (nccf, pitch).  It can produce various
  /// kinds of outputs, using the default options it will be (pov-feature,
  /// normalized-log-pitch, delta-log-pitch).
  class OnlineProcessPitch : public OnlineFeatureInterface {
   public:
    virtual int32 Dim() const { return dim_; }

    virtual bool IsLastFrame(int32 frame) const {
      if (frame <= -1)
        return src_->IsLastFrame(-1);
      else if (frame < opts_.delay)
        return src_->IsLastFrame(-1) == true ? false : src_->IsLastFrame(0);
      else
        return src_->IsLastFrame(frame - opts_.delay);
    }

    virtual int32 NumFramesReady() const;

    virtual void GetFrame(int32 frame, std::vector<float> &feat);

    virtual ~OnlineProcessPitch() {}

    // Does not take ownership of "src".
    OnlineProcessPitch(const ProcessPitchOptions &opts,
                       OnlineFeatureInterface *src);


    /// Computes and returns the POV feature for this frame.
    /// Called from GetFrame().
    inline float GetPovFeature(int32 frame) const;

    /// Computes and returns the delta-log-pitch feature for this frame.
    /// Called from GetFrame().
    inline float GetDeltaPitchFeature(int32 frame);

    /// Computes and returns the raw log-pitch feature for this frame.
    /// Called from GetFrame().
    inline float GetRawLogPitchFeature(int32 frame) const;

    /// Computes and returns the mean-subtracted log-pitch feature for this frame.
    /// Called from GetFrame().
    inline float GetNormalizedLogPitchFeature(int32 frame);

    /// Computes the normalization window sizes.
    inline void GetNormalizationWindow(int32 frame,
                                       int32 src_frames_ready,
                                       int32 *window_begin,
                                       int32 *window_end) const;

    /// Makes sure the entry in normalization_stats_ for this frame is up to date;
    /// called from GetNormalizedLogPitchFeature.
    inline void UpdateNormalizationStats(int32 frame);

    static inline float NccfToPovFeature(float n) {
      if (n > 1.0f) {
        n = 1.0f;
      } else if (n < -1.0f) {
        n = -1.0f;
      }
      float f = pow((1.0001f - n), 0.15f) - 1.0f;
      IDEC_ASSERT(f - f == 0);  // check for NaN,inf.
      return f;
    }


    static float NccfToPov(float n) {
      float ndash = fabs(n);
      if (ndash > 1.0f) ndash =
          1.0f;  // just in case it was slightly outside [-1, 1]

      float r = (float)(-5.2 + 5.4 * exp(7.5 * (ndash - 1.0)) + 4.8 * ndash -
                        2.0 * exp(-10.0 * ndash) + 4.2 * exp(20.0 * (ndash - 1.0)));
      // r is the approximate log-prob-ratio of voicing, log(p/(1-p)).
      float p = (float)(1.0 / (1 + exp(-1.0 * r)));
      IDEC_ASSERT(p - p == 0);  // Check for NaN/inf
      return p;
    }

    // State for thread-safe random number generator
    struct RandomState {
      RandomState();
      unsigned seed;
    };


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

    /// Returns a random number strictly between 0 and 1.
    inline static float RandUniform(struct RandomState *state = NULL) {
      return static_cast<float>((Rand(state) + 1.0) / (RAND_MAX + 2.0));
    }
#ifdef _MSC_VER
#pragma warning (disable: 4244)
#endif
    inline static float RandGauss(struct RandomState *state = NULL) {
      return static_cast<float>(sqrtf(-2 * logf(RandUniform(state)))
                                * cosf(2 * M_PI*RandUniform(state)));
    }


    void ComputeDeltas(const DeltaFeaturesOptions &delta_opts,
                       xnnFloatRuntimeMatrix &input_features,
                       xnnFloatRuntimeMatrix &output_features) {

      output_features.Resize(input_features.NumRows()
                             *(delta_opts.order + 1), input_features.NumCols());

      window_ = delta_opts.window;
      order_ = delta_opts.order;

      for (int32 frame = 0; frame < static_cast<int32>(input_features.NumCols());
           frame++) {
        IDEC_ASSERT(frame < (int)input_features.NumCols());
        int32 num_frames = (int)input_features.NumCols();
        int feat_dim = (int)input_features.NumRows();

        IDEC_ASSERT(static_cast<int32>(output_features.NumRows()) == feat_dim *
                    (order_ + 1));

        float *output_features_row = output_features.Col(frame);

        for (int32 i = 0; i <= order_; i++) {
          const std::vector<float> &scales = scales_[i];
          int32 max_offset = (int)(scales.size() - 1) / 2;
          float *output = output_features_row + i*feat_dim;
          for (int32 j = -max_offset; j <= max_offset; j++) {
            // if asked to read
            int32 offset_frame = frame + j;
            if (offset_frame < 0) offset_frame = 0;
            else if (offset_frame >= num_frames)
              offset_frame = num_frames - 1;
            float scale = scales[j + max_offset];
            if (scale != 0.0) {
              for (int t = 0; t < feat_dim; t++) {
                output[t] += scale * input_features.Col(offset_frame)[t];
              }
            }

          }
        }

      }
    }

   private:
    int window_;
    int order_;
    std::vector<std::vector <float> > scales_;

    static const int32 kRawFeatureDim = 2;  // input: (nccf, pitch)

    ProcessPitchOptions opts_;
    OnlineFeatureInterface *src_;
    int32 dim_;  // Output feature dimension, set in initializer.

    struct NormalizationStats {
      int32 cur_num_frames;      // value of src_->NumFramesReady() when
      // "mean_pitch" was set.
      bool input_finished;       // true if input data was finished when
      // "mean_pitch" was computed.
      double sum_pov;            // sum of pov over relevant range
      double sum_log_pitch_pov;  // sum of log(pitch) * pov over relevant range

      NormalizationStats() : cur_num_frames(-1), input_finished(false),
        sum_pov(0.0), sum_log_pitch_pov(0.0) {
      }
    };

    std::vector<float> delta_feature_noise_;

    std::vector<NormalizationStats> normalization_stats_;


  };

 protected:
  PitchExtractionOptions pitch_opts_;
  ProcessPitchOptions process_opts_;

  OnlinePitchFeature *pitch_extractor;
  OnlineProcessPitch *post_process;

  int cur_frame_;

  std::vector <float > input_data_;
  std::vector <float > output_data_;


  int samp_per_frame;
  int samp_per_chunk;
 public:


  void SelectLags(const PitchExtractionOptions &opts, std::vector<float> &lags);

  FrontendComponent_Waveform2Pitch(ParseOptions &po,
                                   const std::string name = "Waveform2Pitch") : FrontendComponentInterface(po,
                                         name) {
    // register optional configs
    pitch_opts_.Register(&po, name);
    process_opts_.Register(&po, name);
  }

  ~FrontendComponent_Waveform2Pitch() {
    if (NULL != pitch_extractor) {
      delete(pitch_extractor);
      pitch_extractor = NULL;
    }

    if (NULL != post_process) {
      delete(post_process);
      post_process = NULL;
    }

    cur_frame_ = 0;
  }

  virtual void Init() {
    FrontendComponentInterface::Init();

    // pitch extract function inner init
    pitch_extractor = new OnlinePitchFeature(pitch_opts_);

    if (pitch_opts_.simulate_first_pass_online) {
      IDEC_ASSERT(pitch_opts_.frames_per_chunk > 0 &&
                  "--simulate-first-pass-online option does not make sense "
                  "unless you specify --frames-per-chunk");
    }
    post_process = new OnlineProcessPitch(process_opts_, pitch_extractor);

    // prepare output-related staff
    output_dim_ = post_process->Dim();
    output_buff_.Resize(output_dim_, 1);

    if (input_dim_ == 0)
      IDEC_ERROR << "input dimension not set";

    output_data_.resize(output_dim_);

    samp_per_frame = (int)(pitch_opts_.samp_freq * 1.0e-03 *
                           pitch_opts_.frame_shift_ms);
    samp_per_chunk = (int)(pitch_opts_.frames_per_chunk * pitch_opts_.samp_freq *
                           1.0e-03 * pitch_opts_.frame_shift_ms);

    cur_frame_ = 0;
  }


  virtual bool Process() {
    if (input_buf_.empty())
      return(false);

    xnnFloatRuntimeMatrixCircularBuffer &inputBuff = input_buf_[0];


    while (inputBuff.NumCols() >= (size_t)pitch_opts_.frames_per_chunk) {
      input_data_.resize(samp_per_chunk);
      for (int i = 0; i < pitch_opts_.frames_per_chunk; i++) {
        for (int j = 0; j < samp_per_frame; j++) {
          input_data_[i * samp_per_frame + j] = inputBuff.Col(i)[j];
        }
      }

      ComputeAndProcessKaldiPitch(input_data_);



      if (cur_frame_ < post_process->NumFramesReady()) {
        output_data_.resize(output_dim_);
        for (; cur_frame_ < post_process->NumFramesReady(); cur_frame_++) {
          post_process->GetFrame(cur_frame_, output_data_);
          // push one processed frame to succeeding components, return on error
          if (!SendOneFrameToSucceedingComponents(&output_data_[0]))
            return(false);
        }
      }


      for (int i = 0; i < pitch_opts_.frames_per_chunk; i++) {
        inputBuff.PopfrontOneColumn();
      }

      output_data_.clear();
      input_data_.clear();

    }
#if 0

    ComputeAndProcessKaldiPitch(inputBuff.Col(0), outputBuff_.Col(0));

    if (!SendOneFrameToSucceedingComponents())
      return(false);

    inputBuff.PopfrontOneColumn();
#endif



    return(true);

  }

  virtual bool Finalize() {
    IDEC_ASSERT(!input_buf_.empty());

    xnnFloatRuntimeMatrixCircularBuffer &input_buff = input_buf_[0];
    if (input_buff.NumCols() > 0) {
      int last_data_size = (int)(input_buff.NumCols() * pitch_opts_.samp_freq * 1.0e-03 * pitch_opts_.frame_shift_ms);
      input_data_.resize(last_data_size);

      for (size_t i = 0; i < input_buff.NumCols(); i++) {
        for (int j = 0; j < samp_per_frame; j++) {
          input_data_[i * samp_per_frame + j] = input_buff.Col(i)[j];
        }
      }

      ComputeAndProcessKaldiPitch(input_data_);


      for (size_t i = 0; i < input_buff.NumCols(); i++) {
        input_buff.PopfrontOneColumn();
      }
    }

    if (cur_frame_ > 0)
      pitch_extractor->InputFinished();

    if (cur_frame_ < post_process->NumFramesReady()) {
      float *data = new float[output_dim_];
      output_data_.resize(output_dim_);
      for (; cur_frame_ < post_process->NumFramesReady(); cur_frame_++) {
        post_process->GetFrame(cur_frame_, output_data_);
        for (int j = 0; j < output_dim_; j++) {
          data[j] = output_data_[j];
        }
        // push one processed frame to succeeding components, return on error
        if (!SendOneFrameToSucceedingComponents(data))
          return(false);
      }

      delete(data);
      data = NULL;
    }


    output_data_.clear();
    input_data_.clear();


    if (NULL != pitch_extractor) {
      delete(pitch_extractor);
      pitch_extractor = NULL;
    }

    if (NULL != post_process) {
      delete(post_process);
      post_process = NULL;
    }


    // pitch extract function inner init
    pitch_extractor = new OnlinePitchFeature(pitch_opts_);

    if (pitch_opts_.simulate_first_pass_online) {
      IDEC_ASSERT(pitch_opts_.frames_per_chunk > 0 &&
                  "--simulate-first-pass-online option does not make sense "
                  "unless you specify --frames-per-chunk");
    }
    post_process = new OnlineProcessPitch(process_opts_, pitch_extractor);
    cur_frame_ = 0;

    return(true);
  }


  // pitch extract function
  void ComputeAndProcessKaldiPitch(std::vector <float > &input_data);

  template<typename Real> static inline  Real vec_dot(const std::vector<Real> &a,
      const std::vector<Real> &b) {
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

};
}
#endif


