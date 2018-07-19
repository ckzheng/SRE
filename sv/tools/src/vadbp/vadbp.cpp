#include "als_vad.h"
#include<string>
#include <fstream>
#include <iostream>
#include <vector>
#include "base/idec_common.h"
#include "util/parse-options.h"
#include "util/options-itf.h"
#include "base/log_message.h"
#include "fe/frontend.h"
#include "fe/frontend_pipeline.h"

using namespace std;

class DoVadForKaldi {
 public:
  struct Config {
    int frame_shift_ms;
    int frame_decimate_rate;
    string input_pcm_type;
    string vad_conf_path;
    string vad_mdl_path;
    Config() {
      frame_shift_ms = 10;
      frame_decimate_rate = 1;
    }

    void Register(idec::OptionsItf *po, std::string prefix = "") {
      po->Register("Waveform2Filterbank::frame-shift", &frame_shift_ms,
                   "Frame shift in milliseconds");
      po->Register("Waveform2Filterbank::frame-decimate", &frame_decimate_rate,
                   "Frame shift in milliseconds");
      po->Register("input-type", &input_pcm_type, "input pcm data type.");
      po->Register("vad-conf-path", &vad_conf_path, "input pcm data type.");
      po->Register("vad-mdl-path", &vad_mdl_path, "Frame shift in milliseconds");
    }
  };

  struct Fragment {
    unsigned int begin_frm;
    unsigned int end_frm;
    Fragment(int begin, int end) {
      begin_frm = begin;
      end_frm = end;
    }
    int NumFrames() const {
      return end_frm - begin_frm;
    }
  };

 public:
  DoVadForKaldi(const string &cfg_file) {
    idec::ParseOptions *po = new idec::ParseOptions("Waveform2Filterbank::");
    idec::IDEC_FE_AUDIOFORMAT enum_input_type;
    Config conf;
    conf.Register(po, "");
    if (!IsExistence(cfg_file)) {
      idec::IDEC_ERROR << "configuration file " << cfg_file <<
                       " does not exist.";
    }

    po->ReadConfigFile(cfg_file);
    idec::IDEC_DELETE(po);
    InputWavTypeStr2Enum(conf.input_pcm_type, enum_input_type_);
    vad_ = CreateVad(conf.vad_conf_path, conf.vad_mdl_path);
    front_end_ = new idec::FrontendPipeline();
    front_end_->Init(cfg_file, "");
    frame_shift_in_ms_ = conf.frame_decimate_rate * conf.frame_shift_ms;
  }

  ~DoVadForKaldi() {
    AlsVad::Destroy(vad_);
    IDEC_DELETE(front_end_);
  }

  void ReadWave(const std::string &fname,
                std::vector<char> &wave) {
    ifstream ifs(fname.c_str(), ios::binary);
    if (!ifs) {
      cerr << "Cannot WAVE file: " << fname;
      ifs.close();
      return;
    }
    wave.clear();
    ifs.seekg(0, ios::end);
    int len = ifs.tellg();
    len -= 44;
    ifs.seekg(44, ios::beg);
    wave.resize(len);
    ifs.read(&wave[0], len);
    ifs.close();
  }

  bool IsExistence(const string &fname) {
    ifstream ifs(fname.c_str());
    if (!ifs) {
      ifs.close();
      return false;
    }
    ifs.close();
    return true;
  }

  AlsVad *CreateVad(const string &vad_conf_path, const string &vad_mdl_path) {
    if (!IsExistence(vad_conf_path)) {
      cerr << "vad conf " << vad_conf_path <<
           " does not exist";
      return NULL;
    }

    if (!IsExistence(vad_mdl_path)) {
      cerr << "vad model " << vad_mdl_path << "does not exist";
      return NULL;
    }

    AlsVadMdlHandle vad_handler = AlsVad::LoadModel(vad_conf_path.c_str(),
                                  vad_mdl_path.c_str());
    AlsVad *vad = AlsVad::CreateFromModel(vad_handler);
    return vad;
  }

  void DoVad(const char *wave, unsigned int len,
             vector<Fragment> &valid_speech, bool utterance_end) {
    AlsVadResult *result = NULL;
    vad_->SetData2((short *)wave, len, utterance_end);
    result = vad_->DoDetect2();
    int begin_frm, end_frm;
    if (result != NULL) {
      for (int i = 0; i < result->num_segments; i++) {
        AlsVadSpeechBuf &buf(result->speech_segments[i]);
        begin_frm = buf.start_ms / frame_shift_in_ms_;
        end_frm = buf.end_ms / frame_shift_in_ms_;
        if ((valid_speech.size() != 0)
            && (valid_speech[valid_speech.size() - 1].end_frm == begin_frm)) {
          valid_speech[valid_speech.size() - 1].end_frm = end_frm;
        } else {
          valid_speech.push_back(Fragment(begin_frm, end_frm));
        }
      }
      AlsVadResult_Release(&result);
      result = NULL;
    }
  }

  // VadProcess(vad, wave, vad_result, frame_shift_in_ms);
  void VadProcess(vector<char> &wave, vector<char> &vad_result) {
    vad_->BeginUtterance();
    front_end_->BeginUtterance();
    vector<Fragment> valid_speech;
    // do vad, get the frame_by_frame result
    const size_t wavblocksize = 400;
    const size_t len = wave.size();
    unsigned int loop = len / wavblocksize;
    if ((len % wavblocksize) != 0) {
      loop += 1;
    }

    unsigned int base_ptr = 0, begin_pos = 0, frag_len = 0;
    for (int i = 0; i < loop; ++i) {
      frag_len = (i == loop - 1) ? (len % wavblocksize) : wavblocksize;
      if (frag_len == 0) {
        frag_len = wavblocksize;
      }

      front_end_->PushAudio(&wave[0] + base_ptr, frag_len, enum_input_type_);
      DoVad(&wave[0] + base_ptr, frag_len, valid_speech, false);
      base_ptr += frag_len;
    }

    DoVad(NULL, 0, valid_speech, true);
    vad_->EndUtterance();
    front_end_->EndUtterance();
    int N = front_end_->NumFrames();
    vad_result.resize(N, 0);
    for (int i = 0; i < valid_speech.size(); ++i) {
      for (int j = valid_speech[i].begin_frm; j < valid_speech[i].end_frm; ++j) {
        vad_result[j] = 1;
      }
    }
  }

  //void VadProcess(AlsVad *vad, const vector<char> &wave,
  //                vector<int> &vad_result, int frame_shift_in_ms) {
  //  if (NULL == vad) {
  //    cerr << "Vad is NULL! " << __FILE__ << " , " << __LINE__;
  //    return;
  //  }
  //
  //  vad->SetVoiceStartCallback(NULL, NULL);
  //  vad->SetVoiceDetectedCallback(NULL, NULL);
  //  vad->SetVoiceEndCallback(NULL, NULL);
  //  vad->BeginUtterance();
  //
  //  const size_t wavblocksize = 400;
  //  unsigned int loop = wave.size() / wavblocksize;
  //  const int remainder = wave.size() % wavblocksize;
  //  if (remainder != 0) {
  //    loop += 1;
  //  }
  //
  //  unsigned int base_ptr = 0, frag_len = 0;
  //  bool utterance_end = false;
  //  for (int i = 0; i < loop; ++i) {
  //    // frag_len = (i == loop - 1) ? (wave.size() % wavblocksize) : wavblocksize;
  //    frag_len = wavblocksize;
  //    if ((i == loop - 1) && (remainder != 0)) {
  //      frag_len = remainder;
  //      utterance_end = true;
  //    }
  //
  //    AlsVadResult *result = NULL;
  //    vad->SetData2((short *)(&wave[0] + base_ptr), frag_len, utterance_end);
  //    result = vad->DoDetect2();
  //    if (result != NULL) {
  //      for (int i = 0; i < result->num_segments; i++) {
  //        AlsVadSpeechBuf &buf(result->speech_segments[i]);
  //        int b_frm = buf.start_ms / frame_shift_in_ms;
  //        int e_frm = buf.end_ms / frame_shift_in_ms;
  //        int l_e_frm = vad_result[0];
  //        idec::IDEC_ASSERT(l_e_frm <= b_frm);
  //        for (int i = l_e_frm; i < b_frm; ++i) {
  //          vad_result.push_back(0);
  //        }
  //
  //        idec::IDEC_ASSERT(b_frm < e_frm);
  //        for (int i = b_frm; i < e_frm; ++i) {
  //          vad_result.push_back(1);
  //        }
  //        vad_result[0] = e_frm;
  //      }
  //
  //      AlsVadResult_Release(&result);
  //      result = NULL;
  //    }
  //    base_ptr += frag_len;
  //  }
  //  vad->EndUtterance();
  //}

  int InputWavTypeStr2Enum(const std::string str_input_str,
                           idec::IDEC_FE_AUDIOFORMAT &enum_wav_type) {
    enum_wav_type = idec::FE_8K_16BIT_PCM;
    if (str_input_str == "FE_8K_16BIT_PCM") {
      enum_wav_type = idec::FE_8K_16BIT_PCM;
    } else if (str_input_str == "FE_16K_16BIT_PCM") {
      enum_wav_type = idec::FE_16K_16BIT_PCM;
    } else {
      idec::IDEC_ERROR << "unknown input type " << str_input_str;
    }
    return 0;
  }

 private:
  int frame_shift_in_ms_;
  AlsVad *vad_;
  idec::FrontendPipeline *front_end_;
  idec::IDEC_FE_AUDIOFORMAT enum_input_type_;
};


void SplitStringOnFirstSpace(const std::string &str,
                             std::string *first,
                             std::string *rest) {
  const char *white_chars = " \t\n\r\f\v";
  typedef std::string::size_type I;
  const I npos = std::string::npos;
  I first_nonwhite = str.find_first_not_of(white_chars);
  if (first_nonwhite == npos) {
    first->clear();
    rest->clear();
    return;
  }
  // next_white is first whitespace after first nonwhitespace.
  I next_white = str.find_first_of(white_chars, first_nonwhite);

  if (next_white == npos) {  // no more whitespace...
    *first = std::string(str, first_nonwhite);
    rest->clear();
    return;
  }
  I next_nonwhite = str.find_first_not_of(white_chars, next_white);
  if (next_nonwhite == npos) {
    *first = std::string(str, first_nonwhite, next_white - first_nonwhite);
    rest->clear();
    return;
  }

  I last_nonwhite = str.find_last_not_of(white_chars);
  idec::IDEC_ASSERT(last_nonwhite != npos);  // or coding error.

  *first = std::string(str, first_nonwhite, next_white - first_nonwhite);
  *rest = std::string(str, next_nonwhite, last_nonwhite + 1 - next_nonwhite);
}

bool ReadScriptFile(std::istream &is,
                    bool warn,
                    std::vector < std::pair<std::string, std::string> >
                    *script_out) {
  idec::IDEC_ASSERT(script_out != NULL);
  std::string line;
  int line_number = 0;
  while (getline(is, line)) {
    line_number++;
    const char *c = line.c_str();
    if (*c == '\0') {
      if (warn)
        idec::IDEC_WARN << "Empty " << line_number << "'th line in script file";
      return false;  // Empty line so invalid scp file format..
    }

    std::string key, rest;
    SplitStringOnFirstSpace(line, &key, &rest);

    if (key.empty() || rest.empty()) {
      if (warn)
        idec::IDEC_WARN << "Invalid " << line_number << "'th line in script file"
                        << ":\"" << line << '"';
      return false;
    }
    script_out->resize(script_out->size() + 1);
    script_out->back().first = key;
    script_out->back().second = rest;
  }
  return true;
}

void MakeFilename(const string archive_wxfilename, const size_t streampos,
                  std::string *output) {
  std::ostringstream ss;
  ss << ':' << streampos;
  idec::IDEC_ASSERT(ss.str() != ":-1");
  *output = archive_wxfilename + ss.str();
}

string GetBaseName(const string& path) {
#ifdef _MSC_VER
	const char *c = strrchr(path.c_str(), '\\');
#else
	const char *c = strrchr(path.c_str(), '/');
#endif

	if (c == NULL) {
		c = path.c_str();
	} else {
		c++;
	}

	return string(c);
}

void WriteFloatVector(ofstream& archive_os, const vector<char>&vad_result) {
	archive_os.put('\0');
	archive_os.put('B');
	archive_os << "FV" << " ";
	const char len_c = 4;
	archive_os.put(len_c);
	const int total_len = vad_result.size();
	archive_os.write((const char *)&total_len, sizeof(int));
	for (int i = 0; i < vad_result.size(); ++i) {
		float vad_value = vad_result[i];
		archive_os.write((const char *)&vad_value, sizeof(float));
	}
}

int main(int argc, char *argv[]) {
  const char *usage =
    "Usage: compute-vad <conf-path> <wavs-rspecifier> <vad-wspecifier> \n"
    "e.g.: compute-vad mfcc.conf scp:wav.scp ark,scp:vad.ark,vad.scp\n";
  if (argc != 4) {
    for (int i = 0; i < argc; ++i) {
      cout << argv[i] << " ";
    }
    cout << endl;
    cout << usage << endl;
    exit(1);
  }

  //const string wavs_rsp = "D:\\SourceCode\\NLS\\sid\\sv\\data\\sv\\003221273.mkv_10.15.34.78_41330_1_0_0006.wav";
  //const string wavs_rsp = "wav.scp";
  //const string conf_path = "D:\\SourceCode\\NLS\\sid\\sv\\data\\vadbp\\sv.conf";
  const string conf_path = string(argv[1]);
  const string wavs_rsp = string(argv[2]);
  string vad_wsp = string(argv[3]);
  //const string vad_wsp = "ark,scp:vad.ark,vad.scp";
  bool warn = true;
  std::vector < std::pair<std::string, std::string> > script_out;
  vector< std::pair<std::string, std::string> >::iterator iter;
  DoVadForKaldi vad_kaldi(conf_path);  
  string archive_path, script_path, archive_wxfilename;
  //archive_path = "vad_test.ark";
  //script_path = "vad_test.script";
  //archive_wxfilename = "vad.ark";
  int begin_pos = vad_wsp.find_first_of(':');
  if (begin_pos == -1) {
    idec::IDEC_ERROR << "Invalid format " << vad_wsp;
  }
  ++begin_pos;

  int end_pos = vad_wsp.find_first_of(',', begin_pos);
  if (end_pos == -1) {
    idec::IDEC_ERROR << "Invalid format " << vad_wsp;
  }
  ++end_pos;

  archive_path = vad_wsp.substr(begin_pos, end_pos-begin_pos-1);
  script_path = vad_wsp.substr(end_pos);
  archive_wxfilename = GetBaseName(archive_path);

  ifstream wav_is(wavs_rsp.c_str());
  if (!wav_is) {
    idec::IDEC_ERROR << "Open file" << wavs_rsp << " error.";
  }
  ReadScriptFile(wav_is, warn, &script_out);
  wav_is.close();

  std::ofstream archive_os(archive_path.c_str(), ios::binary);
  if (!archive_os) {
    idec::IDEC_ERROR << "Open file" << archive_path << " error.";
  }

  std::ofstream script_os(script_path.c_str());
  if (!script_os) {
    idec::IDEC_ERROR << "Open file" << script_path << " error.";
  }

  vector<char> wave, vad_result;
  for (iter = script_out.begin(); iter != script_out.end(); ++iter) {
    const string &wave_id = iter->first;
    const string &wave_path = iter->second;
    vad_kaldi.ReadWave(wave_path, wave);
    vad_kaldi.VadProcess(wave, vad_result);
    archive_os << wave_id << ' ';
    std::ofstream::pos_type archive_os_pos = archive_os.tellp();    
	WriteFloatVector(archive_os, vad_result);
	/*archive_os.put('\0');
	archive_os.put('B');
	archive_os << "FV" << " ";
	const char len_c = 4;
	archive_os.put(len_c);
	const int total_len = vad_result.size();
	archive_os.write((const char *)&total_len, sizeof(int));
	for (int i = 0; i < vad_result.size(); ++i) {
	float vad_value = vad_result[i];
	archive_os.write((const char *)&vad_value, sizeof(float));
	}*/
    // rxfilename with offset into the archive,
    // e.g. some_archive_name.ark:431541423
	std::string offset_rxfilename;
	MakeFilename(archive_path, archive_os_pos, &offset_rxfilename);
    script_os << wave_id << ' ' << offset_rxfilename << "\n";
  }
  archive_os.close();
  script_os.close();
}