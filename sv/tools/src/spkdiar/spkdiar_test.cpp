#define _CRT_SECURE_NO_WARNINGS

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <exception>
#include "spkdiar_impl.h"
#include "spkdiar_serialize.h"
#include "resource_manager.h"
#include "spkdiar.h"

using namespace std;
using namespace alspkdiar;

void SpkDiarizationTest(SpeakerDiarization *inst,
                        const std::string &wave_file,
                        const std::string &out_result) {
  int ret = ALS_OK;
  inst->BeginSpkDiar();
  ret = inst->SpkDiar(wave_file, out_result);
  if (ret != ALS_OK) {
    std::cerr << "speaker diarization error in file " + wave_file << std::endl;
  }
  inst->EndSpkDiar();
  return;
}

void ReadFileList(const std::string lst, std::vector<std::string> &files) {
  ifstream ifs(lst.c_str());
  if (!ifs) {
    std::cerr << "Cannot VAD file: " << lst.c_str() << std::endl;
    ifs.close();
    return;
  }
  std::string s;
  while (getline(ifs, s)) {
    files.push_back(s);
  }

  ifs.close();
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    cout << argc;
    std::cerr << "spkDiar <conf> <wave list> <output-dir> <is_Test>" << std::endl;
    return -1;
  }

  //string path = "D:\\sv\\data\\spkdiar\\mdl\\final-speech.dubm.16";
  //DiagGmm diag_gmm;
  //diag_gmm.Read(path);
  //string out_path = "D:\\sv\\data\\spkdiar\\mdl\\spkdiar-speech.ubm.16";
  //diag_gmm.WriteNLSFormat(out_path);

  string conf_dir = argv[1];
  string wave_file_list = argv[2];
  string out_result = argv[3];
  string str_flag_test = argv[4];
  //string out_result;
  //wave_file_list = "common.lst";
  out_result = "D:\\sv\\data\\spkdiar\\";
  //wave_file_list = "D:\\sv\\data\\spkdiar\\sandi_result\\wav.lst";
  //wave_file_list = "D:\\sv\\data\\spkdiar\\common.lst";
  wave_file_list = "D:\\spkdiar_test\\16k\\wav.lst";
  //wave_file_list = "D:\\sv\\data\\spkdiar\\wav1.lst";
  //wave_file_list = "D:\\sv\\data\\spkdiar\\spkdiar_result\\spkdiar100\\wav.lst";
  //wave_file_list = "D:\\sv\\data\\spkdiar\\tiananrenshou_poc_result\\trans_wav\\wav.lst";
  std::vector<std::string> wav_files, labels;
  ReadFileList(wave_file_list, wav_files);

  //Denoise denoise("", "fe_.conf");
  //for (int i = 0; i < wav_files.size(); ++i) {
	 // std::vector<char> wave;
	 // wave.reserve(1024);
	 // Serialize::ReadWave("" + wav_files[i], wave);
	 // denoise.FE(&wave[0], wave.size());
	 // SegCluster cluster;
	 // denoise.Process(cluster);
	 // if (cluster.Size() == 0) {
		//  continue;
	 // }
	 // string wave_path = "D:\\sv\\data\\spkdiar\\";
	 // string label_path = "D:\\sv\\data\\spkdiar\\";
	 // Serialize ser;
	 // string file_name;
	 // ser.GetFileName(wav_files[i], file_name);
	 // wave_path += file_name + ".music.wav";
	 // label_path += file_name + ".music.lbl";
	 // ser.SaveWaveAndLabel(&wave[0], wave_path, label_path, cluster);
  //}

  /*
  Denoise denoise("", "fe_dtmf.conf");
  for (int i = 0; i < wav_files.size(); ++i) {
    std::vector<char> wave;
    wave.reserve(1024);
    Serialize::ReadWave("" + wav_files[i], wave);
    denoise.FE(&wave[0], wave.size());
    SegCluster cluster;
    denoise.Process(cluster);
    if (cluster.Size() == 0) {
  	  continue;
    }
    string wave_path = "D:\\sv\\data\\spkdiar\\";
    string label_path = "D:\\sv\\data\\spkdiar\\";
    Serialize ser;
    string file_name;
    ser.GetFileName(wav_files[i], file_name);
    wave_path += file_name + ".wav";
    label_path += file_name + ".lbl";
    ser.SaveWaveAndLabel(&wave[0], wave_path, label_path, cluster);
  }
  */

  void *handler = Init(conf_dir.c_str());
  void *inst = CreateInstance(handler);

  //const std::string wave_path = "C:\\Users\\zhuozhu.zz\\Desktop\\科大讯飞转写\\data\\";

  const std::string wave_path = "";
  const std::string label_path = "";

  bool flag_test = true;
  if (str_flag_test == "false") {
    flag_test = false;
  }

  flag_test = true;

  if (flag_test) {
    for (unsigned int i = 0; i < wav_files.size(); ++i) {
      try {
      cout << "process file " << i << endl;
      SpeakerDiarization *spkdiar = static_cast<SpeakerDiarization *>(inst);
      SpkDiarizationTest(spkdiar, wave_path + wav_files[i], out_result);
       } catch (exception e) {
         std::cerr << "Exception Occurred when running " << wav_files[i] << std::endl;
       }
    }
  } else {
    std::vector<char> wave;
    wave.reserve(1024);
    for (unsigned int i = 0; i < wav_files.size(); ++i) {
      try {
        Serialize::ReadWave(wave_path + wav_files[i], wave);
        AlsSpkdiarResult *res = SpkDiarization(inst, &wave[0], wave.size());
        for (int i = 0; i < res->fragment_num; ++i) {
          cout << res->speech_fragments[i].begin_time << endl;
          cout << res->speech_fragments[i].end_time << endl;
          cout << res->speech_fragments[i].speaker_id << endl;
        }
      } catch (...) {
        std::cout << "Exception Occurred when running " << wav_files[i] << std::endl;
      }
    }
  }

  DestroyInstance(inst);
  UnInit(handler);
  cout << "getchar" << endl;
  getchar();
  return 0;
}
