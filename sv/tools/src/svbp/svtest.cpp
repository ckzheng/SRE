#include "speaker_verification.h"
#include "resource_manager.h"
#include "speaker_verification_impl.h"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>

using namespace std;

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

std::vector<std::string> Tokenizer(const std::string &p_pcstStr, char delim) {
  std::vector<std::string> tokens;
  std::stringstream mySstream(p_pcstStr);
  std::string temp;
  while (getline(mySstream, temp, delim)) {
    tokens.push_back(temp);
  }
  return tokens;
}

void rtrim(string &str) {
  str.erase(str.find_last_not_of(" \t\n\r") + 1);
}

void ReadRegisterList(const string &test_list,
                      map<string, vector<string> > &register_table) {
  ifstream ifs(test_list.c_str(), ios::binary);
  if (!ifs) {
    cerr << "Cannot open file: " << test_list;
    ifs.close();
    return;
  }

  string s, spk_id;
  const char delim = ' ';
  while (getline(ifs, s)) {
    cout << "Read from file: " << s << endl;
    rtrim(s);
    vector <string > v = Tokenizer(s, delim);
    if (v.size() < 2) {
      cout << "skip line " << s << endl;
      continue;
    }
    spk_id = v[0];
    v.erase(v.begin());
    register_table.insert(make_pair(spk_id, v));
  }
}

void ReadTestList(const string &test_list,
                  vector<pair<string, string>> &test_table) {
  ifstream ifs(test_list.c_str(), ios::binary);
  if (!ifs) {
    cerr << "Cannot open file: " << test_list;
    ifs.close();
    return;
  }

  string s, spk_id;
  const char delim = ' ';
  while (getline(ifs, s)) {
    cout << "Read from file: " << s << endl;
    rtrim(s);
    vector <string > v = Tokenizer(s, delim);
    if (v.size() != 2) {
      cout << "skip line " << s << endl;
      continue;
    }
    test_table.push_back(make_pair(v[0], v[1]));
  }
}

void LoadSpeakerInfo(const string &mdl_path, SpeakerInfo *spk_info) {
  ifstream ifs(mdl_path.c_str(), ios::binary);
  ifs.seekg(0, ios::end);
  int mdl_size = ifs.tellg();
  spk_info->data = new char[mdl_size];
  spk_info->length = mdl_size;
  ifs.seekg(0, ios::beg);
  ifs.read(spk_info->data, mdl_size);
  ifs.close();
}

void WriteSpeakerInfo(const string &mdl_path, const string &spk_info) {
  ofstream ofs(mdl_path.c_str(), ios::binary);
  ofs.write(spk_info.c_str(), spk_info.size());
  ofs.close();
}

void WriteTestResult(const string &mdl_path,
                     const vector<string> &test_result) {
  ofstream ofs(mdl_path.c_str(), ios::binary);
  string line;
  for (int i = 0; i < test_result.size(); ++i) {
    line = test_result[i] + "\n";
    ofs.write(line.c_str(), line.size());
  }
  ofs.close();
}

bool IsFileExist(string file_path) {
  ifstream ifs(file_path, ios::binary);
  if (!ifs) {
    ifs.close();
    return false;
  }
  ifs.close();
  return true;
}


bool EndWith(const string &str, const string &strEnd) {
  if (str.empty() || strEnd.empty() || (str.size() < strEnd.size())) {
    return false;
  }
  return str.compare(str.size() - strEnd.size(), strEnd.size(),
                     strEnd) == 0 ? true : false;
}

string RemoveWavSuffix(string path) {
  if (EndWith(path, ".wav") || EndWith(path, ".WAV")) {
    cout << "Warn, already ends with wav suffix." << endl;
    return path.substr(0, path.rfind("."));
  }
  return path;
}

string AddWavSuffix(string path) {
  if (EndWith(path, ".wav") || EndWith(path, ".WAV")) {
    cout << "Warn, already ends with wav suffix." << endl;
    return path;
  }
  return path + ".wav";
}

void RegisterModels(const string &mdl_dir, const string &wave_dir,
                    map<string, vector<string> > &register_table) {
  vector<char> wave_data;
  string spk_id, wave_path, out_mdl_path;
  SpeakerInfo *spk_info = NULL, *spk_info_prev = NULL, *spk_info_next = NULL;
  void *handler = Init("sv.conf", "");
  map <string, vector<string> >::iterator iter;
  for (iter = register_table.begin(); iter != register_table.end(); iter++) {
    spk_id = iter->first;
    spk_id = RemoveWavSuffix(spk_id);
    const vector<string> &waves = iter->second;
    if (waves.size() > 0) {
      wave_path = wave_dir + "/" + AddWavSuffix(waves[0]);
      void *inst = CreateInstance(handler);
      ReadWave(wave_path, wave_data);
      PreProcess(inst, spk_id.c_str());
      ProcessVoiceInput(inst, &wave_data[0], wave_data.size());
      spk_info_prev = PostProcess(inst);
      DestoryInstance(inst);
    }

    spk_info = spk_info_prev;
    if (waves.size() > 1) {
      for (int i = 1; i < waves.size(); ++i) {
        wave_path = wave_dir + "/" + AddWavSuffix(waves[i]);
        void *inst = CreateInstance(handler);
        ReadWave(wave_path, wave_data);
        PreProcess(inst, spk_id.c_str());
        ProcessVoiceInput(inst, &wave_data[0], wave_data.size());
        spk_info_next = PostProcess(inst);
        spk_info = UpdateVoicePrint(inst, spk_info_prev, spk_info_next);
        DestoryInstance(inst);
        DestorySpeakerInfo(spk_info_prev);
        DestorySpeakerInfo(spk_info_next);
        spk_info_prev = spk_info;
      }
    }
    out_mdl_path = mdl_dir + "/" + spk_id;
    string s_mdl(spk_info->data, spk_info->length);
    WriteSpeakerInfo(out_mdl_path, s_mdl);
    DestorySpeakerInfo(spk_info);
  }
  UnInit(handler);
}

void TestEngine(const string &wave_dir, const string &model_dir,
                vector<pair<string, string>> &test_table) {
  void *handler = Init("sv.conf", "");
  void *inst = CreateInstance(handler);
  float score;
  vector<char> wave;
  vector<float> scores;
  vector<string> test_result;
  string spk_id, waves, wave_path, mdl_path, test_mdl_path, test_mdl_info;
  SpeakerInfo *spk_info, *spk_info_orig;
  vector<pair<string, string> >::iterator iter;
  stringstream stream;
  for (iter = test_table.begin(); iter != test_table.end(); iter++) {
    spk_id = iter->first;
    waves = iter->second;
    wave_path = wave_dir + "/" + AddWavSuffix(waves);
    mdl_path = model_dir + "/" + RemoveWavSuffix(spk_id);
    test_mdl_path = model_dir + "/" + RemoveWavSuffix(waves);
    if (!IsFileExist(test_mdl_path)) {
      PreProcess(inst, iter->second.c_str());
      ReadWave(wave_path, wave);
      ProcessVoiceInput(inst, &wave[0], wave.size());
      spk_info = PostProcess(inst);
      test_mdl_info = string(spk_info->data, spk_info->length);
      WriteSpeakerInfo(test_mdl_path, test_mdl_info);
    } else {
      spk_info = new SpeakerInfo;
      LoadSpeakerInfo(test_mdl_path, spk_info);
    }

    spk_info_orig = new SpeakerInfo;
    LoadSpeakerInfo(mdl_path, spk_info_orig);
    CompareVoicePrint(inst, spk_info_orig, spk_info, &score);
    stream << spk_id << " " << waves << " " << score;
    test_result.push_back(stream.str());
    stream.clear();
    stream.str("");
    DestorySpeakerInfo(spk_info);
    DestorySpeakerInfo(spk_info_orig);
  }
  WriteTestResult("test_result.list", test_result);
  DestoryInstance(inst);
  UnInit(handler);
}

int main(int argc, char *argv[]) {

  const string test_list = "testlist.txt";
  const string reg_list = "reglist.txt";
  const string wave_dir = "./20170602";
  const string model_dir = "./20170602";
  
  SpeakerModel spk_mdl;
  map<string, vector<string> > register_table;
  vector<pair<string, string>> test_table;
  ReadTestList(test_list, test_table);
  ReadRegisterList(reg_list, register_table);
  RegisterModels(model_dir, wave_dir, register_table);
  TestEngine(wave_dir, model_dir, test_table);
  return 0;
}
