#include "spkdiar.h"
#include "spkdiar_result.h"
#include "spkdiar_impl.h"
#include "spkdiar_serialize.h"
#include <iostream>
#include <fstream>

using alspkdiar::SpeakerDiarization;
using alspkdiar::Serialize;
using namespace std;

void *Init(const char *conf_dir) {
  if (conf_dir == NULL) {
    conf_dir = "spkdiar.conf";
    cout << "[WARN] invalid config dir, use default spkdiar.conf!" << endl;
  }

  ResourceManager *handler = NULL;
  try {
    handler =  ResourceManager::Instance(conf_dir);
  } catch (...) {
    cerr << "spkdiar init error." << endl;
  }
  return handler;
}

void *CreateInstance(void *handler) {
  SpeakerDiarization *spkdir = NULL;
  ResourceManager *mdl_handler = NULL;
  try {
    mdl_handler = static_cast<ResourceManager *>(handler);
    spkdir = new SpeakerDiarization(mdl_handler);
  } catch (...) {
    cerr << "spkdiar create instance error." << endl;
  }
  return spkdir;
}

int DestroyResult(AlsSpkdiarResult *out_result) {
  int ret = 0;
  try {
    if (out_result != NULL) {
      AlsSpkdiarResult *spk_diar = NULL;
      spk_diar = static_cast<AlsSpkdiarResult *>(out_result);
      ret = Serialize::DestoryResult(spk_diar);
    }
  } catch (...) {
    cerr << "spkdiar destroy result error." << endl;
  }
  return ret;
}

int DestroyInstance(void *inst) {
  try {
    SpeakerDiarization *spkdiar = NULL;
    spkdiar = static_cast<SpeakerDiarization *> (inst);
    if (spkdiar != NULL) {
      delete spkdiar;
    }
    spkdiar = NULL;
  } catch (...) {
    cerr << "spkdiar destroy instance error." << endl;
  }
  return 0;
}

AlsSpkdiarResult *SpkDiarization(void *inst, char *wave,
                                 unsigned int wave_len) {
  int ret = 0;
  vector<char> wave_data;
  AlsSpkdiarResult *out_result = NULL;
  SpeakerDiarization *spk_diar = NULL;
  try {
    out_result = new AlsSpkdiarResult();
    if (inst != NULL) {
      wave_data.resize(wave_len);
      memcpy(&wave_data[0], wave, wave_len);
      spk_diar = static_cast<SpeakerDiarization *>(inst);
      spk_diar->BeginSpkDiar();
      spk_diar->EndSpkDiar();
      ret = spk_diar->SpkDiar(wave_data, out_result);
      if (ret != 0) {
        std::cerr << "Speaker diarization internal error." << std::endl;
      }
    }
  } catch (...) {
    cerr << "spkdiar running error." << endl;
  }
  return out_result;
}

int UnInit(void *handler) {
  int ret = 0;
  ResourceManager *mdl_handler = NULL;
  try {
    mdl_handler = static_cast <ResourceManager *>(handler);
    if (mdl_handler != NULL) {
      ret = mdl_handler->Destroy();
    }
  } catch(...) {
    cerr << "spkdiar uninit error." << endl;
  }
  return ret;
}
