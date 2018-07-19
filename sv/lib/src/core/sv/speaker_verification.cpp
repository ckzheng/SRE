#include "speaker_verification.h"
#include "resource_manager.h"
#include "speaker_verification_impl.h"

void *Init(const char *cfg_file, const char *sys_dir) {
  ResourceManager *res = NULL;
  try {
    if (cfg_file == NULL) {
      return NULL;
    }

    if (sys_dir == NULL) {
      res = ResourceManager::Instance(cfg_file, "");
    } else {
      res = ResourceManager::Instance(cfg_file, sys_dir);
    }
  } catch (...) {
    idec::IDEC_INFO << "CreateInstance() fail in speaker verification!";
    return NULL;
  }
  return res;
}

void *CreateInstance(void *handler) {
  SpeakerVerificationImpl *imp = NULL;
  try {
    if (handler != NULL) {
      ResourceManager *res = static_cast<ResourceManager *>(handler);
      imp = new SpeakerVerificationImpl(*res);
    }
  } catch (...) {
    idec::IDEC_INFO << "CreateInstance() fail in speaker verification!";
    return NULL;
  }
  return imp;
}

int CompareVoicePrint(void *inst, const SpeakerInfo *old_info,
                      const SpeakerInfo *new_info, float *score) {
  try {
    if (inst != NULL) {
      if ((old_info == NULL) || (new_info == NULL)) {
        idec::IDEC_ERROR << "Speaker info is NULL.";
      }

      idec::IDEC_ASSERT(old_info->data != NULL);
      idec::IDEC_ASSERT(new_info->data != NULL);

      SpeakerModel old_spk_mdl, new_spk_mdl;
      string s_mdl1(old_info->data, old_info->length);
      string s_mdl2(new_info->data, new_info->length);

      string spk_id;
      int spk_id_len = 50;
      if (!old_spk_mdl.IsValid(s_mdl1)) {
        if (s_mdl1.size() < spk_id_len) {
          spk_id_len = s_mdl1.size();
        }
        spk_id = s_mdl1.substr(0, spk_id_len);
        idec::IDEC_ERROR << "Speaker info is invalid. speaker model is " << spk_id;
      }

      if (!new_spk_mdl.IsValid(s_mdl2)) {
        if (s_mdl2.size() < spk_id_len) {
          spk_id_len = s_mdl1.size();
        }
        spk_id = s_mdl2.substr(0, spk_id_len);
        idec::IDEC_ERROR << "Speaker info is invalid. speaker model is " << spk_id;
      }

      old_spk_mdl.Deserialize(s_mdl1);
      new_spk_mdl.Deserialize(s_mdl2);

      SpeakerVerificationImpl *imp = static_cast<SpeakerVerificationImpl *>(inst);
      imp->ComputeScore(old_spk_mdl, new_spk_mdl, *score);
    }
  } catch (...) {
    idec::IDEC_INFO << "CompareVoicePrint() fail in speaker verification!";
    return -1;
  }
  return 0;
}

SpeakerInfo *UpdateVoicePrint(void *inst, const SpeakerInfo *old_info,
                              const SpeakerInfo *new_info) {
  SpeakerInfo *spk_info = NULL;
  try {
    if (inst != NULL) {
      idec::IDEC_ASSERT(old_info->data != NULL);
      idec::IDEC_ASSERT(new_info->data != NULL);

      string s_mdl1(old_info->data, old_info->length);
      string s_mdl2(new_info->data, new_info->length);

      SpeakerModel old_spk_mdl, new_spk_mdl, spk_mdl;
      if (!old_spk_mdl.IsValid(s_mdl1)) {
        idec::IDEC_ERROR << "Speaker info is invalid.";
      }

      if (!new_spk_mdl.IsValid(s_mdl2)) {
        idec::IDEC_ERROR << "Speaker info is invalid.";
      }

      old_spk_mdl.Deserialize(s_mdl1);
      new_spk_mdl.Deserialize(s_mdl2);

      SpeakerVerificationImpl *imp = static_cast<SpeakerVerificationImpl *>(inst);
      spk_mdl = imp->UpdateModel(old_spk_mdl, new_spk_mdl);

      string s_mdl;
      spk_mdl.Searialize(s_mdl);
      spk_info = new SpeakerInfo();
      spk_info->length = s_mdl.size();
      spk_info->data = new char[s_mdl.size()];
      memcpy(spk_info->data, s_mdl.c_str(), s_mdl.size());
    }
  } catch (...) {
    idec::IDEC_INFO << "UpdateVoicePrint() fail in speaker verification!";
    return NULL;
  }
  return spk_info;
}

int PreProcess(void *inst, const char *spk_guid) {
  try {
    if (inst != NULL) {
      SpeakerVerificationImpl *imp = static_cast<SpeakerVerificationImpl *>(inst);
      imp->BeginRegister(spk_guid);
    }
  } catch (...) {
    idec::IDEC_INFO << "PreProcess() fail in speaker verification!";
    return -1;
  }
  return 0;
}

int ProcessVoiceInput(void *inst, char *wave, unsigned int len) {
  try {
    if (inst != NULL) {
      SpeakerVerificationImpl *imp = static_cast<SpeakerVerificationImpl *>(inst);
      if ((wave != NULL) && (len != 0)) {
        imp->Register(wave, len);
      }
    }
  } catch (...) {
    idec::IDEC_INFO << "ProcessVoiceInput() fail in speaker verification!";
    return -1;
  }
  return 0;
}

SpeakerInfo *PostProcess(void *inst) {
  SpeakerInfo *spk_info = NULL;
  try {
    if (inst != NULL) {
      SpeakerVerificationImpl *imp = static_cast<SpeakerVerificationImpl *>(inst);
      SpeakerModel spk_mdl;
      imp->EndRegister(spk_mdl);
      string s_mdl;
      spk_mdl.Searialize(s_mdl);
      spk_info = new SpeakerInfo;
      spk_info->length = s_mdl.size();
      spk_info->data = new char[s_mdl.size()];
      memcpy(spk_info->data, s_mdl.c_str(), s_mdl.size());
    }
  } catch (...) {
    idec::IDEC_INFO << "PostProcess() fail in speaker verification!";
    return NULL;
  }
  return spk_info;
}

int DestoryInstance(void *inst) {
  try {
    if (inst != NULL) {
      SpeakerVerificationImpl *imp = static_cast<SpeakerVerificationImpl *>(inst);
      delete imp;
    }
    inst = NULL;
  } catch (...) {
    idec::IDEC_INFO << "DestoryInstance() fail in speaker verification!";
    return -1;
  }
  return 0;
}

int DestorySpeakerInfo(SpeakerInfo *mdl) {
  try {
    if (mdl != NULL) {
      if (mdl->data != NULL) {
        delete[] mdl->data;
      }
      mdl->data = NULL;
      delete mdl;
    }
    mdl = NULL;
  } catch (...) {
    idec::IDEC_INFO << "DestorySpeakerInfo() fail in speaker verification!";
    return -1;
  }
  return 0;
}

int UnInit(void *handler) {
  try {
    if (handler != NULL) {
      ResourceManager *imp = static_cast<ResourceManager *>(handler);
      imp->Destroy();
    }
    handler = NULL;
  } catch (...) {
    idec::IDEC_INFO << "UnInit() fail in speaker verification!";
    return -1;
  }
  return 0;
}

long GetValidSpeechLength(void *inst) {
  try {
    if (inst != NULL) {
      SpeakerVerificationImpl *imp = static_cast<SpeakerVerificationImpl *>(inst);
      return imp->GetValidSpeechLength();
    }
  } catch (...) {
    idec::IDEC_INFO << "GetValidSpeechLength() fail in speaker verification!";
  }
  return 0;
}

float GetSNR(void *inst) {
  try {
    if (inst != NULL) {
      SpeakerVerificationImpl *imp = static_cast<SpeakerVerificationImpl *>(inst);
      return imp->GetSNR();
    }
  } catch (...) {
    idec::IDEC_INFO << "GetSNR() fail in speaker verification!";
  }
  return 0;
}