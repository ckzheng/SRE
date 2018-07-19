#ifndef _SPEAKER_INFO_H_
#define _SPEAKER_INFO_H_
extern "C" {
  struct SpeakerInfo {
    char *data;
    int length;
  };

  void *Init(const char *cfg_file, const char *sys_dir);
  void *CreateInstance(void *handler);
  int CompareVoicePrint(void *instance, const SpeakerInfo *old_info,
                        const SpeakerInfo *new_info, float *score);
  SpeakerInfo *UpdateVoicePrint(void *instance, const SpeakerInfo *old_info,
                                const SpeakerInfo *new_info);
  int PreProcess(void *instance, const char *spk_guid);
  int ProcessVoiceInput(void *instance, char *wave, unsigned int len);
  SpeakerInfo *PostProcess(void *instance);
  int DestoryInstance(void *instance);
  int DestorySpeakerInfo(SpeakerInfo *spk_info);
  int UnInit(void *handler);
  long GetValidSpeechLength(void *instance);
  float GetSNR(void *instance);
}

#endif
