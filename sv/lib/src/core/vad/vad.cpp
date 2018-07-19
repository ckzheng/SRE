#include "als_vad.h"
#include "vad/vad_impl.h"

std::auto_ptr<AlsVadImpl>AlsVadImpl::instance_;
idec::thread::recursive_mutex AlsVadImpl::mutex_;

AlsVad *AlsVad::Create(const char *cfg) {
  return AlsVadImpl::Instance()->Create(cfg);
}

AlsVad *AlsVad::Create(const char *cfg, const char *model_path) {
  return AlsVadImpl::Instance()->Create(cfg, model_path);
}

void AlsVad::Destroy(AlsVad *vad) {
  AlsVadImpl::Instance()->Delete(vad);
}

AlsVadMdlHandle AlsVad::LoadModel(const char *cfg, const char *model_path) {
  return AlsVadImpl::Instance()->LoadModel(cfg, model_path);
}

AlsVadMdlHandle AlsVad::LoadModel(const char *cfg) {
  return AlsVadImpl::Instance()->LoadModel(cfg);
}

void AlsVad::UnLoadModel(AlsVadMdlHandle h) {
  AlsVadImpl::Instance()->UnLoadModel(h);
}


AlsVad  *AlsVad::CreateFromModel(AlsVadMdlHandle h) {
  return AlsVadImpl::Instance()->CreateFromModel(h);
}


void AlsVadResult_Release(AlsVadResult **self) {
  return AlsVadImpl::Instance()->ReleaseResult(*self);
}
