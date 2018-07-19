/*
 * VoicePrintEngineLibrary.cpp
 *
 *  Created on: 2017年6月12日
 *      Author: jxing
 */

#include "com_alibaba_nls_voiceprint_realtime_service_engine_VoicePrintLibrary.h"
#include "speaker_verification.h"
#include "resource_manager.h"
#include "speaker_verification_impl.h"

JNIEXPORT jlong JNICALL
Java_com_alibaba_nls_voiceprint_realtime_service_engine_VoicePrintLibrary_init
(JNIEnv *env, jobject thisObject, jstring cfgFile, jstring sysDir) {
  jboolean iscopy;
  ResourceManager *res = NULL;
  const char *configFile = env->GetStringUTFChars(cfgFile, &iscopy);
  if ( NULL == configFile ) {
    idec::IDEC_INFO << "init, invalid config file";
    env->ReleaseStringUTFChars(cfgFile, configFile);
    return 0;
  }

  const char *systemDir = env->GetStringUTFChars(sysDir, &iscopy);

  try {
    res = ResourceManager::Instance(configFile, systemDir==NULL?"":systemDir);
  } catch (...) {
    idec::IDEC_INFO << "Init() fail in speaker verification!";
  }

  env->ReleaseStringUTFChars(cfgFile, configFile);
  env->ReleaseStringUTFChars(sysDir, systemDir);
  return (jlong)res;
}

JNIEXPORT jlong JNICALL
Java_com_alibaba_nls_voiceprint_realtime_service_engine_VoicePrintLibrary_createInstance
(JNIEnv *env, jobject thisObject, jlong handler) {
  SpeakerVerificationImpl *imp = NULL;

  if ( 0 == handler ) {
    idec::IDEC_INFO << "createInstance, invalid handler";
    return 0;
  }

  try {
    ResourceManager *res = (ResourceManager *) handler;
    imp = new SpeakerVerificationImpl(*res);
  } catch (...) {
    idec::IDEC_INFO << "CreateInstance() fail in speaker verification!";
  }

  return (jlong)imp;
}

JNIEXPORT jint JNICALL
Java_com_alibaba_nls_voiceprint_realtime_service_engine_VoicePrintLibrary_preProcess
(JNIEnv *env, jobject thisObject, jlong instance) {
  int res = 0;

  if ( 0 == instance ) {
    idec::IDEC_INFO << "preProcess, invalid instance";
    return 0;
  }

  try {
    SpeakerVerificationImpl *imp = (SpeakerVerificationImpl *) instance;
    imp->BeginRegister(NULL);
    res = 1;
  } catch (...) {
    idec::IDEC_INFO << "PreProcess() fail in speaker verification!";
  }

  return res;
}

JNIEXPORT jint JNICALL
Java_com_alibaba_nls_voiceprint_realtime_service_engine_VoicePrintLibrary_processVoiceInput
(JNIEnv *env, jobject thisObject, jlong instance, jbyteArray audioFrame,
 jint length, jstring id) {
  int res = 0;
  //  jboolean iscopy;
  const char *session_id = env->GetStringUTFChars(id, NULL);

  if ( 0 == instance ) {
    idec::IDEC_INFO << session_id << ", processVoiceInput, invalid instance";
    env->ReleaseStringUTFChars(id, session_id);
    return 0;
  }

  jbyte *buffer = (jbyte *)env->GetPrimitiveArrayCritical(audioFrame, NULL);
  char *input = (char *)buffer;

  try {
    SpeakerVerificationImpl *imp = (SpeakerVerificationImpl *) instance;
    imp->Register(input, length);
    res = 1;
  } catch(std::exception &ex) {
    idec::IDEC_INFO << session_id <<
                    ", ProcessVoiceInput() fail in speaker verification, Error = " << ex.what();
  } catch (...) {
    idec::IDEC_INFO << session_id <<
                    ", ProcessVoiceInput() fail in speaker verification!";
  }

  env->ReleasePrimitiveArrayCritical(audioFrame, buffer, JNI_ABORT);
  env->ReleaseStringUTFChars(id, session_id);

  return res;
}

JNIEXPORT jbyteArray JNICALL
Java_com_alibaba_nls_voiceprint_realtime_service_engine_VoicePrintLibrary_postProcess
(JNIEnv *env, jobject thisObject, jlong instance, jstring id) {
  // jboolean iscopy;
  const char *session_id = env->GetStringUTFChars(id, NULL);
  if ( 0 == instance ) {
    idec::IDEC_INFO << session_id << ", postProcess, invalid instance ";
    env->ReleaseStringUTFChars(id, session_id);
    return env->NewByteArray(0);
  }

  jbyteArray resultArray;

  try {
    SpeakerModel spk_mdl;
    std::string s_mdl;

    SpeakerVerificationImpl *imp = (SpeakerVerificationImpl *) instance;
    imp->EndRegister(spk_mdl);
    spk_mdl.Searialize(s_mdl);
    resultArray = env->NewByteArray(s_mdl.size());
    env->SetByteArrayRegion(resultArray, 0, s_mdl.size(), (jbyte *) s_mdl.c_str());
    env->ReleaseStringUTFChars(id, session_id);
  } catch(std::exception &ex) {
    idec::IDEC_INFO << session_id <<
                    ", PostProcess() fail in speaker verification, Error = " << ex.what();
    resultArray = env->NewByteArray(0);
  } catch (...) {
    idec::IDEC_INFO << session_id <<
                    ", PostProcess() fail in speaker verification!";
    resultArray = env->NewByteArray(0);
  }

  return resultArray;
}

JNIEXPORT jfloat JNICALL
Java_com_alibaba_nls_voiceprint_realtime_service_engine_VoicePrintLibrary_compareVoicePrint
(JNIEnv *env, jobject instance, jlong handler, jbyteArray array1,
 jint lenArray1, jbyteArray array2, jint lenArray2, jstring id) {
  float score = -999;
  //  jboolean iscopy;
  const char *session_id = env->GetStringUTFChars(id, NULL);
  if ( 0 == handler ) {
    idec::IDEC_INFO << session_id << ", compareVoicePrint, invalid handler ";
    env->ReleaseStringUTFChars(id, session_id);
    return score;
  }

  jbyte *feature1 = (jbyte *)env->GetPrimitiveArrayCritical(array1, NULL);
  jbyte *feature2 = (jbyte *)env->GetPrimitiveArrayCritical(array2, NULL);

  if ( NULL == feature1 || NULL == feature2 || lenArray1 != lenArray2 ) {
    idec::IDEC_INFO << session_id << ", Speaker info is NULL or length not equal";
    env->ReleasePrimitiveArrayCritical(array1, feature1, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(array1, feature2, JNI_ABORT);
    env->ReleaseStringUTFChars(id, session_id);
    return score;
  }

  try {
    ResourceManager *res = (ResourceManager *) handler;
    SpeakerVerificationImpl imp(*res);

    SpeakerModel old_spk_mdl, new_spk_mdl;
    string s_mdl1((char *)feature1, lenArray1);
    string s_mdl2((char *)feature2, lenArray2);

    std::string spk_id;
    int spk_id_len = 50;
    if (!old_spk_mdl.IsValid(s_mdl1)) {
      if (s_mdl1.size() < spk_id_len) {
        spk_id_len = s_mdl1.size();
      }
      spk_id = s_mdl1.substr(0, spk_id_len);
      idec::IDEC_ERROR << session_id <<
                       ", Speaker info is invalid. speaker model is " << spk_id;
    }

    if (!new_spk_mdl.IsValid(s_mdl2)) {
      if (s_mdl2.size() < spk_id_len) {
        spk_id_len = s_mdl1.size();
      }
      spk_id = s_mdl2.substr(0, spk_id_len);
      idec::IDEC_ERROR << session_id <<
                       ", Speaker info is invalid. speaker model is " << spk_id;
    }

    old_spk_mdl.Deserialize(s_mdl1);
    new_spk_mdl.Deserialize(s_mdl2);

    imp.ComputeScore(old_spk_mdl, new_spk_mdl, score);

  } catch(std::exception &ex) {
    idec::IDEC_INFO << session_id <<
                    ", CompareVoicePrint() fail in speaker verification, Error = " << ex.what();
  } catch (...) {
    idec::IDEC_INFO << session_id <<
                    ", CompareVoicePrint() fail in speaker verification!";
  }

  env->ReleasePrimitiveArrayCritical(array1, feature1, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(array1, feature2, JNI_ABORT);
  env->ReleaseStringUTFChars(id, session_id);
  return score;
}

JNIEXPORT jbyteArray JNICALL
Java_com_alibaba_nls_voiceprint_realtime_service_engine_VoicePrintLibrary_updateVoicePrint
(JNIEnv *env, jobject instance, jlong handler, jbyteArray array1,
 jint lenArray1, jbyteArray array2, jint lenArray2, jstring id) {
  //  jboolean iscopy;
  const char *session_id = env->GetStringUTFChars(id, NULL);
  if ( 0 == handler ) {
    idec::IDEC_INFO << session_id << ", updateVoicePrint, invalid handler ";
    env->ReleaseStringUTFChars(id, session_id);
    return 0;
  }

  jbyte *feature1 = (jbyte *)env->GetPrimitiveArrayCritical(array1, NULL);
  jbyte *feature2 = (jbyte *)env->GetPrimitiveArrayCritical(array2, NULL);

  jbyteArray resultArray;

  if ( NULL == feature1 || NULL == feature2 || lenArray1 != lenArray2 ) {
    idec::IDEC_INFO << session_id << ", Speaker info is NULL or length not equal";
    env->ReleasePrimitiveArrayCritical(array1, feature1, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(array1, feature2, JNI_ABORT);
    env->ReleaseStringUTFChars(id, session_id);
    return env->NewByteArray(0);
  }

  try {
    ResourceManager *res = (ResourceManager *) handler;
    SpeakerVerificationImpl imp(*res);
    string s_mdl1((char *)feature1, lenArray1);
    string s_mdl2((char *)feature2, lenArray2);
    SpeakerModel old_spk_mdl, new_spk_mdl, spk_mdl;

    if (!old_spk_mdl.IsValid(s_mdl1)) {
      idec::IDEC_ERROR << session_id << ", Speaker info is invalid.";
    }

    if (!new_spk_mdl.IsValid(s_mdl2)) {
      idec::IDEC_ERROR << session_id << ", Speaker info is invalid.";
    }

    old_spk_mdl.Deserialize(s_mdl1);
    new_spk_mdl.Deserialize(s_mdl2);
    spk_mdl = imp.UpdateModel(old_spk_mdl, new_spk_mdl);
    std::string s_mdl;
    spk_mdl.Searialize(s_mdl);

    resultArray = env->NewByteArray(s_mdl.size());
    env->SetByteArrayRegion(resultArray, 0, s_mdl.size(),(jbyte *) s_mdl.c_str());
  } catch(std::exception &ex) {
    idec::IDEC_INFO << session_id <<
                    ", UpdateVoicePrint() fail in speaker verification, Error = " << ex.what();
    resultArray = env->NewByteArray(0);
  } catch (...) {
    idec::IDEC_INFO << session_id <<
                    ", UpdateVoicePrint() fail in speaker verification!";
    resultArray = env->NewByteArray(0);
  }

  env->ReleasePrimitiveArrayCritical(array1, feature1, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(array1, feature2, JNI_ABORT);
  env->ReleaseStringUTFChars(id, session_id);
  return resultArray;
}

JNIEXPORT jint JNICALL
Java_com_alibaba_nls_voiceprint_realtime_service_engine_VoicePrintLibrary_destroyInstance
(JNIEnv *env, jobject thisObject, jlong instance) {
  if ( 0 == instance ) {
    idec::IDEC_INFO << "destroyInstance, invalid instance ";
    return 0;
  }

  SpeakerVerificationImpl *imp = (SpeakerVerificationImpl *)instance;
  delete imp;

  return 1;
}

JNIEXPORT jint JNICALL
Java_com_alibaba_nls_voiceprint_realtime_service_engine_VoicePrintLibrary_unInit
(JNIEnv *env, jobject thisObject, jlong handler) {
  if ( 0 == handler ) {
    idec::IDEC_INFO << "unInit, invalid handler";
    return 0;
  }

  try {
    ResourceManager *imp = (ResourceManager *)handler;
    imp->Destroy();
    //delete imp;
    return 1;
  } catch (...) {
    idec::IDEC_INFO << "UnInit() fail in speaker verification!";
  }
}

JNIEXPORT jint JNICALL Java_com_alibaba_nls_voiceprint_realtime_service_engine_VoicePrintLibrary_getAgeAndSex
  (JNIEnv *env, jobject thisObject, jbyteArray thisArray, jint thisInt, jstring thisString) {
    idec::IDEC_INFO << "[WARN] Call unimplement interface Java_com_alibaba_nls_voiceprint_realtime_service_engine_VoicePrintLibrary_getAgeAndSex ";
    return 0; 
}

JNIEXPORT jlong JNICALL Java_com_alibaba_nls_voiceprint_realtime_service_engine_VoicePrintLibrary_getValidSpeechLength
  (JNIEnv * env, jobject thisObject, jlong instance) {
  if ( 0 == instance ) {
    idec::IDEC_INFO << "destroyInstance, invalid instance ";
    return 0;
  }

  try {
    SpeakerVerificationImpl *imp = (SpeakerVerificationImpl *)instance;
    return imp->GetValidSpeechLength();
  } catch (...) {
    idec::IDEC_INFO << "GetValidSpeechLength() fail in speaker verification!";
    return 0;
  }
}

