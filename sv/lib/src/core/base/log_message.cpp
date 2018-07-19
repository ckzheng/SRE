#include <stdexcept>
#include <string>
#include "base/log_message.h"
#ifdef __BIONIC__
#include "logger.h"
#ifdef PRINT_LOG
#define JNI_LOG
#endif  // __BIONIC__
#endif  // PRINT_LOG
namespace idec {

LogMessage::LogMessage(const char *msg_type, const char *file_name,
                       const char *function_name, int line_num) {
  msg_type_ = msg_type;
  if ((msg_type_.compare("Error") == 0) ||
      (msg_type_.compare("Warning") == 0)) {
    stream_ << file_name << " " << function_name << " " << line_num << " ";
  }
}

LogMessage::~LogMessage() {
  if (msg_type_.compare("Error") == 0) {
#ifdef JNI_LOG
    LOGE("Error: %s.", stream_.str().c_str());
#endif  // JNI_LOG
#ifndef __BIONIC__
    std::cerr << "Error: " << stream_.str() << std::endl;
    throw std::runtime_error(stream_.str());
#endif  // __BIONIC__
  } else if (msg_type_.compare("Warning") == 0) {
#ifdef JNI_LOG
    LOGI("Warning: %s.", stream_.str().c_str());
#endif  // JNI_LOG
#ifndef __BIONIC__
    std::cerr << "Warning: " << stream_.str() << std::endl;
#endif  // __BIONIC__
  } else if (msg_type_.compare("Information") == 0) {
#ifdef JNI_LOG
    LOGI("%s.", stream_.str().c_str());
#endif  // JNI_LOG
#ifndef __BIONIC__
    std::cerr << stream_.str() << std::endl;
#endif  // __BIONIC__
  } else if (msg_type_.compare("Verbose") == 0) {
#ifdef JNI_LOG
    LOGI("[verb] %s.", stream_.str().c_str());
#endif  // JNI_LOG
#ifndef __BIONIC__
#ifdef IDEC_VERBOSE_ENABLED
    std::cerr << "[verb] " << stream_.str() << std::endl;
#endif  // IDEC_VERBOSE_ENABLED
#endif  // __BIONIC__
  } else {
#ifdef JNI_LOG
    LOGI("%s.", stream_.str().c_str());
#endif  // JNI_LOG
#ifndef __BIONIC__
    std::cerr << stream_.str() << std::endl;
#endif  // __BIONIC__
  }
}
};  // namespace idec

