#ifndef ASR_DECODER_SRC_CORE_BASE_LOG_MESSAGE_H_
#define ASR_DECODER_SRC_CORE_BASE_LOG_MESSAGE_H_
#include <iostream>
#include <sstream>
#include "base/idec_common.h"
namespace idec {
#ifdef _MSC_VER
#define __func__ __FUNCTION__
#endif  // _MSC_VER
#define IDEC_MESSAGE(type) \
LogMessage(type, __func__, __FILE__, __LINE__).GetStream()
#define IDEC_ERROR IDEC_MESSAGE("Error")  // error
#define IDEC_WARNING IDEC_MESSAGE("Warning")  // warning
#define IDEC_WARN IDEC_WARNING
#define IDEC_INFO IDEC_MESSAGE("Information")  // high-level information
#define IDEC_VERB IDEC_MESSAGE("Verbose")  // detailed information
class LogMessage {
 public:
  LogMessage(const char *msg_type, const char *file_name,
             const char *function_name, int line_number);
  ~LogMessage();
  std::ostringstream &GetStream() {
    return stream_;
  }

 private:
  std::ostringstream stream_;
  std::string msg_type_;
};
};  // namespace idec
#endif  // ASR_DECODER_SRC_CORE_BASE_LOG_MESSAGE_H_

