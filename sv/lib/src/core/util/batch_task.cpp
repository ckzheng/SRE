#include "util/batch_task.h"
#include "base/log_message.h"
#include <cstring>
#include <cstdlib>
namespace idec {
BatchTask::BatchTask() {
  num_parsed_ = 0;
}

BatchTask::~BatchTask() {
}


const char *BatchTask::fgetline(FILE *f, char *buf, int size) {
  char *p = fgets(buf, size, f);
  if (p == NULL) {		// EOF reached: next time feof() = true
    buf[0] = 0;
    return buf;
  }
  size_t n = strlen(p);

  // check for buffer overflow
  if (n >= (size_t)size - 1) {
    IDEC_ERROR << "input line too long (max. %d characters allowed";
  }

  // remove newline at end
  if (n > 0 && p[n - 1] == '\n') {	// UNIX and Windows style
    n--;
    p[n] = 0;
    if (n > 0 && p[n - 1] == '\r') {	// Windows style
      n--;
      p[n] = 0;
    }
  } else if (n > 0 && p[n - 1] == '\r') {	// Mac style
    n--;
    p[n] = 0;
  }

  return buf;
}
int BatchTask::DoBatchTask(const char *list_file) {
  BeginDoBatchTask();
  char line[1024] = "";
#ifdef _MSC_VER
  FILE *fp;
  fopen_s(&fp, list_file, "rt");
#else
  FILE *fp = fopen(list_file, "rt");
#endif
  if (NULL == fp) {
    printf("can not open the tlist file %s", list_file);
    return -1;
  }
  num_parsed_ = 0;
  while (1) {
    fgetline(fp, line, 1024);
    if (line[0] == '\0') {
      break;
    }
    if (DoOneTask(line) >= 0) {
      num_parsed_++;
    }
  }
  EndDoBatchTask();

  if (NULL != fp)
    fclose(fp);
  return 1;
}

}
