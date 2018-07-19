#ifndef __BATCH_LIST_H__
#define __BATCH_LIST_H__
#include <cstdlib>
#include <cstdio>
using namespace std;

namespace idec {
class  BatchTask {
 public:
  BatchTask();
  virtual ~BatchTask();
  int			DoBatchTask(const char *list_file);
  virtual		int BeginDoBatchTask() = 0;
  virtual		int EndDoBatchTask() = 0;
  virtual		int DoOneTask(char *onetask) = 0;
 private:
  const char *fgetline(FILE *f, char *buf, int size);
  int			 num_parsed_;
};
}
#endif
