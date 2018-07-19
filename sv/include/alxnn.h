#ifndef _ALI_XNN_INCLUDED
#define _ALI_XNN_INCLUDED

#ifndef WIN32
#include <unistd.h>
#else
#include <crtdefs.h>
#endif // WIN32

#include <string>
#include <map>
#include <vector>
#include "als_error.h"

class XnnEvaluate
{
public:
    ALSAPI_EXPORT static ALS_RETCODE Create(XnnEvaluate** ppObj, int reserved = 0); 
    virtual ALS_RETCODE Release() = 0;
    virtual ~XnnEvaluate() {};

    virtual ALS_RETCODE LoadModel(const char* model_path) = 0;
    virtual ALS_RETCODE SaveModel(const char* model_path) = 0;
    virtual size_t GetInputDim() = 0;
    virtual size_t GetOutputDim() = 0;
    virtual ALS_RETCODE Evaluate(
        const float* feature_data, size_t vdim, float* result_like, size_t udim, bool use_real_prob = false) = 0;
};

namespace ws
{
  class AliTokenizerFactory;
  class AliTokenizer;
  class SegResult;
}

class Phrase2ID
{
public:
    Phrase2ID() : tokenizer_(NULL), seg_result_(NULL) {}
    bool Initialize(const char *voc_file, const char *tokenizer_conf_file);
    bool StringToId(const std::string &phrase, std::vector<int> &ids);
    void IdToFeature(const std::vector<int> &ids, int dim, float *feature);
    bool StringToFeature(const std::string &phrase, int dim, float *feature);
    void Close();
private:
    std::map<std::string, int> word_map_;
    ws::AliTokenizerFactory* factory_;
    ws::AliTokenizer* tokenizer_;
    ws::SegResult* seg_result_;
};

#endif // _ALI_XNN_INCLUDED
