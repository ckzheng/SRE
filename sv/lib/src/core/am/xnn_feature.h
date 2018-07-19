// xnn-evaluator/xnn-feature.h

// Copyright 2015 Alibaba-inc  [zhijie.yzj]

#ifndef XNN_FEATURE_H_
#define XNN_FEATURE_H_

#include <iostream>
#include <fstream>
#include "am/xnn_runtime.h"
#include "am/xnn_kaldi_utility.h"
#include "base/log_message.h"

namespace idec {

class XnnFeature
{
public:
    xnnFloatRuntimeMatrix feat_;

public:
    void ReadKaldi(const std::string &fn)
    {
        using namespace xnnKaldiUtility;

        std::ifstream is;
        bool binary = true;

        is.open(fn.c_str(), binary ? std::ios::binary | std::ios::in : std::ios::in);
        if (!is.is_open())
            IDEC_ERROR << "error opening " << fn;

        std::string token;
        ReadToken(is, binary, &token);

        // make sure the input file is binary
        if (is.peek() != '\0')
            IDEC_ERROR << "only support kaldi binary format";
        is.get();
        if (is.peek() != 'B')
            IDEC_ERROR << "only support kaldi binary format";
        is.get();

        ReadToken(is, binary, &token);
        if (token != "FM") {
            IDEC_ERROR << ": Expected token " << "FM" << ", got " << token;
        }

        int32 rows, cols;
        ReadBasicType(is, binary, &rows);  // throws on error.
        ReadBasicType(is, binary, &cols);  // throws on error.

        feat_.Resize(cols, rows);

        for (int32 i = 0; i < rows; i++) {
            is.read(reinterpret_cast<char*>(feat_.Col(i)), sizeof(Real)*cols);
            if (is.fail()) IDEC_ERROR << "read matrix error";
        }

        is.close();
    }

    size_t numFrames() { return feat_.NumCols(); };

    size_t getNFrame(XnnFeature &feat, size_t startT, size_t N)
    {
        if (startT >= numFrames())
        {
            feat.feat_.Clear();
            return(0);
        }
        size_t n = std::min(numFrames() - startT, N);

        feat.feat_.Resize(feat_.NumRows(), n);
        for (size_t t = 0; t < n; ++t)
        {
#ifdef _MSC_VER
            memcpy_s(feat.feat_.Col(t), feat.feat_.NumRows()*sizeof(float), feat_.Col(t + startT), feat_.NumRows()*sizeof(float));
#else
            memcpy(feat.feat_.Col(t), feat_.Col(t + startT), feat_.NumRows()*sizeof(float));
#endif
        }

        return(n);
    }

    void contextExpansion(int contextExpLeft = 0, int contextExpRight = 0)
    {
        if (contextExpLeft != 0 || contextExpRight != 0)
        {
            xnnFloatRuntimeMatrix rawfeat = feat_;
            //rawfeat = feat_;
            feat_.Resize(rawfeat.NumRows()*(contextExpLeft + contextExpRight + 1), rawfeat.NumCols());

            for (int t = 0; t<(int)rawfeat.NumCols(); ++t)
            {
                Real *pData = feat_.Col(t);
                for (int c = -contextExpLeft; c <= contextExpRight; ++c)
                {
                    int fr = std::min((int)rawfeat.NumCols() - 1, std::max(0, t + c));
                    for (size_t d = 0; d<rawfeat.NumRows(); ++d, ++pData)
                    {
                        *pData = rawfeat.Col(fr)[d];
                    }
                }
            }
        }
    }
};

}

#endif
