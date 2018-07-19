
#include <map>
#include <string>
#include "am/kaldi_am.h"

namespace idec{
    AcousticModel* AcousticModelMaker::MakeFromBinaryFile(const std::string &file_name) {
        AcousticModel * acoustic_model = NULL;
        if(KaldiAM::IsValidKaldiAm(file_name.c_str())){
            KaldiAM * kaldi_am = new KaldiAM();
            acoustic_model = kaldi_am;
        }

        try {
            if (acoustic_model != NULL) {
                acoustic_model->ReadBinary(file_name.c_str());
            }
        }
        catch (const std::exception &e) {
            std::cerr << e.what();
            IDEC_DELETE(acoustic_model);
        }

        return acoustic_model;
    }
};
