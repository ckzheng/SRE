#ifndef __LTS_H__
#define __LTS_H__
#include <string>
#include <vector>
#include "base/idec_types.h"
#include "util/options-itf.h"
#include "base/log_message.h"

namespace idec {
    typedef enum {
        kLtsSuccess = 1,
        kLtsCanNotHandleWord = 2,
        kLtsNoneOutputBuffer= 3
    }LtsRetCode;

    struct LtsOption {
        int max_pron_per_word;
        bool add_sil_each_word;
        bool add_sil_whole_word;
        bool debug_dump_lts;

        LtsOption() {
            max_pron_per_word = 32;
            add_sil_each_word = false;
            add_sil_whole_word = false;
            debug_dump_lts = false;
        }

        void Register(OptionsItf *po, std::string prefix = "Lts.") {
            po->Register(prefix + "maxPronPerWord", &max_pron_per_word, "");
            po->Register(prefix + "addSilEachWord", &add_sil_each_word, ".");
            po->Register(prefix + "addSilWholeWord", &add_sil_whole_word, "");
            po->Register(prefix + "debugDumpLts", &debug_dump_lts, "");

            if (add_sil_whole_word && add_sil_each_word) {
                IDEC_ERROR << "add_sil_whole_word and add_sil_each_word cannot set to true at same time";
            }
        }
    };
    class Lts {
    public:
        LtsRetCode virtual Init() = 0;
        LtsRetCode virtual Unit() = 0;
        LtsRetCode virtual TextToPhone(const std::string & input_text, std::vector<std::string>* out_phones) = 0;
        LtsRetCode virtual TextToPhone(const std::string & input_text, std::vector<PhoneId>* out_phones) = 0;
        LtsRetCode virtual TextToPhone(const std::string & input_text, std::vector<std::vector<PhoneId> >* out_phones, const LtsOption & opt) = 0;

        virtual ~Lts() {};
    };
};

#endif
