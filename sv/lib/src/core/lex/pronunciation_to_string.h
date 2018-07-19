#ifndef ASR_DECODER_SRC_CORE_LEX_PRONUNCIATION_TO_STRING_H_
#define ASR_DECODER_SRC_CORE_LEX_PRONUNCIATION_TO_STRING_H_
#include <string>
#include "base/idec_common.h"
#include "base/idec_types.h"
#include "util/text-utils.h"
namespace idec {
class PronunciationToString {
 public:
  explicit PronunciationToString(PhoneSet *phone_set) : phone_set_(phone_set),
  max_phone_len_(0), min_phone_len_(UINT_MAX) {}
  virtual ~PronunciationToString() {
    PronMap::iterator it;
    for (it = pron_map_.begin(); it != pron_map_.end(); it++) {
      delete [](it->first);
    }
  }

  IDEC_RETCODE LoadTxt(std::string file_name) {
    char line_buf[1024];
    FILE *fp = fopen(file_name.c_str(), "rt");
    if (fp == NULL) {
      return IDEC_OPEN_ERROR;
    }
    while (fgets(line_buf, sizeof(line_buf), fp) != NULL) {
      std::vector<std::string> tokens;
      std::string line_trimed = line_buf;
      Trim(&line_trimed);
      SplitStringToVector(line_trimed, "\n\t ", true, &tokens);
      bool valid = true;
      for (size_t p = 1; p < tokens.size(); p++) {
        int phone_idx = phone_set_->PhoneName2PhoneId(tokens[p].c_str());
        if (phone_idx == -1) {
          valid = false;
          break;
        }
      }
      if (tokens.size() >= 1 && valid) {
        // assemble the phone string
        size_t num_phones = tokens.size() - 1;
        min_phone_len_ = std::min(num_phones, min_phone_len_);
        max_phone_len_ = std::max(num_phones, max_phone_len_);
        PhoneId *phones = new PhoneId[num_phones + 1];
        for (size_t p = 1; p < tokens.size(); p++) {
          int phone_idx = phone_set_->PhoneName2PhoneId(tokens[p].c_str());
          phones[p - 1] = (PhoneId)phone_idx;
        }
        phones[num_phones] = kPhnIdStrTemrSym;
        PronMap::iterator it = pron_map_.find(phones);
        // insert or replace key
        if (it == pron_map_.end()) {
            // insert, key memory ownership goes to pron_map_
            pron_map_[phones] = tokens[0];
        }
        else {
            delete[] phones;
            it->second = tokens[0];
        }
      }
    }
    fclose(fp);
    return IDEC_SUCCESS;
  }
  struct Hyp {
    size_t start_pos;
    std::string word;
  };
  std::string PronToString(PhoneId *phone_str, size_t phone_str_len) {
    // segmentation hyps
    PhoneId phn_sub_str[1024];
    std::vector<std::vector<Hyp> > seg_hyps;
    seg_hyps.resize(phone_str_len);
    // 1) do all the segmentations,
    // try to matching [start_pos end_pos]
    for (size_t start_pos = 0; start_pos < phone_str_len; start_pos++) {
      if (start_pos == 0 || seg_hyps[start_pos - 1].size() != 0) {
        for (size_t len = min_phone_len_;
             len <= max_phone_len_ && start_pos + len <= (size_t)phone_str_len;
             len++) {
          size_t end_pos = start_pos + len - 1;
          memcpy(phn_sub_str, phone_str + start_pos, len);
          phn_sub_str[len + 1] = kPhnIdStrTemrSym;
          if (pron_map_.find(phn_sub_str) != pron_map_.end()) {
            Hyp hyp;
            hyp.start_pos = start_pos;
            hyp.word = pron_map_[phn_sub_str];
            seg_hyps[end_pos].push_back(hyp);
          }
        }
      }
    }
    // 2) backtracking process
    std::vector<std::string> segmented_result;
    int cur_end_pos = static_cast<int>(phone_str_len) - 1;
    while (cur_end_pos > 0 && seg_hyps[cur_end_pos].size() != 0) {
      // random choose one
      segmented_result.push_back(seg_hyps[cur_end_pos].back().word);
      cur_end_pos =
        static_cast<int>(seg_hyps[cur_end_pos].back().start_pos) - 1;
    }
    if (cur_end_pos > 0) {
      return "";
    } else {
      std::string return_val;
      for (size_t i = 0; i < segmented_result.size(); i++) {
        return_val += segmented_result[segmented_result.size() - i - 1];
        if (i != segmented_result.size() - 1) {
          return_val += " ";
        }
      }
      return return_val;
    }
  }

 private:
  typedef unordered_map<PhoneId *, std::string, PhoneStringHashFunctions,
          PhoneStringHashFunctions> PronMap;
  PhoneSet *phone_set_;
  PronMap pron_map_;
  size_t max_phone_len_;
  size_t min_phone_len_;
};
}  // namespace idec
#endif  // ASR_DECODER_SRC_CORE_LEX_PRONUNCIATION_TO_STRING_H_

