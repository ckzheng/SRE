// tree/context-dep.cc

// Copyright 2009-2011  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "kaldi/kaldi-idec-common.h"
#include "kaldi/context-dep.h"
namespace idec {

namespace kaldi {



bool ContextDependency::Compute(const std::vector<int32> &phoneseq,
                                int32 pdf_class,
                                int32 *pdf_id) const {
  KALDI_ASSERT(static_cast<int32>(phoneseq.size()) == N_);
  EventType  event_vec;
  event_vec.reserve(N_+1);
  event_vec.push_back(std::make_pair
                      (static_cast<EventKeyType>(kPdfClass),  // -1
                       static_cast<EventValueType>(pdf_class)));
  KALDI_COMPILE_TIME_ASSERT(kPdfClass < 0);  // or it would not be sorted.
  for (int32 i = 0; i < N_; i++) {
    event_vec.push_back(std::make_pair
                        (static_cast<EventKeyType>(i), static_cast<EventValueType>(phoneseq[i])));
    KALDI_ASSERT(static_cast<EventAnswerType>(phoneseq[i]) != -1);  // >=0 ?
  }
  KALDI_ASSERT(pdf_id != NULL);
  return to_pdf_->Map(event_vec, pdf_id);
}


void ContextDependency::Write (std::ostream &os, bool binary) const {
  WriteToken(os, binary, "ContextDependency");
  WriteBasicType(os, binary, N_);
  WriteBasicType(os, binary, P_);
  WriteToken(os, binary, "ToPdf");
  to_pdf_->Write(os, binary);
  WriteToken(os, binary, "EndContextDependency");
}


void ContextDependency::Read (std::istream &is, bool binary) {
  if (to_pdf_) {
    delete to_pdf_;
    to_pdf_ = NULL;
  }
  ExpectToken(is, binary, "ContextDependency");
  ReadBasicType(is, binary, &N_);
  ReadBasicType(is, binary, &P_);
  EventMap *to_pdf = NULL;
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "ToLength") {  // back-compat.
    EventMap *to_num_pdf_classes = EventMap::Read(is, binary);
    if (to_num_pdf_classes) delete to_num_pdf_classes;
    ReadToken(is, binary, &token);
  }
  if (token == "ToPdf") {
    to_pdf = EventMap::Read(is , binary);
  } else {
    KALDI_ERR << "Got unexpected token " << token
              << " reading context-dependency object.";
  }
  ExpectToken(is, binary, "EndContextDependency");
  to_pdf_ = to_pdf;
}

void ContextDependency::GetPdfInfo(const std::vector<int32> &phones,
                                   const std::vector<int32> &num_pdf_classes,  // indexed by phone,
                                   std::vector<std::vector<std::pair<int32, int32> > > *pdf_info) const {

  EventType vec;
  KALDI_ASSERT(pdf_info != NULL);
  pdf_info->resize(NumPdfs());
  for (size_t i = 0 ; i < phones.size(); i++) {
    int32 phone = phones[i];
    vec.clear();
    vec.push_back(std::make_pair(static_cast<EventKeyType>(P_),
                                 static_cast<EventValueType>(phone)));
    // Now get length.
    KALDI_ASSERT(static_cast<size_t>(phone) < num_pdf_classes.size());
    EventAnswerType len = num_pdf_classes[phone];

    for (int32 pos = 0; pos < len; pos++) {
      vec.resize(2);
      vec[0] = std::make_pair(static_cast<EventKeyType>(P_),
                              static_cast<EventValueType>(phone));
      vec[1] = std::make_pair(kPdfClass, static_cast<EventValueType>(pos));
      std::sort(vec.begin(), vec.end());
      std::vector<EventAnswerType>
      pdfs;  // pdfs that can be at this pos as this phone.
      to_pdf_->MultiMap(vec, &pdfs);
      SortAndUniq(&pdfs);
      if (pdfs.empty()) {
        KALDI_WARN << "ContextDependency::GetPdfInfo, no pdfs returned for position "<<
                   pos << " of phone " << phone << ".   Continuing but this is a serious error.";
      }
      for (size_t j = 0; j < pdfs.size(); j++) {
        KALDI_ASSERT(static_cast<size_t>(pdfs[j]) < pdf_info->size());
        (*pdf_info)[pdfs[j]].push_back(std::make_pair(phone, pos));
      }
    }
  }
  for (size_t i = 0; i < pdf_info->size(); i++) {
    std::sort( ((*pdf_info)[i]).begin(),  ((*pdf_info)[i]).end());
    KALDI_ASSERT(IsSortedAndUniq( ((*pdf_info)[i])));  // should have no dups.
  }
}




ContextDependency *
MonophoneContextDependency(const std::vector<int32> phones,
                           const std::vector<int32> phone2num_pdf_classes) {
  std::vector<std::vector<int32> > phone_sets(phones.size());
  for (size_t i = 0; i < phones.size(); i++) phone_sets[i].push_back(phones[i]);
  std::vector<bool> share_roots(phones.size(), false);  // don't share roots.
  // N is context size, P = position of central phone (must be 0).
  int32 num_leaves = 0, P = 0, N = 1;
  EventMap *pdf_map = ContextDependency::GetStubMap(P, phone_sets,
                      phone2num_pdf_classes, share_roots, &num_leaves);
  return new ContextDependency(N, P, pdf_map);
}

ContextDependency *
MonophoneContextDependencyShared(const std::vector<std::vector<int32> >
                                 phone_sets,
                                 const std::vector<int32> phone2num_pdf_classes) {
  std::vector<bool> share_roots(phone_sets.size(), false);  // don't share roots.
  // N is context size, P = position of central phone (must be 0).
  int32 num_leaves = 0, P = 0, N = 1;
  EventMap *pdf_map = ContextDependency::GetStubMap(P, phone_sets,
                      phone2num_pdf_classes, share_roots, &num_leaves);
  return new ContextDependency(N, P, pdf_map);
}


EventMap *ContextDependency::GetStubMap(int32 P,
                                        const std::vector<std::vector<int32> > &phone_sets,
                                        const std::vector<int32> &phone2num_pdf_classes,
                                        const std::vector<bool> &share_roots,
                                        int32 *num_leaves_out) {

  {
    // Checking inputs.
    KALDI_ASSERT(!phone_sets.empty() && share_roots.size() == phone_sets.size());
    std::set<int32> all_phones;
    for (size_t i = 0; i < phone_sets.size(); i++) {
      KALDI_ASSERT(IsSortedAndUniq(phone_sets[i]));
      KALDI_ASSERT(!phone_sets[i].empty());
      for (size_t j = 0; j < phone_sets[i].size(); j++) {
        KALDI_ASSERT(all_phones.count(phone_sets[i][j]) == 0);  // check not present.
        all_phones.insert(phone_sets[i][j]);
      }
    }
  }

  // Initially create a single leaf for each phone set.

  size_t max_set_size = 0;
  int32 highest_numbered_phone = 0;
  for (size_t i = 0; i < phone_sets.size(); i++) {
    max_set_size = std::max(max_set_size, phone_sets[i].size());
    highest_numbered_phone =
      std::max(highest_numbered_phone,
               *std::max_element(phone_sets[i].begin(), phone_sets[i].end()));
  }

  if (phone_sets.size() ==
      1) {  // there is only one set so the recursion finishes.
    if (share_roots[0]) {  // if "shared roots" return a single leaf.
      return new ConstantEventMap((*num_leaves_out)++);
    } else { // not sharing roots -> work out the length and return a
      // TableEventMap splitting on length.
      EventAnswerType max_len = 0;
      for (size_t i = 0; i < phone_sets[0].size(); i++) {
        EventAnswerType len;
        EventValueType phone = phone_sets[0][i];
        KALDI_ASSERT(static_cast<size_t>(phone) < phone2num_pdf_classes.size());
        len = phone2num_pdf_classes[phone];
        KALDI_ASSERT(len > 0);
        if (i == 0) max_len = len;
        else {
          if (len != max_len) {
            KALDI_WARN << "Mismatching lengths within a phone set: " << len
                       << " vs. " << max_len << " [unusual, but not necessarily fatal]. ";
            max_len = std::max(len, max_len);
          }
        }
      }
      std::map<EventValueType, EventAnswerType> m;
      for (EventAnswerType p = 0; p < max_len; p++)
        m[p] = (*num_leaves_out)++;
      return new TableEventMap(kPdfClass,  // split on hmm-position
                               m);
    }
  } else if (max_set_size == 1
             && static_cast<int32>(phone_sets.size()) <= 2 * highest_numbered_phone) {
    // create table map splitting on phone-- more efficient.
    // the part after the && checks that this would not contain a very sparse vector.
    std::map<EventValueType, EventMap *> m;
    for (size_t i = 0; i < phone_sets.size(); i++) {
      std::vector<std::vector<int32> > phone_sets_tmp;
      phone_sets_tmp.push_back(phone_sets[i]);
      std::vector<bool> share_roots_tmp;
      share_roots_tmp.push_back(share_roots[i]);
      EventMap *this_stub = GetStubMap(P, phone_sets_tmp, phone2num_pdf_classes,
                                       share_roots_tmp,
                                       num_leaves_out);
      KALDI_ASSERT(m.count(phone_sets_tmp[0][0]) == 0);
      m[phone_sets_tmp[0][0]] = this_stub;
    }
    return new TableEventMap(P, m);
  } else {
    // Do a split.  Recurse.
    size_t half_sz = phone_sets.size() / 2;
    std::vector<std::vector<int32> >::const_iterator half_phones =
      phone_sets.begin() + half_sz;
    std::vector<bool>::const_iterator half_share =
      share_roots.begin() + half_sz;
    std::vector<std::vector<int32> > phone_sets_1, phone_sets_2;
    std::vector<bool> share_roots_1, share_roots_2;
    phone_sets_1.insert(phone_sets_1.end(), phone_sets.begin(), half_phones);
    phone_sets_2.insert(phone_sets_2.end(), half_phones, phone_sets.end());
    share_roots_1.insert(share_roots_1.end(), share_roots.begin(), half_share);
    share_roots_2.insert(share_roots_2.end(), half_share, share_roots.end());

    EventMap *map1 = GetStubMap(P, phone_sets_1, phone2num_pdf_classes,
                                share_roots_1, num_leaves_out);
    EventMap *map2 = GetStubMap(P, phone_sets_2, phone2num_pdf_classes,
                                share_roots_2, num_leaves_out);

    std::vector<EventKeyType> all_in_first_set;
    for (size_t i = 0; i < half_sz; i++)
      for (size_t j = 0; j < phone_sets[i].size(); j++)
        all_in_first_set.push_back(phone_sets[i][j]);
    std::sort(all_in_first_set.begin(), all_in_first_set.end());
    KALDI_ASSERT(IsSortedAndUniq(all_in_first_set));
    return new SplitEventMap(P, all_in_first_set, map1, map2);
  }
}



} // end namespace kaldi.
}
