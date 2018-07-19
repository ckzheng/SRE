#include <iostream>
#include "base/log_message.h"
#include "util/io_base.h"
#include "util/file_output.h"
#include "util/text-utils.h"
#include "util/scoped_ptr.h"
#include "util/encode_converter.h"
#include "lex/phone_set.h"
#include "lex/lts.h"
#include "lex/sorted_pronunciation_lexicon.h"

namespace idec{
#pragma pack(push,1)
    struct SortedPronunciationLexicon::BinHeader {
        LexUnitId  num_lex_unit;
        LexUnitXId num_lex_unitx;
        LexUnitId  num_special_lex_unit;
        LexUnitXId num_special_lex_unitx;
        uint8_t    num_bit_compile_platform;
        uint32     word_str_buf_size;
        uint32     phone_buf_size;
    };
#pragma pack(pop)

    SortedPronunciationLexicon::SortedPronunciationLexicon(PhoneSet *phone_set) :PronunciationLexicon(phone_set_){
        this->phone_set_ = phone_set;
        this->lex_unitx_ = NULL;
        this->lex_unit_ = NULL;
        num_lex_unit_ = 0;
        num_lex_unitx_ = 0;
        num_special_lex_unit_ = 0;
        num_special_lex_unitx_ = 0;
        phone_buf_ = NULL;
        word_str_buf_ = NULL;
    }

    SortedPronunciationLexicon::~SortedPronunciationLexicon() {
        Clear();
    }

    IDEC_RETCODE SortedPronunciationLexicon::LoadTxtFile(const char *file_name) {
        //1) load line by line, construct lex_unit
        std::list<LexUnitTxtLoading *> all_lex_unit;
        char line_buf[1024];
        FILE *fp = fopen(file_name, "rt");
        if (fp == NULL) {
            return IDEC_OPEN_ERROR;
        }
        // parsing them now
        while (fgets(line_buf, sizeof(line_buf), fp) != NULL) {
            LexUnitTxtLoading *lex_unit = NULL;
            ReadLexUnit(line_buf, &lex_unit);
            if (lex_unit != NULL) {
                all_lex_unit.push_back(lex_unit);
            }
        }
        fclose(fp);

        //2) sort by str,
        //3) construct lex_unitx and fill lex_unit_id/lex_unitx_id/pronunciation variation
        BuildFromTxtLoadUnit(all_lex_unit);
        return IDEC_SUCCESS;
    }


    void SortedPronunciationLexicon::BuildFromTxtLoadUnit(std::list<LexUnitTxtLoading*> &all_lex_unit) {
        // 1) sort & uniq
        all_lex_unit.sort(CmpByStringAndPhone);

        size_t num_uniq = 0;
        std::list<LexUnitTxtLoading*>::iterator it = all_lex_unit.begin();
        std::list<LexUnitTxtLoading*>::iterator result = all_lex_unit.begin();

        if (all_lex_unit.size() == 1) {
            num_uniq = 1;
        }
        else {
            int num_pron_var = 0;
            int max_pron_var_allowed = 250;
            std::string last_word = "";
            while (++it != all_lex_unit.end()) {
                bool string_and_phone_eqn = EqualStringAndPhone(*result, *it);
                bool string_eqn = EqualString(*result, *it);

                // count the num_pron_var of current word
                if (string_eqn) {
                    num_pron_var++;
                }
                else {
                    if (num_pron_var > max_pron_var_allowed) {
                        IDEC_WARNING
#ifdef _MSC_VER
                            << EncodeConverter::UTF8ToLocaleAnsi(last_word) << ": "
#endif
                            << num_pron_var << " of pron var, truncate to " << max_pron_var_allowed;
                    }
                    num_pron_var = 0;
                }



                last_word = (*it)->word_str;
                if (string_and_phone_eqn || num_pron_var > max_pron_var_allowed) {
                    IDEC_DELETE_ARRAY((*it)->phones);
                    IDEC_DELETE_ARRAY((*it)->word_str);
                    IDEC_DELETE(*it);
                }
                else {
                    *(++result) = *it;
                    if (num_uniq == 0) {
                        num_uniq++;
                    }
                    num_uniq++;
                }
            }
        }
        all_lex_unit.resize(num_uniq);
        // remove the extra pronunciation variants
        // we only support 250  pronunciation variants per word


        // insert the specials
        InsertSpecials(all_lex_unit);

        // 4)  build the lex_unitx and fill all the ids
        BuildCompact(all_lex_unit);
    }



    void SortedPronunciationLexicon::InsertSpecials(std::list<LexUnitTxtLoading*> &all_lex_units) {

        // (1) insert the sentence delimiters and the unknown symbol (if not in the language model they will be removed later)
        // unknown (before the word identity is known)
        LexUnitTxtLoading* lexUnitUnknown = new LexUnitTxtLoading;
        lexUnitUnknown->lex_unit_type = LEX_UNIT_TYPE_UNKNOWN;
        char *strLexUnitFit = new char[strlen(LEX_UNIT_UNKNOWN) + 1];
        strcpy(strLexUnitFit, LEX_UNIT_UNKNOWN);
        lexUnitUnknown->word_str = strLexUnitFit;
        lexUnitUnknown->num_phones = 0;
        lexUnitUnknown->phones = NULL;

    
        // 2) end of sentence "</s>" (must be created before the beginning of sentence to preserve alphabetical order)
        LexUnitTxtLoading *lexUnitEndSentence = new LexUnitTxtLoading();
        lexUnitEndSentence->lex_unit_type = LEX_UNIT_TYPE_SENTENCE_DELIMITER;
        strLexUnitFit = new char[strlen(LEX_UNIT_END_SENTENCE) + 1];
        strcpy(strLexUnitFit, LEX_UNIT_END_SENTENCE);
        lexUnitEndSentence->word_str = strLexUnitFit;
        lexUnitEndSentence->num_phones = 0;
        lexUnitEndSentence->phones = NULL;
 
        //3) beginning of sentence "<s>"
        LexUnitTxtLoading*lexUnitBegSentence = new LexUnitTxtLoading();
        lexUnitBegSentence->lex_unit_type = LEX_UNIT_TYPE_SENTENCE_DELIMITER;
        strLexUnitFit = new char[strlen(LEX_UNIT_BEG_SENTENCE) + 1];
        strcpy(strLexUnitFit, LEX_UNIT_BEG_SENTENCE);
        lexUnitBegSentence->word_str = strLexUnitFit;
        lexUnitBegSentence->num_phones = 0;
        lexUnitBegSentence->phones = NULL;
 

        //4) the special silence word
        LexUnitTxtLoading* lexUnitSilence = new LexUnitTxtLoading();
        lexUnitSilence->lex_unit_type = LEX_UNIT_TYPE_SIL;
        strLexUnitFit = new char[strlen(LEX_UNIT_SILENCE_SYMBOL) + 1];
        strcpy(strLexUnitFit, LEX_UNIT_SILENCE_SYMBOL);
        lexUnitSilence->word_str = strLexUnitFit;
        lexUnitSilence->num_phones = 1;
        lexUnitSilence->phones = new PhoneId[1];
        lexUnitSilence->phones[0] = (PhoneId)phone_set_->PhoneIdOfSilence();



        all_lex_units.push_front(lexUnitSilence); // 3
        all_lex_units.push_front(lexUnitBegSentence); // 2
        all_lex_units.push_front(lexUnitEndSentence); // 1
        all_lex_units.push_front(lexUnitUnknown); // 0

        num_special_lex_unit_ = 4;
        num_special_lex_unitx_ = 4;
    }


    // pre-requisite : all_lex_units is sort & uniq on string field
    // note that the phones and word_str memory is transfer to internals 
    void SortedPronunciationLexicon::BuildCompact(std::list<LexUnitTxtLoading*> &all_lex_units) {
        // clear the old
        Clear();


        num_lex_unit_ = static_cast<LexUnitId>(all_lex_units.size());
        num_lex_unitx_ = 0;
        word_str_buf_size_ = 0;
        phone_buf_size_ = 0;
        
        // scan all_lex_unit for counting
        const char* last_str = "";
        for (std::list<LexUnitTxtLoading*>::iterator it = all_lex_units.begin(); it != all_lex_units.end();it++) {
            phone_buf_size_ += (*it)->num_phones;
            if (strcmp(last_str, (*it)->word_str) != 0) {
                num_lex_unitx_++;
                last_str = (*it)->word_str;
                word_str_buf_size_ += static_cast<uint32>(strlen(last_str) + 1);
            }
        }


        lex_unit_ = new LexUnit[num_lex_unit_];
        lex_unitx_ = new LexUnitX[num_lex_unitx_];
        word_str_buf_ = new char[word_str_buf_size_];
        phone_buf_ = new PhoneId[phone_buf_size_/sizeof(PhoneId)];

        // another scanning for fill all the things
        LexUnitId  lex_unit_id = 0;
        LexUnitXId lex_unitx_id = 0;
        uint8_t pron_variant_id = 0;
        LexUnitX *cur_lex_unitx = NULL;
        last_str = "";

        char* word_ptr = word_str_buf_;
        PhoneId* phone_ptr = phone_buf_;


        for (std::list<LexUnitTxtLoading*>::iterator it = all_lex_units.begin(); it != all_lex_units.end(); it++) {
            // new lex_unitx
            if (strcmp(last_str, (*it)->word_str) != 0) {
                last_str = (*it)->word_str;
                cur_lex_unitx = lex_unitx_ + lex_unitx_id;
                cur_lex_unitx->lex_unit_id_start = lex_unit_id;
                cur_lex_unitx->word_str = word_ptr;
                strcpy(cur_lex_unitx->word_str, (*it)->word_str);
                word_ptr += ( strlen(word_ptr) +1);
                cur_lex_unitx->num_pron_vairant = 0;
                pron_variant_id = 0;
                lex_unitx_id++;
            }

            // new lex_unit
            lex_unit_[lex_unit_id].lex_unit_type = (*it)->lex_unit_type;
            lex_unit_[lex_unit_id].num_phones = (*it)->num_phones;
            lex_unit_[lex_unit_id].phones = phone_ptr;
            if ((*it)->num_phones > 0) {
                memcpy(phone_ptr, (*it)->phones, (*it)->num_phones);
            }
            phone_ptr += ((*it)->num_phones);
            lex_unit_[lex_unit_id].lex_unit_id = lex_unit_id;
            lex_unit_[lex_unit_id].pron_variant_id = pron_variant_id;

            cur_lex_unitx->num_pron_vairant++;
            lex_unit_id++;
            pron_variant_id++;
        }
        IDEC_ASSERT(word_str_buf_size_ == static_cast<size_t>(word_ptr - word_str_buf_));
        IDEC_ASSERT(phone_buf_size_ == (phone_ptr - phone_buf_)*sizeof(PhoneId));

        // free all the temp memories
        for (std::list<LexUnitTxtLoading*>::iterator it = all_lex_units.begin(); it != all_lex_units.end(); it++) {
            IDEC_DELETE_ARRAY((*it)->phones);
            IDEC_DELETE_ARRAY((*it)->word_str);
            IDEC_DELETE(*it);
        }
        all_lex_units.clear();
    }



    IDEC_RETCODE SortedPronunciationLexicon::Save(const char *file_name) {

        IDEC_RETCODE ret = IDEC_SUCCESS;
        FileOutput fo(file_name, true);
        ret = fo.Open();
        if (ret == IDEC_SUCCESS) {
            Save(fo.GetStream());
            fo.Close();
        }

        return ret;
    }
    IDEC_RETCODE SortedPronunciationLexicon::Load(const char *file_name) {
     
        IDEC_RETCODE ret = IDEC_SUCCESS;
        if (IsValidBinaryFile(file_name)) {
            FileInput fi(file_name, true);
            ret = fi.Open();
            if (ret == IDEC_SUCCESS) {
                Load(fi.GetStream());
                fi.Close();
            }
        }
        else {
            ret = LoadTxtFile(file_name);
        }
        return ret;
    }


    IDEC_RETCODE SortedPronunciationLexicon::SaveTxtFile(const char *file_name) {
        using namespace  std;

        IDEC_RETCODE ret = IDEC_SUCCESS;
        FileOutput fo(file_name, false);
        ret = fo.Open();
        if (ret == IDEC_SUCCESS) {
            ostream &oss = fo.GetStream();
            for (LexUnitXId id = 0; id < static_cast<LexUnitXId> (NumOfLexUnitX()); id++) {
                for (Scoped_Ptr<LexUnitIterator> it(LexUnitXId2LexUnits(id)); it->HasNext(); it->Next()) {
                    LexUnit * lex_unit = it->CurrentItem();

                    if (IsSpecialSymbol(lex_unit))
                        continue;

                    oss << LexUnitXId2Str(id) << " ";
                    for (int p = 0; p < lex_unit->num_phones; p++) {
                        oss << phone_set_->PhoneId2PhoneName(lex_unit->phones[p]) << " ";
                    }
                    oss << "\n";
                }
            }
            fo.Close();
        }
        return ret;
    }



    LexUnit* SortedPronunciationLexicon::SearchSpecial(const char *str) {
        // first search the specials since the specials are not sorted by str
        for (LexUnitXId id = 0; id < num_special_lex_unitx_; id++) {
            if (strcmp(str, lex_unitx_[id].word_str) == 0) {
                return lex_unit_ + lex_unitx_[id].lex_unit_id_start;
            }
        }
        return NULL;
    }


    LexUnit *SortedPronunciationLexicon::LexUnitSilence() {
        return SearchSpecial(LEX_UNIT_SILENCE_SYMBOL);
    }
    LexUnit *SortedPronunciationLexicon::LexUnitSentenceBegin() {
        return SearchSpecial(LEX_UNIT_BEG_SENTENCE);
    }
    LexUnit *SortedPronunciationLexicon::LexUnitSentenceEnd() {
        return SearchSpecial(LEX_UNIT_END_SENTENCE);
    }

    // bsearch on lexUnitX 
    LexUnitXId  SortedPronunciationLexicon::Str2LexUnitXId(const char *str) const {
        const LexUnitX* match = Str2LexUnitX(str);
        // find Lex it
        if (match != NULL) {
            return static_cast<LexUnitXId>(match - lex_unitx_);
        }
        else {
            return kInvalidLexUnitId;
        }
    }

    const LexUnitX * SortedPronunciationLexicon::Str2LexUnitX(const char *str) const {

        // first search the specials since the specials are not sorted by str
        for (LexUnitXId id = 0; id < num_special_lex_unitx_; id++) {
            if (strcmp(str, lex_unitx_[id].word_str) == 0) {
                return lex_unitx_ + id;
            }
        }

        LexUnitX lex_unitx_k;
        lex_unitx_k.word_str = const_cast<char*>(str);
        LexUnitX* lo = std::lower_bound(lex_unitx_ + num_special_lex_unitx_, lex_unitx_ + num_lex_unitx_, lex_unitx_k, CmpByLexStr);

        // find it
        if (lo != lex_unitx_ + num_lex_unitx_ && strcmp(lo->word_str, str) == 0) {
            return lo;
        }
        else {
            return NULL;
        }
    }

    //str=>lexUnitX=>LexUnit
    LexUnitIterator*  SortedPronunciationLexicon::Str2LexUnits(const char *str) {
        const LexUnitX* match = Str2LexUnitX(str);
        if (match != NULL) {
            return new SortedLexUnitIterator(lex_unit_ + match->lex_unit_id_start,
                                             lex_unit_ + match->lex_unit_id_start + match->num_pron_vairant);
        }
        else {
            return NULL;
        }
    }

    LexUnitIterator* SortedPronunciationLexicon::LexUnitXId2LexUnits(const LexUnitXId &lex_unitx_id) {
        if (lex_unitx_id < 0 || lex_unitx_id >= num_lex_unitx_)
            return NULL;
        else {
            const LexUnitX* match = lex_unitx_ + lex_unitx_id;
            return new SortedLexUnitIterator(lex_unit_ + match->lex_unit_id_start,
                                             lex_unit_ + match->lex_unit_id_start + match->num_pron_vairant);
        }
    }


    const LexUnit* SortedPronunciationLexicon::LexUnitId2LexUnit(const LexUnitId &lex_unit_id) {
        if (lex_unit_id < 0 || lex_unit_id >= num_lex_unit_)
            return NULL;
        else {
            return lex_unit_ + lex_unit_id;
        }
    }

    LexUnitIterator*  SortedPronunciationLexicon::GetLexUnitIterator() {
        return new SortedLexUnitIterator(lex_unit_, lex_unit_ + num_lex_unit_);
    }

    const char*  SortedPronunciationLexicon::LexUnitXId2Str(const LexUnitXId &lex_unitx_id) {
        if (lex_unitx_id < 0 || lex_unitx_id >= num_lex_unitx_)
            return NULL;
        else {
            return lex_unitx_[lex_unitx_id].word_str;
        }
    }
    // bsearch in lex_unitx_, key is lex_unit_id
    const LexUnitXId SortedPronunciationLexicon::LexUnitIdToLexUnitXId(const LexUnitId &lex_unit_id) {
        if (lex_unit_id < 0 || lex_unit_id >= num_lex_unit_)
            return kInvalidLexUnitId;

        LexUnitX lex_unitx_k;
        lex_unitx_k.lex_unit_id_start = lex_unit_id;
        LexUnitX* up = std::upper_bound(lex_unitx_, lex_unitx_ + num_lex_unitx_, lex_unitx_k, CmpByLexUnitId);

        IDEC_ASSERT((up == (lex_unitx_ + num_lex_unitx_))
                    || (up->lex_unit_id_start > lex_unit_id && (up - 1)->lex_unit_id_start <= lex_unit_id));
        return static_cast<LexUnitXId>(up - lex_unitx_ - 1);
    }

    const char* SortedPronunciationLexicon::LexUnitId2Str(const LexUnitId &lex_unit_id) {
        const LexUnitXId lex_unitx_id = LexUnitIdToLexUnitXId(lex_unit_id);
        if (lex_unitx_id == kInvalidLexUnitId)
            return NULL;
        return LexUnitXId2Str(lex_unitx_id);
    }


    void SortedPronunciationLexicon::Clear() {
        for (LexUnitXId id = 0; id < num_lex_unit_; id++) {
            lex_unit_[id].phones = NULL;
        }

        // write lex unitx
        for (LexUnitXId id = 0; id < num_lex_unitx_; id++) {
            lex_unitx_[id].word_str = NULL;
        }
        IDEC_DELETE_ARRAY(lex_unit_);
        IDEC_DELETE_ARRAY(lex_unitx_);
        num_lex_unit_ = 0;
        num_lex_unitx_ = 0;
        IDEC_DELETE_ARRAY(phone_buf_);
        IDEC_DELETE_ARRAY(word_str_buf_);
        word_str_buf_size_ = 0;
        phone_buf_size_ = 0;

    }

    // image layout:

    // hdr
    // lex_unit[] w/o phones
    // phone table

    // lex_unitx[]
    // string length table
    // string table
    void SortedPronunciationLexicon::Save(std::ostream &oss) {
        BinHeader hdr;
        hdr.num_lex_unit = num_lex_unit_;
        hdr.num_lex_unitx = num_lex_unitx_;
        hdr.num_special_lex_unit = num_special_lex_unit_;
        hdr.num_special_lex_unitx = num_special_lex_unitx_;
        hdr.word_str_buf_size = word_str_buf_size_;
        hdr.phone_buf_size = phone_buf_size_;
        hdr.num_bit_compile_platform = sizeof(void*) * 8;

        // write marker
        oss.write("sortedpls", 9);

        // write header
        oss.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

        // write lex_unit
        size_t tol_phone_num = 0;
        for (LexUnitId id = 0; id < num_lex_unit_; id++) {
            WriteLexUnit(oss, lex_unit_[id]);
            tol_phone_num += lex_unit_[id].num_phones;
        }
        
        // write phone table
        IDEC_ASSERT(tol_phone_num*sizeof(PhoneId) == phone_buf_size_);
        oss.write(reinterpret_cast<const char*>(phone_buf_), phone_buf_size_);


        // write lex unitx
        for (LexUnitXId id = 0; id < num_lex_unitx_; id++) {
            WriteLexUnitX(oss, lex_unitx_[id]);
        }

        // write string table
        oss.write(reinterpret_cast<const char*>(word_str_buf_), word_str_buf_size_);

        // write the string length table
        size_t tot_str_len = 0;
        for (LexUnitXId id = 0; id < num_lex_unitx_; id++) {
            uint16 str_len_plus1 = static_cast<uint16>(strlen(lex_unitx_[id].word_str) +1);
            oss.write(reinterpret_cast<const char*>(&str_len_plus1), sizeof(str_len_plus1));
            tot_str_len += (str_len_plus1);
        }
        IDEC_ASSERT(tot_str_len*sizeof(char) == word_str_buf_size_);
    }


    bool SortedPronunciationLexicon::IsValidBinaryFile(const char *file_name) {

        bool ret = false;
        FileInput fi(file_name, true);
        fi.Open();
        // read marker
        char marker[9];
        fi.GetStream().read(marker, 9);
        if (strncmp(marker, "sortedpls", 9) == 0) {
            ret = true;
        }
        fi.Close();

        return ret;
    }


    void SortedPronunciationLexicon::Load(std::istream &iss) {
        Clear();

        // read marker
        char marker[9];
        iss.read(marker, 9);
        if (strncmp(marker, "sortedpls", 9) != 0) {
            IDEC_ERROR << "wrong sorted pls image";
        }

        // read header
        BinHeader hdr;
        iss.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (hdr.num_bit_compile_platform != sizeof(void*) * 8) {
            /*IDEC_ERROR << "fatal error, binary lexicon is built on "
                << hdr.num_bit_compile_platform << "-bits platform"
                << "but runtime code is compiled with" << sizeof(void*) * 8 << "-bits platform\n";
        */}

        num_lex_unit_ = hdr.num_lex_unit;
        num_lex_unitx_ = hdr.num_lex_unitx;
        num_special_lex_unit_ = hdr.num_special_lex_unit;
        num_special_lex_unitx_ = hdr.num_special_lex_unitx;
        word_str_buf_size_ = hdr.word_str_buf_size;
        phone_buf_size_ = hdr.phone_buf_size;

        lex_unit_ = new LexUnit[num_lex_unit_];
        lex_unitx_ = new LexUnitX[num_lex_unitx_];
        phone_buf_ = new PhoneId[phone_buf_size_/sizeof(PhoneId)];
        word_str_buf_ = new char[word_str_buf_size_];


        // read lex_unit
        for (LexUnitId id = 0; id < num_lex_unit_; id++) {
            ReadLexUnit(iss, lex_unit_ + id);
        }

        // build the phone table
        iss.read(reinterpret_cast<char*>(phone_buf_), phone_buf_size_);
        PhoneId* phone_ptr = phone_buf_;
        for (LexUnitId id = 0; id < num_lex_unit_; id++) {
            lex_unit_[id].phones = phone_ptr;
            phone_ptr += lex_unit_[id].num_phones;
        }
        IDEC_ASSERT((phone_ptr - phone_buf_)*sizeof(PhoneId) == phone_buf_size_);

        
        // read lex unitx
        for (LexUnitXId id = 0; id < num_lex_unitx_; id++) {
            ReadLexUnitX(iss, lex_unitx_ + id);
        }

        // build the string table
        iss.read(reinterpret_cast<char*>(word_str_buf_), word_str_buf_size_);

        char* word_ptr = word_str_buf_;
        for (LexUnitXId id = 0; id < num_lex_unitx_; id++) {
            lex_unitx_[id].word_str = word_ptr;

            uint16 word_len = 0;
            iss.read(reinterpret_cast<char*>(&word_len), sizeof(word_len));
#ifdef _DEBUG
            IDEC_ASSERT((strlen(word_ptr) + 1) == (size_t)word_len);
#endif
            word_ptr += word_len;
        }
        IDEC_ASSERT((word_ptr - word_str_buf_)*sizeof(char) == word_str_buf_size_);


    }

    void SortedPronunciationLexicon::ReadLexUnit(std::istream &iss, LexUnit *lex_unit) {
        iss.read(reinterpret_cast<char*>(lex_unit), sizeof(*lex_unit) - sizeof(PhoneId*));
        lex_unit->phones = NULL;
    }
    void SortedPronunciationLexicon::WriteLexUnit(std::ostream &oss, const LexUnit &lex_unit) {
        oss.write(reinterpret_cast<const char*>(&lex_unit), sizeof(lex_unit)-sizeof(PhoneId*));
    }
    void SortedPronunciationLexicon::ReadLexUnitX(std::istream &iss, LexUnitX *lex_unitx) {
        iss.read(reinterpret_cast<char*>(lex_unitx), sizeof(*lex_unitx) -sizeof(char*));
        lex_unitx->word_str = NULL;
    }
    void SortedPronunciationLexicon::WriteLexUnitX(std::ostream &oss, const LexUnitX &lex_unitx) {
        oss.write(reinterpret_cast<const char*>(&lex_unitx), sizeof(lex_unitx) - sizeof(char*));
    }

#ifdef CONVERT_ENGLISH_TO_LOWER_LEXICON
    static void ToLower(std::string &input_text) {
        for (size_t i = 0; i < input_text.size(); i++) {
            if (isascii(input_text[i]) && isalpha(input_text[i])) {
                if (isupper(input_text[i])) {
                    input_text[i] = tolower(input_text[i]);
                }  
            }
        }
    }
#endif

    // parsing from a text line
    void SortedPronunciationLexicon::ReadLexUnit(const char *line, LexUnitTxtLoading **lex_unit_loading) {
        using namespace  std;
        *lex_unit_loading = NULL;
        // skip comments
        if ((line[0] == '#') && (line[1] == '#')){
            return;
        }

        vector<string> tokens;
        string line_trimed = line;
        Trim(&line_trimed);
        SplitStringToVector(line_trimed, "\n\t ", true, &tokens);
        if (tokens.size() >= 1 
            && PronunciationLexicon::GetLexUnitType(tokens[0].c_str()) != LEX_UNIT_TYPE_SENTENCE_DELIMITER
            && PronunciationLexicon::GetLexUnitType(tokens[0].c_str()) != LEX_UNIT_TYPE_UNKNOWN) {
            *lex_unit_loading = new LexUnitTxtLoading();
            LexUnitTxtLoading * u(*lex_unit_loading);

            // str, convert into lower-case
#ifdef CONVERT_ENGLISH_TO_LOWER_LEXICON
             ToLower(tokens[0]);
#endif
            u->word_str = new char[tokens[0].size() + 1];
            strcpy(u->word_str, tokens[0].c_str());

            // lex type
            u->lex_unit_type = PronunciationLexicon::GetLexUnitType(tokens[0].c_str());

            // phones
            u->num_phones = (uint16)tokens.size() - 1;
            u->phones = NULL;
            if (u->num_phones) {
                u->phones = new PhoneId[u->num_phones];

                for (size_t p = 1; p < tokens.size(); p++) {
                    int phone_idx = phone_set_->PhoneName2PhoneId(tokens[p].c_str());
                    if (phone_idx == -1) {
                        IDEC_ERROR << "unknown phonetic symbol: \"" << tokens[p] <<
                            "\", it is not defined in the phonetic symbol set: line " << line;
                    }
                    u->phones[p - 1] = (PhoneId) phone_idx;
                }
            }
        }
    }


    // build itself from a word_list
    void SortedPronunciationLexicon::BuildFromWordList(const std::vector<std::string> &word_list,
                           PronunciationLexicon *base_lexicon, Lts *lts,
                           std::vector<std::string> &oov_list, const LtsOption &lts_opt) {

        // clear first
        IDEC_ASSERT(lts != NULL);
        oov_list.resize(0);
        Clear();

        std::list<LexUnitTxtLoading *> all_lex_unit;
        std::vector<std::vector<PhoneId> > phones;
        for (std::vector<std::string>::const_iterator word_iterator = word_list.begin(); word_iterator != word_list.end(); ++word_iterator) {

            // 1) get from lexicon first
            // 2) then call lts
           /* Scoped_Ptr<LexUnitIterator> lu_iterator(base_lexicon->Str2LexUnits(word_iterator->c_str()));
            if (lu_iterator.Get() != NULL) {
                while (lu_iterator->HasNext()) {
                    LexUnit *lex_unit = lu_iterator->CurrentItem();

                    // skip the specials
                    if (!IsSpecialSymbol(lex_unit)) {
                        LexUnitTxtLoading *lex_unit_txt = new LexUnitTxtLoading();
                        // copy the lex_unit part
                        lex_unit_txt->CopyFrom(*lex_unit);
                        // copy string
                        lex_unit_txt->word_str = new char[word_iterator->size() + 1];
                        strcpy(lex_unit_txt->word_str, word_iterator->c_str());
                        all_lex_unit.push_back(lex_unit_txt);

                        // make a word ending with silence phone
                        if (add_sil_proun
                            && lex_unit->num_phones > 0
                            && lex_unit->phones[lex_unit->num_phones - 1] != phone_set_->PhoneIdOfSilence()) {

                            LexUnitTxtLoading *lex_unit_sil = new LexUnitTxtLoading();
                            // copy the lex_unit part
                            lex_unit_sil->CopyFrom(*lex_unit);
                            // add one silence phone
                            lex_unit_sil->num_phones++;
                            IDEC_DELETE_ARRAY(lex_unit_sil->phones);
                            lex_unit_sil->phones = new PhoneId[lex_unit_sil->num_phones];
                            memcpy(lex_unit_sil->phones, lex_unit->phones, lex_unit->num_phones);
                            lex_unit_sil->phones[lex_unit_sil->num_phones - 1] = phone_set_->PhoneIdOfSilence();
                            // copy string
                            lex_unit_sil->word_str = new char[word_iterator->size() + 1];
                            strcpy(lex_unit_sil->word_str, word_iterator->c_str());
                            all_lex_unit.push_back(lex_unit_sil);
                        }
                    }

                    lu_iterator->Next();
                }
            }
            // we must limit the number of prouns generated for one word
            // since the proun variant id is uchar and max is 255
            // another reason we do not think it make sense to have a word with many prouns
            else */
            if (kLtsSuccess == lts->TextToPhone(*word_iterator, &phones, lts_opt)) {
                for (size_t t = 0; t < phones.size(); t++) {
                    LexUnitTxtLoading *lex_unit = new LexUnitTxtLoading();

                    // type
                    lex_unit->lex_unit_type = LEX_UNIT_TYPE_STANDARD;//?

                    // phones
                    lex_unit->num_phones = static_cast<uint16> (phones[t].size());
                    lex_unit->phones = new PhoneId[phones[t].size()];
                    memcpy(lex_unit->phones, &(phones[t][0]), lex_unit->num_phones);
                    
                    // string
                    lex_unit->word_str = new char[word_iterator->size() + 1];
                    strcpy(lex_unit->word_str, word_iterator->c_str());
                    all_lex_unit.push_back(lex_unit);

                    // make a word ending with silence phone
                    /*if (add_sil_proun
                        && (lex_unit->num_phones > 0)
                        && (lex_unit->phones[lex_unit->num_phones - 1] != phone_set_->PhoneIdOfSilence())) {
                        LexUnitTxtLoading *lex_unit_sil = new LexUnitTxtLoading();
                        // type
                        lex_unit_sil->lex_unit_type = LEX_UNIT_TYPE_STANDARD;//?

                        // phones
                        lex_unit_sil->num_phones = static_cast<uint16>(phones[t].size() + 1);
                        lex_unit_sil->phones = new PhoneId[phones[t].size() + 1];
                        memcpy(lex_unit_sil->phones, &(phones[t][0]), lex_unit->num_phones);
                        lex_unit_sil->phones[lex_unit_sil->num_phones - 1] = phone_set_->PhoneIdOfSilence();


                        // string
                        lex_unit_sil->word_str = new char[word_iterator->size() + 1];
                        strcpy(lex_unit_sil->word_str, word_iterator->c_str());

                        all_lex_unit.push_back(lex_unit_sil);
                    }*/
                }
            }
            else {
                oov_list.push_back(*word_iterator);
            }
        }

        // the real building thing
        BuildFromTxtLoadUnit(all_lex_unit);
    }


};

