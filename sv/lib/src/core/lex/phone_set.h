#ifndef PHONESET_H
#define PHONESET_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <map>
#include "base/idec_common.h"
#include "base/idec_types.h"
#include "base/idec_return_code.h"


namespace idec {


#define MAX_PHONETIC_SYMBOL_LENGTH              10 
#define PHONETIC_SYMBOL_SILENCE                "sil"
#define PHONETIC_SYMBOL_EPSILON                "<eps>"   //just for fill garbage to be consistent with kaldi
#define PHONETIC_SYMBOL_CONTEXT_PADDING        "<>"

typedef struct {
    PhoneId     phone_id;            // phone index, 1-based as kaldi does, 0 is reserved for 
    std::string phone_str;           // phone name
    bool        is_cxt_dep;          // whether context modeling can be applied to the phone 
} Phone;

typedef std::vector<Phone*> VPhone;

class PhoneSet {

   private: 
      
      std::string file_name_;
      std::vector<std::string>  phone_names_;
      std::map<std::string, int> name2id_;
      VPhone phones_;
     
   public:
   
      //constructor
      PhoneSet(const char *file_name);
   
      //destructor
      ~PhoneSet(); 
      
      //load the phone set
      IDEC_RETCODE Load();
 
      // return the number of phones
      inline unsigned int NumPhn() {       
          return (unsigned int)phones_.size() - 1;//skip the <eps>
      }
     
      inline unsigned int PaddingSymbol(){ return (unsigned int)phones_.size(); }
         
        // return whether the phone is a special phone (can not appear in lexical word transcriptions)
        bool IsSpecialPhone(int iPhone);    
        
        inline const char *PhoneId2PhoneName(int phone_id) {
        
            if (phone_id == (int)PaddingSymbol()) {
                return PHONETIC_SYMBOL_CONTEXT_PADDING;
            }
            
            return phone_names_[phone_id].c_str();
        }
        
        // return the index of the given phonetic symbol (-1 in case it is not found)
        inline int PhoneName2PhoneId(const char *phone_name) {         
            std::map<std::string,int>::const_iterator it = name2id_.find(phone_name);
            if (it == name2id_.end()) {
                return -1;
            }
               
            return it->second;
        }    
        
        // return the index of the silence phonetic symbol
        inline int PhoneIdOfSilence() {            
            std::map<std::string, int>::const_iterator it = name2id_.find(PHONETIC_SYMBOL_SILENCE);
            if (it == name2id_.end()) {
                return -1;
            }
                
            return it->second;
        }    
        
        // return whether context modeling affect the phone
        inline bool isPhoneContextModeled(PhoneId phoneId) {
        
            assert(phoneId <= phones_.size());
        
            return phones_[phoneId]->is_cxt_dep;
        }
        
        // print the phonetic symbol set
        void Print();
      
};

};    // end-of-namespace

#endif


