#include "util/file_input.h"
#include "util/io_base.h"
#include "base/log_message.h"
#include "lex/phone_set.h"
#include <ctype.h>



namespace idec {
using namespace std;

//constructor
PhoneSet::PhoneSet(const char* str_file) {

   file_name_ = str_file;
}

// destructor
PhoneSet::~PhoneSet() {

    for(VPhone::iterator it = phones_.begin() ; it != phones_.end() ; ++it) {
        delete *it;
    }
}
   
// load the phone set
IDEC_RETCODE PhoneSet::Load() {

    IDEC_RETCODE  ret = IDEC_SUCCESS;
    FileInput file(file_name_.c_str(),false);
    ret = file.Open();
    if (ret != IDEC_SUCCESS) {
        return ret;
    }



    //push back the <eps>
    string phone_name = PHONETIC_SYMBOL_EPSILON;
    PhoneId phone_id = (PhoneId)(phone_names_.size());
    phone_names_.push_back(phone_name);
    name2id_.insert(map<string, int>::value_type(phone_name, (int)phone_id));
    
    Phone *phone = new Phone;
    phone->phone_id = (PhoneId)(phone_names_.size());
    phone->phone_str = phone_name;
    phones_.push_back(phone);
    
    string str_line;
    int num_line = 0;
    while(std::getline(file.GetStream(), str_line)) {
        
        ++num_line;
    
        // skip comments and blank lines
        if (str_line.empty() || (str_line.c_str()[0] == '#')) {
            continue;
        }
        
        // read the phone
        std::stringstream s(str_line);
        string str_phone;
        IOBase::ReadString(s, str_phone);

        //skip the <eps>
        if (str_phone == PHONETIC_SYMBOL_EPSILON){
           continue;
        }

        
        Phone *phone = new Phone;
        phone->is_cxt_dep = true;
        if ((str_phone.c_str()[0] == '(') && (str_phone.c_str()[str_phone.length()-1] == ')')) {    
            phone->is_cxt_dep = false;
            str_phone = str_phone.substr(1,str_phone.length()-2);
        }    
        
        // check length
        if (str_phone.length() > MAX_PHONETIC_SYMBOL_LENGTH) {
            IDEC_ERROR << "incorrect phonetic symbol name \"" << str_phone << "\" found in line " << num_line << ", too many characters";
        }
        
        // check that the phone name is correct
        for (unsigned int i = 0; i < str_phone.length(); ++i) {
            if (!isalnum(str_phone.c_str()[i]) && (str_phone.c_str()[i] != '_')) {
                IDEC_ERROR << "incorrect phonetic symbol name \"" << str_phone << "\" found in line " << num_line << ", wrong symbol";
            }
        }

        if (phone_names_.size()>MAX_BASEPHONES) {
            IDEC_ERROR << "only support phone set smaller than 255!";
        }
        PhoneId phone_id = (PhoneId)(phone_names_.size());
        phone_names_.push_back(str_phone);
        name2id_.insert(map<string, int>::value_type(str_phone, (int)(phone_id)));

        phone->phone_id = phone_id;
        phone->phone_str = str_phone;
        phones_.push_back(phone);
    }
    
    file.Close();
   
   // check that there is at least one phone
   if (phone_names_.empty()) {
       IDEC_ERROR << "no phonetic symbols found in the phonetic symbol file";
   }
   
   // check that the silence is defined
   if (phone_names_.size() != name2id_.size() || 
       phone_names_.size() != phones_.size() ||
       name2id_.size() != phones_.size()
       ) {
           IDEC_ERROR << "s.th.wrong in loading code";
   }    

   if (name2id_.find(PHONETIC_SYMBOL_SILENCE) == name2id_.end()) {
       IDEC_ERROR << "phonetic symbol " << PHONETIC_SYMBOL_SILENCE << " must be defined in the phonetic symbol file";
   }
   return ret;
}

// print the phonetic symbol set
void PhoneSet::Print() {

    cout << "-- phonetic symbol set ---------------------------------------------------\n";
    cout << " file: " << file_name_ << "\n";
    cout << " #phones: " << phones_.size() << "\n";
    for(VPhone::iterator it = phones_.begin(); it != phones_.end(); ++it) {
        cout << static_cast<int>((*it)->phone_id) << " " << (*it)->phone_str;
        if ((*it)->is_cxt_dep == false) {
            cout << " (context independent)\n";
        } else {
            cout << "\n";
        }
    }
    cout << "--------------------------------------------------------------------------\n";
}    

};    // end-of-namespace

