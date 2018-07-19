#ifndef __ALI_SPEECH_GRAMMAR_H__
#define __ALI_SPEECH_GRAMMAR_H__
#include "als_error.h"

// the type of grammar supported 
const char kAlsCfgGrammarType[]     = "AlsCfgGrammar";
const char kAlsNgramGrammarType[]   = "AlsNgramGrammar";
const char kAlsClassLmGrammarType[] = "AlsClassLmGrammar";

enum GrammarStatus {
  NO_GRAMMAR,
  GRAMMAR_READY,
  GRAMMAR_CHANGING,
  GRAMMAR_LOADING,
  GRAMMAR_LOADED,
};

struct AlsWordClassGrammarImage {
    char        *class_name;
    const char  *grammar_type;
    char        *image;
    int         image_size;
};

class AlsGrammar {
public:
    virtual ~AlsGrammar() {}
    /** factory method to create a grammar
    */
    static ALSAPI_EXPORT ALS_RETCODE Create(const char *grammar_type, AlsGrammar**);

    /** delete itself
    */
    ALSAPI_EXPORT ALS_RETCODE Destroy();

    /** fileName  -- file name of the grammar file
    *  resourceDir -- addition 
    */
   virtual ALS_RETCODE Compile(const char *file_name, const char *resource_dir) = 0;


    /** Read a compiled grammar from file
    */
   virtual ALS_RETCODE Load(const char *file_prefix) = 0;


    /** Saves a compiled grammar to a file.
    */
   virtual ALS_RETCODE Save(const char *file_prefix) = 0;
};

#endif
