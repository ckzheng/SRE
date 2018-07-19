#ifndef ALI_SPEECH_RESCORE_GRAMMAR_H_
#define ALI_SPEECH_RESCORE_GRAMMAR_H_
#include "als_error.h"

// the type of grammar supported 
const char kAlsBiasGrammarType[] = "AlsBiasGrammar";
const char kAlsInterpolationGrammarType[] = "AlsInterpolationGrammar";

class AlsRescoreGrammar {
public:
  virtual ~AlsRescoreGrammar() {}
  /** factory method to create a grammar
  */
  static ALSAPI_EXPORT ALS_RETCODE Create(const char *grammar_type, AlsRescoreGrammar**);
  
  /** Find a rescore grammar from grammars
  */
  static ALSAPI_EXPORT ALS_RETCODE Find(const char *grammar_name, AlsRescoreGrammar**);

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

  /** Read a compiled grammar from memory
  */
  virtual ALS_RETCODE Load(const char *mem, int size) = 0;

  /** Saves a compiled grammar to a file.
  */
  virtual ALS_RETCODE Save(const char *file_prefix) = 0;
};

#endif
