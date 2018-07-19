#ifndef __ALI_RECOGNIZER_H__
#define __ALI_RECOGNIZER_H__
#include "als_audio_segment.h"
#include "als_grammar.h"
#include "als_rescore_grammar.h"
#include "als_recognizer_result.h"


/**
* Recognizer status.
*/
typedef enum AlsRecognizerStatus_t {
    /**
    * Reserved value.
    */
    SR_RECOGNIZER_EVENT_INVALID,
    /**
    * Recognizer could not find a match for the utterance.
    */
    SR_RECOGNIZER_EVENT_NO_MATCH,
    /**
    * Recognizer processed one frame of audio.
    */
    SR_RECOGNIZER_EVENT_INCOMPLETE,
    /**
    * Recognizer has just been started.
    */
    SR_RECOGNIZER_EVENT_STARTED,
    /**
    * Recognizer is stopped.
    */
    SR_RECOGNIZER_EVENT_STOPPED,
    /**
    * Beginning of speech detected.
    */
    SR_RECOGNIZER_EVENT_START_OF_VOICING,
    /**
    * End of speech detected.
    */
    SR_RECOGNIZER_EVENT_END_OF_VOICING,
    /**
    * Beginning of utterance occurred too soon.
    */
    SR_RECOGNIZER_EVENT_SPOKE_TOO_SOON,
    /**
    * Recognition match detected.
    */
    SR_RECOGNIZER_EVENT_RECOGNITION_RESULT,
    /**
    * Timeout occurred before beginning of utterance.
    */
    SR_RECOGNIZER_EVENT_START_OF_UTTERANCE_TIMEOUT,
    /**
    * Timeout occurred before speech recognition could complete.
    */
    SR_RECOGNIZER_EVENT_RECOGNITION_TIMEOUT,
    /**
    * Not enough samples to process one frame.
    */
    SR_RECOGNIZER_EVENT_NEED_MORE_AUDIO,
    /**
    * More audio encountered than is allowed by 'swirec_max_speech_duration'.
    */
    SR_RECOGNIZER_EVENT_MAX_SPEECH,
} AlsRecognizerStatus;

/**
* Type of RecognizerResult returned by SR_RecognizerAdvance().
*/
typedef enum AlsRecognizerResultType_t {
    /**
    * Reserved value.
    */
    SR_RECOGNIZER_RESULT_TYPE_INVALID,
    /**
    * The result is complete from a full recognition of audio.
    */
    SR_RECOGNIZER_RESULT_TYPE_COMPLETE,
    /**
    * No results at this time.
    */
    SR_RECOGNIZER_RESULT_TYPE_NONE,
} AlsRecognizerResultType;


class AlsRecognizer {
public:
    static ALSAPI_EXPORT ALS_RETCODE  Create(AlsRecognizer **recogizer);
    virtual ALS_RETCODE Destroy();
    virtual ~AlsRecognizer() {};

    virtual ALS_RETCODE LoadGrammar(AlsGrammar*grammar, const char* rule) = 0;
    virtual ALS_RETCODE LoadRescoreGrammar(AlsRescoreGrammar *rescore_grammar) = 0;
    virtual ALS_RETCODE LoadWordClassSubGrammarFile(const char *word_class_name, const char *lm_file, const char *grammar_type) = 0;
    virtual ALS_RETCODE LoadWordClassSubGrammarImage(AlsWordClassGrammarImage *word_class_grammars, int num_grammar) = 0;
    virtual ALS_RETCODE UnLoadWordClassSubGrammar(const char **word_class_name, int num_grammar) = 0;
    virtual ALS_RETCODE UnLoadGrammar() = 0;
    virtual ALS_RETCODE UnLoadRescoreGrammar() = 0;
    virtual ALS_RETCODE StartUtterance() = 0;
    virtual ALS_RETCODE PutSpeech(AlsAudioSegment*speech_segment) = 0;
    virtual ALS_RETCODE Advance(AlsRecognizerStatus* status) = 0;
    virtual ALS_RETCODE GetResult(AlsRecognizerResult **) = 0;
    virtual ALS_RETCODE EndUttrerance() = 0;
};

#endif
