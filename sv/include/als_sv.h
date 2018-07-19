#ifndef ALS_SV_H_
#define ALS_SV_H_

#include "als_error.h"
#include <vector>
#include <string>

class AlsSpeakerVerification {
public:
    /*ALSAPI_EXPORT*/ static AlsSpeakerVerification*  Create();
    /*ALSAPI_EXPORT*/ static void  Destroy(AlsSpeakerVerification*);
    /*ALSAPI_EXPORT*/ const char*GetVersion();

    virtual ~AlsSpeakerVerification() {};

    // initialization 
    virtual ALS_RETCODE Init(const char*sys_dir, const char* cfg_file) = 0;
    virtual ALS_RETCODE UnInit() = 0;
    virtual ALS_RETCODE SetStatCoeff(const char flag) = 0;

    virtual void BeginRegister(const char*spk_id) = 0;
    virtual void BeginRegisterUtterance() = 0;
    virtual void Register(char*wave, unsigned int len) = 0;
    virtual void CancelRegisterUtterance() = 0;
    virtual void EndRegisterUtterance() = 0;
    virtual void EndRegister() = 0;
    virtual void CancelRegister() =0;

    virtual ALS_RETCODE BeginLogIn(const char* spk_guid) = 0;
    virtual ALS_RETCODE LogIn(char*wave, unsigned int len) = 0;
    virtual ALS_RETCODE EndLogIn(float*confidence) = 0;

    virtual ALS_RETCODE DeleteUser(const char*spk_id) = 0;
    virtual ALS_RETCODE GetUserList(std::vector<std::string> &user_list) = 0;

};

#endif
