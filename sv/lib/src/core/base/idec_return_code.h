#ifndef __IDEC_RETURN_CODE_H
#define __IDEC_RETURN_CODE_H

typedef enum IDEC_ReturnCode_t {
  /**
  * Operation completed successfully.
  */
  IDEC_SUCCESS,

  /**
  * Intermediate stage of operation completed successfully, we wish to indicate
  * that remaining stages of operation may proceed.
  */
  IDEC_CONTINUE_PROCESSING,

  /**
  * Indicates a fatal error.
  */
  IDEC_FATAL_ERROR,

  /**
  * Buffer overflow occurred.
  */
  IDEC_BUFFER_OVERFLOW,

  /**
  * Error typing to open an entity or the operation failed because the entity was not opened.
  */
  IDEC_OPEN_ERROR,

  /**
  * Error trying to open an entity that is already open.
  */
  IDEC_ALREADY_OPEN,

  /**
  * Error typing to close a entity or the operation failed because the entity was not closed.
  */
  IDEC_CLOSE_ERROR,

  /**
  * Error trying to close a entity that was already closed.
  */
  IDEC_ALREADY_CLOSED,

  /**
  * Error trying to read a file.
  */
  IDEC_READ_ERROR,

  /**
  * Error trying to write to a entity.
  */
  IDEC_WRITE_ERROR,

  /**
  * Error trying to flush a entity.
  */
  IDEC_FLUSH_ERROR,

  /**
  * Error trying to seek a entity.
  */
  IDEC_SEEK_ERROR,

  /**
  * Error trying to allocate memory.
  */
  IDEC_OUT_OF_MEMORY,

  /**
  * Specified argument is out of bounds.
  */
  IDEC_ARGUMENT_OUT_OF_BOUNDS,

  /**
  * Failed to locate the specified entity.
  */
  IDEC_NO_MATCH_ERROR,

  /**
  * Passed in argument contains an invalid value. Such as when a NULL pointer
  * is passed in when when an actual value is expected.
  */
  IDEC_INVALID_ARGUMENT,

  /**
  * Indicates that request functionality is not supported.
  */
  IDEC_NOT_SUPPORTED,

  /**
  * Indicates that the object is not in a state such that the operation can
  * be successfully performed.
  */
  IDEC_INVALID_STATE,

  /**
  * Indicates that a thread could not be created.
  */
  IDEC_THREAD_CREATION_ERROR,

  /**
  * Indicates that a resource with the same identifier already exists.
  */
  IDEC_IDENTIFIER_COLLISION,

  /**
  * Indicates that the operation timed out.
  */
  IDEC_TIMED_OUT,

  /**
  * Indicates that the object being retrieved isn't of the expected type.
  * For example, when retrieving an integer from a HashMap we find out the
  * value is actually of type float.
  */
  IDEC_INVALID_RESULT_TYPE,

  /**
  * Indicates that the invoked function has not been implemented.
  */
  IDEC_NOT_IMPLEMENTED,

  /**
  * A connection was forcibly closed by a peer. This normally results from
  * a loss of the connection on the remote socket due to a timeout or a reboot.
  */
  IDEC_CONNECTION_RESET_BY_PEER,

  /**
  * Indicates that a process could not be created.
  */
  IDEC_PROCESS_CREATE_ERROR,

  /**
  * Indicates that an attempt to create a mutex failed because the OS is running out of resources.
  */
  IDEC_MUTEX_CREATION_ERROR,

  /**
  * Indicates a deadlock situation has occurred.
  */
  IDEC_DEADLOCK,

  /**
  * Indicates a grammar compiling error
  */
  IDEC_GRAMMAR_ERROR

} IDEC_RETCODE;

#endif

