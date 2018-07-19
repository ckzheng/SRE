#ifndef __REFCOUNTEDOBJECT_H
#define __REFCOUNTEDOBJECT_H
/**
 * @class RefCountedObject
 * base class for objects that employ
 * reference counting based garbage collection.
 * Reference-counted objects inhibit construction
 * by copying and assignment.
 */
class RefCountedObject {
 public:
  /**Creates the RefCountedObject. The initial reference count is one.*/
  RefCountedObject();
  explicit RefCountedObject(int rc) : ref_counter_(rc) {}

 public:
  /** Increments the object's reference count. */
  void AcquireReference() const;

  /** Decrements the object's reference count
   * and call the Clear() function when the count reaches zero.
   */
  void ReleaseReference() const;

  /*
   * clear up function when refcount = 0;
   */
  virtual void Clear() const = 0;

  virtual void WrongRef() const = 0;

  /** Returns the reference count.*/
  int  ReferenceCount() const { return ref_counter_; };
 protected:
  /** Destroys the RefCountedObject. */
  virtual ~RefCountedObject();
 private:
  RefCountedObject(const RefCountedObject &);
  RefCountedObject &operator = (const RefCountedObject &);

  mutable int  ref_counter_;
};

#endif
