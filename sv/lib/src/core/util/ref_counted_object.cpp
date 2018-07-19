#include "ref_counted_object.h"
#include <cstdlib>
#include <cstdio>

RefCountedObject::RefCountedObject()
  : ref_counter_(1) {
}

RefCountedObject::~RefCountedObject() {
}

void RefCountedObject::AcquireReference() const {
  //printf("%p ++ %d\n", this, m_nRefCounter + 1);
  ++ref_counter_;
}

void RefCountedObject::ReleaseReference() const {
  //printf("%p -- %d\n", this, m_nRefCounter - 1);
  int rc = --ref_counter_;
  if (rc == 0) {
    Clear();
  }

  if (rc < 0) {
    WrongRef();
  }
}

