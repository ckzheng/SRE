/*
*   intrusive linked list implementation.
*       Jiayu DU [jiayu.djy@alibaba-inc.com] 2012-2016
*
*   inspired by coho project(Warcraft core code-base): https://github.com/webcoyote/coho
*   for more info, please visit: http://www.codeofhonor.com/blog/avoiding-game-crashes-related-to-linked-lists

*   This module defines a linked - list implementation that uses "embedded"
*   links rather than separately allocated link - nodes as does STL and
*   more or less all other linked - list implementations.
*
*   Why is this cool:
*       1. No additional memory allocations(malloc) required to link
*           an object into a linked list.
*       2. Not necessary to traverse an additional pointer references
*           to get to the object being dereferenced.
*       3. Probably most importantly, when objects get deleted, they
*           automatically unlink themselves from the lists they're
*           linked to, eliminating many common types of bugs.
*/

#ifndef LINK_H
#define LINK_H

namespace idec {

class Link {
 public:
  Link() {
    prev_ = this;
    next_ = this;
  }

  ~Link() { Unlink(); }

  Link *Prev() { return prev_; }
  Link *Next() { return next_; }

  void InsertBefore(Link *ref);
  void InsertAfter(Link *ref);

  int  Linked();
  void Unlink();

 private:
  Link *prev_;
  Link *next_;
  // Hide
  Link(const Link &);
  Link &operator= (const Link &);
};

} // namespace idec

#endif

