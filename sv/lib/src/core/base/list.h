/*
*   intrusive linked list implementation.
*       Jiayu DU [jiayu.djy@alibaba-inc.com] 2012-2016
*
*   inspired by coho project(Warcraft core code-base):
*       https://github.com/webcoyote/coho
*   for more info, please visit:
*       http://www.codeofhonor.com/blog/avoiding-game-crashes-related-to-linked-lists

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

#ifndef LIST_H
#define LIST_H

#include <cstdlib>
#include "base/link.h"

namespace idec {

class List {
 public:
  List() :
    num_elems_(0),
    offset_(0)
  { }

  List(size_t offset) :
    num_elems_(0),
    offset_(offset)
  { }

  ~List() { UnlinkAll(); }

  void SetOffset(size_t offset);

  bool Empty();
  size_t NumElems();

  void *Head();
  void *Tail();
  void *Prev(void *node);
  void *Next(void *node);

  void InsertHead(void *node);
  void InsertTail(void *node);

  void UnlinkNode(void *node);
  void UnlinkAll();

 private:
  Link *GetLinkFromNode(void *node);
  void *GetNodeFromLink(Link *link);

  Link     origin_;
  size_t   offset_;
  size_t   num_elems_;

  // Hide
  List(const List &);
  List &operator= (const List &);
};

} // namespace idec

#endif
