#include "base/list.h"

namespace idec {

void List::SetOffset(size_t offset) {
  offset_ = offset;
}

Link *List::GetLinkFromNode(void *node) {
  return (Link *)((size_t)node + offset_);
}

void *List::GetNodeFromLink(Link *link) {
  return (link == &origin_ ? NULL : (void *)((size_t)link - offset_));
}

bool List::Empty() {
  return (Head() == NULL);
}

size_t List::NumElems() {
  return num_elems_;
}

void *List::Head() {
  return GetNodeFromLink(origin_.Next());
}

void *List::Tail() {
  return GetNodeFromLink(origin_.Prev());
}

void *List::Prev(void *node) {
  return GetNodeFromLink(GetLinkFromNode(node)->Prev());
}

void *List::Next(void *node) {
  return GetNodeFromLink(GetLinkFromNode(node)->Next());
}

void List::UnlinkNode(void *node) {
  Link *link = GetLinkFromNode(node);
  if (link->Linked()) {
    link->Unlink();
    num_elems_--;
  }
}

void List::InsertHead(void *node) {
  UnlinkNode(node);
  GetLinkFromNode(node)->InsertAfter(&origin_);
  num_elems_++;
}

void List::InsertTail(void *node) {
  UnlinkNode(node);
  GetLinkFromNode(node)->InsertBefore(&origin_);
  num_elems_++;
}

void List::UnlinkAll() {
  while (!Empty()) {
    UnlinkNode(Head());
  }
}

} // namespace idec