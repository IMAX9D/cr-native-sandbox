// Thread-local causal context; records the native caller, never nearest units.
namespace elite_origin {
inline std::atomic<uint64_t> next_command{0};
inline thread_local uintptr_t action_owner=0;
inline thread_local uint64_t command=0;
inline thread_local int32_t side=-1,deck=-1;
struct ActionScope {
  uintptr_t previous;
  explicit ActionScope(void* owner):previous(action_owner){action_owner=reinterpret_cast<uintptr_t>(owner);}
  ~ActionScope(){action_owner=previous;}
};
struct CommandScope {
  uint64_t previous,id;int32_t previous_side,previous_deck;
  CommandScope(int32_t owner,int32_t slot,bool active):previous(command),id(active?next_command.fetch_add(1)+1:0),previous_side(side),previous_deck(deck){
    if(active){command=id;side=owner;deck=slot;}
  }
  ~CommandScope(){command=previous;side=previous_side;deck=previous_deck;}
};
}
