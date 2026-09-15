// Independently relocated object-manager add/remove boundaries, x86_64 150535029.
//
// The collection-add boundary 0xf7d320 has exactly one live caller: the
// registration entry 0xf87a60, a virtual method that pre-processes the object
// and then adds it. The entity's creation context (the spawn routine that
// instantiated it) is what names the owner of a deferred spawn, for example the
// area object a hero ability leaves behind.
//
// Two telemetry tiers live in this one header:
//   production  - the boundaries and the fields the adapter consumes: identity,
//                 remove/heal, the collection add, and the creation context that
//                 supplies a parent when the object stores none.
//   diagnostic  - ELITE_LIFECYCLE_DIAGNOSTIC. Extra hooks (registration entry,
//                 component builder, state applier) and provenance fields used
//                 to *find* the production channels. They are evidence tooling,
//                 not data the pipeline reads, and they are excluded from the
//                 frozen production artifact.
namespace elite_lifecycle {
constexpr size_t kCapacity=32768;
struct Event {uint64_t sequence=0,command=0;int32_t tick=0,key=0,side=-1,kind=0,operation=0,parent=0,data_id=0,birth=-1,command_side=-1,command_deck=-1,amount=0,pre_hp=0,post_hp=0,source_candidate=0;
  uint64_t context_caller=0;int32_t context_key=0;
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
  uint64_t caller_rva=0,factory_rva=0,builder_caller=0,apply_caller=0;int32_t spawn_arg2=0,spawn_arg3=0,parent_slot=0,parent_context=0,owner_alt=0,owner_alt_slot=0,builder_type=-1,effect_source=0,effect_source_side=-1,object_state=-1,apply_flag=0,context_owner=0,context_owner_side=-1,context_a2=0,context_a3=0,probe_count=0,probe_slot[3]={0,0,0},probe_key[3]={0,0,0},probe_side[3]={-1,-1,-1};
#endif
};
struct Slot {std::atomic<uint64_t> committed{0};Event event;};
inline std::array<Slot,kCapacity> ring;
inline std::atomic<uint64_t> sequence{0},rejected{0},null_id_removals{0};
inline uintptr_t module_base=0;
using Spawn=void(*)(void*,void*,int32_t,int32_t);
using Remove=void(*)(void*,void*);
using Heal=void(*)(void*,int32_t,int32_t,void*,int32_t,int32_t,int32_t);
using Context=void(*)(void*,int32_t,int32_t);
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
using Build=void(*)(void*,int32_t);
using Apply=void(*)(void*,int32_t);
inline Spawn original_register=nullptr;
inline Build original_builder=nullptr;
inline Apply original_apply=nullptr;
#endif
inline Spawn original_spawn=nullptr;
inline Context original_context=nullptr;
inline Remove original_remove=nullptr;
inline Heal original_heal=nullptr;
inline bool installed=false;
inline uint64_t mark(){return sequence.load();}
inline bool attack_effect_since(uint64_t after,int32_t owner){
  const uint64_t last=sequence.load();if(last-after>=kCapacity)return false;
  for(uint64_t n=after+1;n<=last;++n){const auto& slot=ring[(n-1)%kCapacity];if(slot.committed.load()!=n)continue;const auto& e=slot.event;
    if((e.operation==1 && e.key>=4000000 && e.key<5000000 && e.parent==owner) ||
       (e.operation==3 && e.amount>0 && e.source_candidate==owner))return true;
  }
  return false;
}

#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
// A second candidate owner is reported only when the pointer resolves to a live
// object of this world, in the same generation range, on the same side. A
// wrong slot therefore yields nothing instead of a plausible-looking parent.
inline bool owner_key_at(const void* pointer,size_t offset,int32_t self,int32_t side,int32_t& result){
  thread_local SafeMemoryReader memory;uint64_t candidate=0,candidate_world=0;int32_t key=0,owner_side=0;
  const uintptr_t object=reinterpret_cast<uintptr_t>(pointer);
  if(!memory.read(object+offset,&candidate)||!candidate)return false;
  if(!memory.read(candidate+0x10,&candidate_world)||candidate_world!=elite_telemetry::world)return false;
  if(!memory.read(candidate+8,&key)||key<3000000||key>=6000000||key==self)return false;
  if(!memory.read(candidate+0x78,&owner_side)||side<0||owner_side!=side)return false;
  result=key;return true;
}
#endif

inline bool fact(void* manager,void* pointer,Event& e){
  if(!elite_telemetry::enabled||reinterpret_cast<uintptr_t>(manager)!=elite_telemetry::world||!pointer)return false;
  thread_local SafeMemoryReader memory;const uintptr_t object=reinterpret_cast<uintptr_t>(pointer);uint64_t world=0,data=0;
  if(!memory.read(object+0x10,&world)||world!=elite_telemetry::world||!memory.read(object+8,&e.key)||e.key<3000000||e.key>=6000000||
     !memory.read(object+0x78,&e.side)||!memory.read(object+0x30,&e.kind)||!memory.read(elite_telemetry::battle+0x60,&e.tick)||
     !memory.read(object+0x48,&data)||!data||!memory.read(data+0x40,&e.data_id)){rejected.fetch_add(1);return false;}
  if(e.key>=5000000)memory.read(object+0xa8,&e.birth);
  uint64_t parent=0,parent_world=0;int32_t key=0;
  const size_t offset=e.key<4000000?0x130:e.key<5000000?0x100:0;
  if(offset && memory.read(object+offset,&parent)&&parent&&memory.read(parent+0x10,&parent_world)&&parent_world==world&&
     memory.read(parent+8,&key)&&key>=3000000&&key<6000000)e.parent=key;
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
  if(e.parent)e.parent_slot=static_cast<int32_t>(offset),e.parent_context=1;
#endif
  parent=elite_origin::action_owner;
  if(!e.parent && parent && parent!=object && memory.read(parent+0x10,&parent_world)&&parent_world==world&&
     memory.read(parent+8,&key)&&key>=3000000&&key<6000000)e.parent=key;
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
  if(e.parent && !e.parent_context)e.parent_context=2;
  memory.read(object+0x11c,&e.object_state);
  const size_t alt_offset=e.key<4000000?0x100:e.key<5000000?0x130:0;
  if(alt_offset && owner_key_at(pointer,alt_offset,e.key,e.side,e.owner_alt))e.owner_alt_slot=static_cast<int32_t>(alt_offset);
#endif
  e.command=elite_origin::command;e.command_side=elite_origin::side;e.command_deck=elite_origin::deck;
  return true;
}
inline void publish(Event e){
  e.sequence=sequence.fetch_add(1)+1;auto& s=ring[(e.sequence-1)%kCapacity];s.committed.store(0,std::memory_order_release);
  s.event=e;s.committed.store(e.sequence,std::memory_order_release);
}
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
// Which 8-byte slots of an entity hold a live object reference of this world?
// The answer is read from memory, validated (world pointer, generation range,
// side), and published as a calibration for the whole entity class: for a
// projectile the known owner slot must reappear, so a slot found on an object
// whose lineage is otherwise absent is evidence rather than a guess.
inline int32_t owner_probes(const void* pointer,int32_t self,int32_t* slot,int32_t* key,int32_t* side,int32_t capacity){
  thread_local SafeMemoryReader memory;const uintptr_t object=reinterpret_cast<uintptr_t>(pointer);
  int32_t found=0;
  for(size_t offset=0;offset<=0x160&&found<capacity;offset+=8){
    if(offset==0x10)continue; // world back-pointer, not an owner
    uint64_t candidate=0,candidate_world=0;int32_t candidate_key=0,candidate_side=-1;
    if(!memory.read(object+offset,&candidate)||!candidate)continue;
    if(!memory.read(candidate+0x10,&candidate_world)||candidate_world!=elite_telemetry::world)continue;
    if(!memory.read(candidate+8,&candidate_key)||candidate_key<3000000||candidate_key>=6000000||candidate_key==self)continue;
    if(!memory.read(candidate+0x78,&candidate_side)||candidate_side<0||candidate_side>1)continue;
    slot[found]=static_cast<int32_t>(offset);key[found]=candidate_key;side[found]=candidate_side;++found;
  }
  return found;
}
// The active-effect entry carries the entity that applied it (entry+0x38), the
// same relation the detail writer already exposes as active_effects[].source_key
// and the one the reference interface tracks as sourceEntityKey. Reading it at
// add time works for objects too short-lived to appear in any snapshot, and the
// candidate is accepted only when it resolves to a live object of this world.
inline bool effect_source_of(const void* pointer,int32_t self,int32_t& key,int32_t& side){
  thread_local SafeMemoryReader memory;const uintptr_t object=reinterpret_cast<uintptr_t>(pointer);
  uint64_t components=0,effect_component=0,effect_owner=0,entries=0;int32_t count=0,capacity=-1,size=-1;
  if(!memory.read(object+0x18,&components)||!components)return false;
  if(!memory.read(object+0x24,&count)||count<4||count>32)return false;
  if(!memory.read(components+24,&effect_component)||!effect_component)return false;
  if(!memory.read(effect_component+8,&effect_owner)||effect_owner!=object)return false;
  if(!memory.read(effect_component+0x18,&entries)||!entries)return false;
  if(!memory.read(effect_component+0x20,&capacity)||!memory.read(effect_component+0x24,&size))return false;
  if(size<0||capacity<size||capacity>256)return false;
  for(int32_t j=0;j<size;++j){
    uint64_t entry=0,source=0,source_world=0;int32_t candidate=0,candidate_side=-1;
    if(!memory.read(entries+static_cast<uint64_t>(j)*8,&entry)||!entry)continue;
    if(!memory.read(entry+0x38,&source)||!source)continue;
    if(!memory.read(source+0x10,&source_world)||source_world!=elite_telemetry::world)continue;
    if(!memory.read(source+8,&candidate)||candidate<3000000||candidate>=6000000||candidate==self)continue;
    if(!memory.read(source+0x78,&candidate_side)||candidate_side<0)continue;
    key=candidate;side=candidate_side;return true;
  }
  return false;
}
#endif
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
// Thread-local causal context for the registration entry. A spawn recorded while
// this scope is active belongs to the factory that opened it, so the deferred
// path is named by its caller rather than by proximity or a later duplicate.
inline thread_local uintptr_t register_factory=0;
inline thread_local int32_t register_depth=0;
struct RegisterScope {
  uintptr_t previous_factory;int32_t previous_depth;
  explicit RegisterScope(uintptr_t factory):previous_factory(register_factory),previous_depth(register_depth){
    register_factory=factory;register_depth=previous_depth+1;
  }
  ~RegisterScope(){register_factory=previous_factory;register_depth=previous_depth;}
};
// The component builder is shared, but its caller identifies which subsystem
// asked for the entity. Recording that caller is the tick/deferred-action
// context, not a guess from the entity set.
inline thread_local uintptr_t builder_caller=0;
inline thread_local int32_t builder_type=-1;
struct BuilderScope {
  uintptr_t previous_caller;int32_t previous_type;
  BuilderScope(uintptr_t caller,int32_t type):previous_caller(builder_caller),previous_type(builder_type){
    builder_caller=caller;builder_type=type;
  }
  ~BuilderScope(){builder_caller=previous_caller;builder_type=previous_type;}
};
inline void builder_hook(void* object,int32_t type){
  const uintptr_t caller=reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  BuilderScope scope(module_base&&caller>module_base?caller-module_base:0,type);
  original_builder(object,type);
}
// The state-applier wrapper is reached from only a handful of sites; its own
// caller is what distinguishes a spell from an ability landing or a death
// response, so it is recorded as the creation context alongside the flag.
inline thread_local uintptr_t apply_caller=0;
inline thread_local int32_t apply_flag=0;
struct ApplyScope {
  uintptr_t previous_caller;int32_t previous_flag;
  ApplyScope(uintptr_t caller,int32_t flag):previous_caller(apply_caller),previous_flag(apply_flag){
    apply_caller=caller;apply_flag=flag;
  }
  ~ApplyScope(){apply_caller=previous_caller;apply_flag=previous_flag;}
};
#endif
// The spawn routine that drives the state-applier receives the context entity
// that owns the new effect. Its own +0x100 slot is the same owner relation the
// projectile path already exposes, so the owner of a parentless effect is read
// from the context entity instead of being inferred from the entity set.
inline thread_local uintptr_t context_caller=0,context_object=0;
inline thread_local int32_t context_a2=0,context_a3=0,context_key=0;
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
inline thread_local int32_t context_owner=0,context_owner_side=-1;
#endif
struct ContextScope {
  uintptr_t previous_caller,previous_object;int32_t previous_a2,previous_a3,previous_key;
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
  int32_t previous_owner,previous_side;
#endif
  ContextScope(uintptr_t caller,uintptr_t object,int32_t a2,int32_t a3)
    :previous_caller(context_caller),previous_object(context_object),previous_a2(context_a2),previous_a3(context_a3),
     previous_key(context_key)
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
    ,previous_owner(context_owner),previous_side(context_owner_side)
#endif
  {
    context_caller=caller;context_object=object;context_a2=a2;context_a3=a3;context_key=0;
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
    context_owner=0;context_owner_side=-1;
#endif
    thread_local SafeMemoryReader memory;uint64_t self_world=0;int32_t key=0;
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
    uint64_t owner=0,owner_world=0;int32_t side=-1;
#endif
    if(!object)return;
    if(memory.read(object+0x10,&self_world)&&self_world==elite_telemetry::world&&
       memory.read(object+8,&key)&&key>=3000000&&key<6000000)context_key=key;
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
    if(memory.read(object+0x100,&owner)&&owner&&memory.read(owner+0x10,&owner_world)&&owner_world==elite_telemetry::world&&
       memory.read(owner+8,&key)&&key>=3000000&&key<6000000&&memory.read(owner+0x78,&side)&&side>=0&&side<=1){
      context_owner=key;context_owner_side=side;
    }
#endif
  }
  ~ContextScope(){
    context_caller=previous_caller;context_object=previous_object;context_a2=previous_a2;context_a3=previous_a3;
    context_key=previous_key;
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
    context_owner=previous_owner;context_owner_side=previous_side;
#endif
  }
};
inline void context_hook(void* object,int32_t a2,int32_t a3){
  const uintptr_t caller=reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  ContextScope scope(module_base&&caller>module_base?caller-module_base:0,reinterpret_cast<uintptr_t>(object),a2,a3);
  original_context(object,a2,a3);
}
// The spell/effect creator receives the entity that owns the effect it builds.
// Its arguments are probed as native pointers and only accepted when they
// resolve to a live entity of this world, so a deferred effect that stores no
// parent of its own still gets the creation context the adapter consumes.
inline thread_local int32_t creator_arg_key[4]={0,0,0,0};
inline thread_local int32_t creator_arg_side[4]={-1,-1,-1,-1};
inline thread_local int32_t creator_owner_key=0,creator_owner_side=-1;
using Creator=void(*)(void*,void*,void*,void*);
inline Creator original_creator=nullptr;
inline int32_t entity_side_at(uintptr_t pointer){
  if(!pointer)return -3;
  thread_local SafeMemoryReader memory;uint64_t world=0;int32_t key=0,side=-1;
  if(!memory.read(pointer+0x10,&world)||world!=elite_telemetry::world)return -3;
  if(!memory.read(pointer+8,&key)||key<3000000||key>=6000000)return -3;
  if(!memory.read(pointer+0x78,&side)||side<0||side>1)return -3;
  return side;
}
struct CreatorScope {
  uintptr_t previous_caller;int32_t previous_key,previous_a2,previous_a3;
  int32_t previous_args[4],previous_sides[4],previous_owner,previous_owner_side;
  CreatorScope(uintptr_t caller,const uintptr_t* args)
    :previous_caller(context_caller),previous_key(context_key),previous_a2(context_a2),previous_a3(context_a3),
     previous_owner(creator_owner_key),previous_owner_side(creator_owner_side){
    for(int i=0;i<4;++i){previous_args[i]=creator_arg_key[i];previous_sides[i]=creator_arg_side[i];}
    context_caller=caller;context_a2=0;context_a3=0;
    for(int i=0;i<4;++i)creator_arg_key[i]=0,creator_arg_side[i]=-1;
    creator_owner_key=0;creator_owner_side=-1;
    int32_t chosen=0,owner=0,owner_side=-1;
    for(int i=0;i<4;++i){
      const int32_t side=entity_side_at(args[i]);
      if(side<0)continue;
      thread_local SafeMemoryReader memory;int32_t key=0;
      if(!memory.read(args[i]+8,&key))continue;
      creator_arg_key[i]=key;creator_arg_side[i]=side;
      if(chosen==0&&key>=5000000)chosen=key;
      uint64_t candidate=0,candidate_world=0;int32_t candidate_key=0,candidate_side=-1;
      if(memory.read(args[i]+0x100,&candidate)&&candidate&&
         memory.read(candidate+0x10,&candidate_world)&&candidate_world==elite_telemetry::world&&
         memory.read(candidate+8,&candidate_key)&&candidate_key>=5000000&&candidate_key<6000000&&
         memory.read(candidate+0x78,&candidate_side)&&candidate_side>=0&&candidate_side<=1&&owner==0){
        owner=candidate_key;owner_side=candidate_side;
      }
    }
    creator_owner_key=owner;creator_owner_side=owner_side;
    if(owner)context_key=owner;              // a unit owner is the strongest lineage
    else if(chosen)context_key=chosen;       // otherwise a unit argument
    else for(int i=0;i<4;++i)if(creator_arg_key[i]){context_key=creator_arg_key[i];break;}
  }
  ~CreatorScope(){
    context_caller=previous_caller;context_key=previous_key;context_a2=previous_a2;context_a3=previous_a3;
    creator_owner_key=previous_owner;creator_owner_side=previous_owner_side;
    for(int i=0;i<4;++i){creator_arg_key[i]=previous_args[i];creator_arg_side[i]=previous_sides[i];}
  }
};
inline void creator_hook(void* a1,void* a2,void* a3,void* a4){
  const uintptr_t caller=reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  const uintptr_t args[4]={reinterpret_cast<uintptr_t>(a1),reinterpret_cast<uintptr_t>(a2),
                           reinterpret_cast<uintptr_t>(a3),reinterpret_cast<uintptr_t>(a4)};
  CreatorScope scope(module_base&&caller>module_base?caller-module_base:0,args);
  original_creator(a1,a2,a3,a4);
}
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
inline void apply_hook(void* object,int32_t flag){
  const uintptr_t caller=reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  ApplyScope scope(module_base&&caller>module_base?caller-module_base:0,flag);
  original_apply(object,flag);
}
inline void register_hook(void* manager,void* object,int32_t a2,int32_t a3){
  const uintptr_t caller=reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  RegisterScope scope(module_base&&caller>module_base?caller-module_base:0);
  original_register(manager,object,a2,a3);
}
#endif
inline void spawn_hook(void* manager,void* object,int32_t a2,int32_t a3){
  original_spawn(manager,object,a2,a3);Event e;e.operation=1;
  e.context_caller=context_caller;e.context_key=context_key;
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
  const uintptr_t caller=reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  e.spawn_arg2=a2;e.spawn_arg3=a3;
  if(module_base && caller>module_base)e.caller_rva=caller-module_base;
  e.factory_rva=register_factory;e.builder_caller=builder_caller;e.builder_type=builder_type;
  e.apply_caller=apply_caller;e.apply_flag=apply_flag;
  e.context_owner=context_owner;e.context_owner_side=context_owner_side;e.context_a2=context_a2;e.context_a3=context_a3;
#endif
  if(fact(manager,object,e)){
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
    effect_source_of(object,e.key,e.effect_source,e.effect_source_side);
    if(object)e.probe_count=owner_probes(object,e.key,e.probe_slot,e.probe_key,e.probe_side,3);
#endif
    publish(e);
  }
}
inline void remove_hook(void* manager,void* object){
  // The removed entry's key is read before the original method retires it.
  thread_local SafeMemoryReader memory;int32_t key=0;
  if(object && memory.read(reinterpret_cast<uintptr_t>(object)+8,&key)&&key==0){null_id_removals.fetch_add(1);return;} // relocated original null-ID branch
  Event e;e.operation=2;
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
  const uintptr_t caller=reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  if(module_base && caller>module_base)e.caller_rva=caller-module_base;
#endif
  const bool valid=fact(manager,object,e);
  original_remove(manager,object);if(valid)publish(e);
}
inline void heal_hook(void* component,int32_t amount,int32_t a2,void* a3,int32_t a4,int32_t a5,int32_t a6){
  thread_local SafeMemoryReader memory;uint64_t target=0;int32_t shield_before=0,shield_after=0;Event e;e.operation=3;
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
  const uintptr_t caller=reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  if(module_base && caller>module_base)e.caller_rva=caller-module_base;
#endif
  const uintptr_t comp=reinterpret_cast<uintptr_t>(component);
  const bool valid=component&&memory.read(comp+8,&target)&&target&&
    fact(reinterpret_cast<void*>(elite_telemetry::world),reinterpret_cast<void*>(target),e)&&
    memory.read(comp+0x10,&e.pre_hp)&&memory.read(comp+0x28,&shield_before);
  original_heal(component,amount,a2,a3,a4,a5,a6);
  if(!valid)return;
  if(!memory.read(comp+0x10,&e.post_hp)||!memory.read(comp+0x28,&shield_after)){rejected.fetch_add(1);return;}
  e.amount=e.post_hp-e.pre_hp+shield_after-shield_before;
  if(e.amount<=0)return;
  uint64_t source_world=0;int32_t source_key=0;
  if(elite_origin::action_owner&&memory.read(elite_origin::action_owner+0x10,&source_world)&&source_world==elite_telemetry::world&&
     memory.read(elite_origin::action_owner+8,&source_key)&&source_key>=3000000&&source_key<6000000)e.source_candidate=source_key;
  publish(e);
}
inline bool install(uintptr_t base,std::string& error){
  if(installed)return true;
  static const unsigned char spawn[]={0x55,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x50,0x48,0x85,0xf6};
  static const unsigned char remove[]={0x55,0x41,0x57,0x41,0x56,0x53,0x50,0x8b,0x46,0x08,0x85,0xc0,0x74,0x5b};
  static const unsigned char heal[]={0x55,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x48,0x83,0xec,0x18};
  static const unsigned char context[]={0x55,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x48,0x81,0xec,0x98,0x00,0x00,0x00};
  static const unsigned char creator[]={0x55,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x48,0x81,0xec,0xb8,0x00,0x00,0x00};
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
  static const unsigned char registration[]={0x55,0x41,0x57,0x41,0x56,0x41,0x54,0x53,0x48,0x85,0xf6,0x0f,0x84,0xca,0x00,0x00,0x00};
  static const unsigned char builder[]={0x55,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x48,0x83,0xec,0x48};
  static const unsigned char apply[]={0x41,0x56,0x53,0x50,0x89,0xf3,0x49,0x89,0xfe,0x85,0xf6,0x7e,0x17,0x41,0x83,0xbe,0x1c,0x01,0x00,0x00,0x08};
#endif
  // verify = fingerprint bytes, copied = whole instructions moved into the
  // trampoline, resume = offset where the original body continues. remove
  // deliberately copies 12 while resuming at 14: the relocated null-ID branch
  // is re-implemented in the hook. patch = bytes overwritten in the original.
  struct Spec{uintptr_t rva;const unsigned char* bytes;size_t verify,copied,resume,patch;uintptr_t hook;void** original;};
  Spec specs[]={{0xf7d320,spawn,14,14,14,14,reinterpret_cast<uintptr_t>(&spawn_hook),reinterpret_cast<void**>(&original_spawn)},
                {0xf7d520,remove,14,12,14,14,reinterpret_cast<uintptr_t>(&remove_hook),reinterpret_cast<void**>(&original_remove)},
                {0xfbc350,heal,14,14,14,14,reinterpret_cast<uintptr_t>(&heal_hook),reinterpret_cast<void**>(&original_heal)},
                {0xf82790,context,17,17,17,14,reinterpret_cast<uintptr_t>(&context_hook),reinterpret_cast<void**>(&original_context)},
                {0xf6bcd0,creator,17,17,17,14,reinterpret_cast<uintptr_t>(&creator_hook),reinterpret_cast<void**>(&original_creator)},
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
                {0xf87a60,registration,17,17,17,17,reinterpret_cast<uintptr_t>(&register_hook),reinterpret_cast<void**>(&original_register)},
                {0xf704c0,builder,14,14,14,14,reinterpret_cast<uintptr_t>(&builder_hook),reinterpret_cast<void**>(&original_builder)},
                {0xf742c0,apply,21,21,21,14,reinterpret_cast<uintptr_t>(&apply_hook),reinterpret_cast<void**>(&original_apply)},
#endif
                };
  SafeMemoryReader memory;unsigned char observed[24];
  for(const auto& s:specs)if(!memory.read_bytes(base+s.rva,observed,s.verify)||std::memcmp(observed,s.bytes,s.verify)){
    error="lifecycle hook fingerprint mismatch";return false;
  }
  for(const auto& s:specs){
    auto* trampoline=static_cast<unsigned char*>(mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    if(trampoline==MAP_FAILED){error="lifecycle trampoline allocation failed";return false;}
    std::memcpy(trampoline,s.bytes,s.copied);elite_telemetry::jump(trampoline+s.copied,base+s.rva+s.resume);
    if(mprotect(trampoline,4096,PROT_READ|PROT_EXEC)){error="lifecycle trampoline protection failed";return false;}
    const uintptr_t page=(base+s.rva)&~uintptr_t(4095);
    if(mprotect(reinterpret_cast<void*>(page),4096,PROT_READ|PROT_WRITE|PROT_EXEC)){error="lifecycle hook protection failed";return false;}
    *s.original=trampoline;elite_telemetry::jump(reinterpret_cast<unsigned char*>(base+s.rva),s.hook);
    if(s.patch>14)std::memset(reinterpret_cast<unsigned char*>(base+s.rva)+14,0x90,s.patch-14);
    __builtin___clear_cache(reinterpret_cast<char*>(base+s.rva),reinterpret_cast<char*>(base+s.rva+s.patch));
    if(mprotect(reinterpret_cast<void*>(page),4096,PROT_READ|PROT_EXEC)){error="lifecycle protection restore failed";return false;}
  }
  module_base=base;installed=true;return true;
}
inline bool begin(uintptr_t base,std::string& error){
  if(!install(base,error))return false;sequence.store(0);rejected.store(0);null_id_removals.store(0);elite_origin::next_command.store(0);for(auto& s:ring)s.committed.store(0);return true;
}
inline std::string events_json(uint64_t after){
  const uint64_t last=sequence.load(),oldest=last>=kCapacity?last-kCapacity+1:1;
  std::string result="{\"schema\":\"native-lifecycle.x86-v2\",\"last_sequence\":"+std::to_string(last)+",\"rejected\":"+std::to_string(rejected.load())+",\"null_id_removals\":"+std::to_string(null_id_removals.load())+",\"gap\":";
  result+=(after+1<oldest||after>last)?"true":"false";result+=",\"events\":[";bool first=true;
  for(uint64_t n=std::max(after+1,oldest);n<=last;++n){const auto& s=ring[(n-1)%kCapacity];if(s.committed.load()!=n)continue;const auto& e=s.event;
    if(!first)result+=",";first=false;
    char provenance[64];std::snprintf(provenance,sizeof(provenance),",\"context_caller\":\"0x%llx\",\"context_key\":%d",
      static_cast<unsigned long long>(e.context_caller),e.context_key);
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
    char diagnostic[576];std::snprintf(diagnostic,sizeof(diagnostic),",\"parent_slot\":%d,\"parent_context\":%d,\"owner_alt_key\":%d,\"owner_alt_slot\":%d,\"spawn_arg2\":%d,\"spawn_arg3\":%d,\"caller_rva\":\"0x%llx\",\"factory_rva\":\"0x%llx\",\"builder_caller\":\"0x%llx\",\"builder_type\":%d,\"apply_caller\":\"0x%llx\",\"apply_flag\":%d,\"context_owner\":%d,\"context_owner_side\":%d,\"context_a2\":%d,\"context_a3\":%d,\"effect_source_key\":%d,\"effect_source_side\":%d,\"object_state\":%d,\"probes\":[{\"slot\":%d,\"key\":%d,\"side\":%d},{\"slot\":%d,\"key\":%d,\"side\":%d},{\"slot\":%d,\"key\":%d,\"side\":%d}]",
      e.parent_slot,e.parent_context,e.owner_alt,e.owner_alt_slot,e.spawn_arg2,e.spawn_arg3,static_cast<unsigned long long>(e.caller_rva),static_cast<unsigned long long>(e.factory_rva),static_cast<unsigned long long>(e.builder_caller),e.builder_type,static_cast<unsigned long long>(e.apply_caller),e.apply_flag,e.context_owner,e.context_owner_side,e.context_a2,e.context_a3,e.effect_source,e.effect_source_side,e.object_state,
      e.probe_slot[0],e.probe_key[0],e.probe_side[0],e.probe_slot[1],e.probe_key[1],e.probe_side[1],e.probe_slot[2],e.probe_key[2],e.probe_side[2]);
#endif
    result+="{\"sequence\":"+std::to_string(e.sequence)+",\"tick\":"+std::to_string(e.tick)+",\"key\":"+std::to_string(e.key)+",\"side\":"+std::to_string(e.side)+",\"kind\":"+std::to_string(e.kind)+",\"operation\":"+std::to_string(e.operation)+",\"parent_key\":"+std::to_string(e.parent)+",\"native_data_id\":"+std::to_string(e.data_id)+",\"birth_tick\":"+std::to_string(e.birth)+",\"command_sequence\":"+std::to_string(e.command)+",\"command_side\":"+std::to_string(e.command_side)+",\"command_deck_index\":"+std::to_string(e.command_deck)+",\"heal_amount\":"+std::to_string(e.amount)+",\"pre_hp\":"+std::to_string(e.pre_hp)+",\"post_hp\":"+std::to_string(e.post_hp)+",\"source_candidate_key\":"+std::to_string(e.source_candidate)+provenance
#ifdef ELITE_LIFECYCLE_DIAGNOSTIC
      +diagnostic
#endif
      +"}";
  }
  result+="]}";return result;
}
} // namespace elite_lifecycle
