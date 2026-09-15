// Independent x86_64 150535029 phase instrumentation. Layout/event design was
// compared with FirstLight CR (Apache-2.0, commit 28d66cc0); all function RVAs
// and instruction boundaries below were reidentified in this owned runtime.
// No gameplay argument, return value or timing is changed.
namespace elite_lifecycle { inline uint64_t mark();inline bool attack_effect_since(uint64_t,int32_t); }
namespace elite_phase {
constexpr size_t kCache=8192;
struct Record {int32_t key=0,start=-1,release=-1,reset=-1,execute=-1,execute_result=0,scale_tick=-1,scale_input=-1,scale_output=-1;uint64_t start_seq=0,release_seq=0,execute_seq=0;};
inline std::array<Record,kCache> records{};
inline uintptr_t module=0;
inline std::atomic<uint64_t> sequence{0},overflow{0};
inline bool installed=false;
inline std::atomic<uint64_t> attack_calls{0},attack_nonzero{0},attack_owner_misses{0};
using Dispatch=void(*)(void*,void*,void*);
using Attack=int32_t(*)(void*,void*,int32_t,void*,int32_t,int32_t,int32_t);
using Target=void(*)(void*,void*,int32_t,int32_t,int32_t);
using Scale=int32_t(*)(void*,int32_t);
inline Dispatch original_dispatch=nullptr;
inline Attack original_attack=nullptr;
inline Target original_target=nullptr;
inline Scale original_scale=nullptr;

inline Record* record_for(uintptr_t object,int32_t& tick){
  if(!elite_telemetry::enabled || !object)return nullptr;
  thread_local SafeMemoryReader memory;
  uint64_t world=0;int32_t key=0;
  if(!memory.read(object+0x10,&world)||world!=elite_telemetry::world||!memory.read(object+8,&key)||
     key<5000000||key>=6000000||!memory.read(elite_telemetry::battle+0x60,&tick))return nullptr;
  const size_t index=static_cast<size_t>(key-5000000);
  if(index>=kCache){overflow.fetch_add(1);return nullptr;}
  Record& r=records[index];r.key=key;return &r;
}

inline uintptr_t component_owner(void* component){
  thread_local SafeMemoryReader memory;uint64_t owner=0;
  if(component)memory.read(reinterpret_cast<uintptr_t>(component)+8,&owner);
  return owner;
}

inline void dispatch_hook(void* owner,void* target,void* action){
  if(!action)return; // relocated original null branch
  const uintptr_t caller=reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  elite_origin::ActionScope scope(owner);
  original_dispatch(owner,target,action);
  if(caller==module+0xfb90d9){
    int32_t tick=-1;if(auto* r=record_for(reinterpret_cast<uintptr_t>(owner),tick)){r->start=tick;r->start_seq=sequence.fetch_add(1)+1;}
  }
}

inline int32_t attack_hook(void* component,void* target,int32_t a2,void* a3,int32_t a4,int32_t a5,int32_t a6){
  const uintptr_t owner=component_owner(component);
  int32_t tick=-1;auto* r=record_for(owner,tick);
  const uint64_t damage_before=elite_telemetry::sequence.load(),life_before=elite_lifecycle::mark();
  elite_origin::ActionScope scope(reinterpret_cast<void*>(owner));
  const int32_t result=original_attack(component,target,a2,a3,a4,a5,a6);
  const auto call=attack_calls.fetch_add(1);
  if(result)attack_nonzero.fetch_add(1);
  if(call<4)std::fprintf(stderr,"ELITE_PHASE_ATTACK component=%p owner=%llx result=%d\n",component,static_cast<unsigned long long>(owner),result);
  if(r){
    r->execute=tick;r->execute_result=result;r->execute_seq=sequence.fetch_add(1)+1;
    bool confirmed=elite_lifecycle::attack_effect_since(life_before,r->key);
    const uint64_t last=elite_telemetry::sequence.load();
    if(last-damage_before<elite_telemetry::kCapacity)for(uint64_t n=damage_before+1;n<=last;++n){
      const auto& slot=elite_telemetry::ring[(n-1)%elite_telemetry::kCapacity];
      if(slot.committed.load()==n && slot.event.attacker==r->key && slot.event.actual>0)confirmed=true;
    }
    if(confirmed){r->release=tick;r->release_seq=sequence.fetch_add(1)+1;}
  }
  return result;
}

inline void target_hook(void* component,void* target,int32_t a2,int32_t a3,int32_t a4){
  const uintptr_t owner=component_owner(component);
  original_target(component,target,a2,a3,a4);
  if(!target){int32_t tick=-1;if(auto* r=record_for(owner,tick))r->reset=tick;}
}

inline int32_t scale_hook(void* component,int32_t input){
  const uintptr_t caller=reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  const int32_t result=original_scale(component,input);
  if(caller==module+0xfb9624){int32_t tick=-1;if(auto* r=record_for(component_owner(component),tick)){
    r->scale_tick=tick;r->scale_input=input;r->scale_output=result;
  }}
  return result;
}

struct Spec {uintptr_t rva;const unsigned char* bytes;size_t length,skip;uintptr_t hook;void** original;};
inline bool install(uintptr_t base,std::string& error){
  if(installed)return true;
  static const unsigned char dispatch[]={0x48,0x85,0xd2,0x74,0x5a,0x41,0x57,0x41,0x56,0x41,0x54,0x53,0x50,0x48,0x89,0xfb};
  static const unsigned char attack[]={0x55,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x48,0x81,0xec,0xb8,0,0,0};
  static const unsigned char target[]={0x55,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x50,0x44,0x89,0xc5};
  static const unsigned char scale[]={0x55,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x50,0x89,0xf3,0xbd,0x64,0,0,0};
  Spec specs[]={
    {0xf7ab70,dispatch,sizeof(dispatch),5,reinterpret_cast<uintptr_t>(&dispatch_hook),reinterpret_cast<void**>(&original_dispatch)},
    {0xfb64d0,attack,sizeof(attack),0,reinterpret_cast<uintptr_t>(&attack_hook),reinterpret_cast<void**>(&original_attack)},
    {0xfb3d20,target,sizeof(target),0,reinterpret_cast<uintptr_t>(&target_hook),reinterpret_cast<void**>(&original_target)},
    {0xfb2b00,scale,sizeof(scale),0,reinterpret_cast<uintptr_t>(&scale_hook),reinterpret_cast<void**>(&original_scale)}
  };
  SafeMemoryReader memory;std::array<unsigned char,32> observed{};
  for(const auto& s:specs)if(!memory.read_bytes(base+s.rva,observed.data(),s.length)||std::memcmp(observed.data(),s.bytes,s.length)){
    error="x86 phase hook fingerprint mismatch";return false;
  }
  static const unsigned char speed_head[]={0x41,0x57,0x41,0x56,0x41,0x54,0x53,0x50,0x48,0x8d,0x05,0x51,0xd9,0xb1,0x00};
  if(!memory.read_bytes(base+0xf74c50,observed.data(),sizeof(speed_head))||std::memcmp(observed.data(),speed_head,sizeof(speed_head))){
    error="effective speed getter fingerprint mismatch";return false;
  }
  module=base;
  for(const auto& s:specs){
    auto* trampoline=static_cast<unsigned char*>(mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    if(trampoline==MAP_FAILED){error="phase trampoline allocation failed; recycle worker";return false;}
    std::memcpy(trampoline,s.bytes+s.skip,s.length-s.skip);
    elite_telemetry::jump(trampoline+s.length-s.skip,base+s.rva+s.length);
    if(mprotect(trampoline,4096,PROT_READ|PROT_EXEC)){error="phase trampoline protection failed";return false;}
    const uintptr_t page=(base+s.rva)&~uintptr_t(4095);
    if(mprotect(reinterpret_cast<void*>(page),4096,PROT_READ|PROT_WRITE|PROT_EXEC)){error="phase patch protection failed";return false;}
    *s.original=trampoline;
    elite_telemetry::jump(reinterpret_cast<unsigned char*>(base+s.rva),s.hook);
    for(size_t i=14;i<s.length;++i)*reinterpret_cast<unsigned char*>(base+s.rva+i)=0x90;
    __builtin___clear_cache(reinterpret_cast<char*>(base+s.rva),reinterpret_cast<char*>(base+s.rva+s.length));
    if(mprotect(reinterpret_cast<void*>(page),4096,PROT_READ|PROT_EXEC)){error="phase code protection restore failed";return false;}
  }
  installed=true;return true;
}

inline bool begin(uintptr_t base,std::string& error){
  if(!install(base,error))return false;
  for(auto& r:records)r=Record{};sequence.store(0);overflow.store(0);attack_calls.store(0);attack_nonzero.store(0);return true;
}

inline std::string snapshot(uintptr_t base,uintptr_t object,int32_t key){
  if(key<5000000||key>=6000000)return "{\"applicable\":false}";
  SafeMemoryReader memory;uint64_t components=0,attack=0,owner=0,vt=0,data=0;
  bool valid=memory.read(object+0x18,&components)&&components&&memory.read(components,&attack)&&attack&&
    memory.read(attack+8,&owner)&&owner==object&&memory.read(attack,&vt)&&vt==base+0x196ece0&&memory.read(object+0x48,&data)&&data;
  if(!valid)return "{\"applicable\":true,\"valid\":false}";
  int32_t stage=-1,timeline=-1,load=-1,deploy=-1,previous=-1,hit=-1,dash=-1;uint64_t target=0;
  valid=memory.read(attack+0x20,&stage)&&memory.read(attack+0x24,&timeline)&&memory.read(attack+0x28,&load)&&
    memory.read(object+0x15c,&deploy)&&memory.read(object+0x160,&previous)&&memory.read(data+0x418,&hit)&&memory.read(data+0x6a0,&dash)&&memory.read(attack+0x10,&target);
  valid=valid&&stage>=-1&&stage<=255&&timeline>=0&&load>=0&&deploy>=0&&previous>=0&&hit>=0&&hit<3600000;
  Record r;const size_t index=static_cast<size_t>(key-5000000);if(index<kCache && records[index].key==key)r=records[index];
  std::string out="{\"applicable\":true,\"valid\":";out+=valid?"true":"false";
  out+=",\"hooks_installed\":";out+=installed?"true":"false";
  out+=",\"overflow\":"+std::to_string(overflow.load());
  out+=",\"attack_calls\":"+std::to_string(attack_calls.load())+",\"attack_nonzero\":"+std::to_string(attack_nonzero.load());
  out+=",\"sequence_index\":"+std::to_string(stage)+",\"timeline_ms\":"+std::to_string(timeline)+",\"load_remaining_ms\":"+std::to_string(load);
  out+=",\"deploy_remaining_ms\":"+std::to_string(deploy)+",\"deploy_previous_ms\":"+std::to_string(previous)+",\"hit_speed_ms\":"+std::to_string(hit)+",\"attack_dash_time_ms\":"+std::to_string(dash);
  out+=",\"target_present\":";out+=target?"true":"false";
  out+=",\"last_start_tick\":"+std::to_string(r.start)+",\"last_release_tick\":"+std::to_string(r.release)+",\"last_reset_tick\":"+std::to_string(r.reset);
  out+=",\"last_execute_tick\":"+std::to_string(r.execute)+",\"last_execute_result\":"+std::to_string(r.execute_result)+",\"execute_sequence\":"+std::to_string(r.execute_seq);
  out+=",\"start_sequence\":"+std::to_string(r.start_seq)+",\"release_sequence\":"+std::to_string(r.release_seq);
  out+=",\"scale_tick\":"+std::to_string(r.scale_tick)+",\"scale_input_ms\":"+std::to_string(r.scale_input)+",\"scale_output_ms\":"+std::to_string(r.scale_output)+"}";
  if(valid && installed){
    uint64_t movement=0,movement_owner=0,movement_vtable=0;int32_t charge=-2;
    if(memory.read(components+8,&movement)&&movement&&memory.read(movement+8,&movement_owner)&&movement_owner==object&&
       memory.read(movement,&movement_vtable)&&movement_vtable==base+0x196ee48&&memory.read(movement+0x1e0,&charge)&&charge>=-1){
      const int32_t speed=reinterpret_cast<int32_t(*)(void*)>(base+0xf74c50)(reinterpret_cast<void*>(object));
      out.pop_back();out+=",\"classic_charge_progress\":"+std::to_string(charge)+",\"effective_movement_speed\":"+std::to_string(speed)+"}";
    }
  }
  return out;
}
} // namespace elite_phase
