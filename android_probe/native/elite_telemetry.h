// Included after SafeMemoryReader. Exact 150535029 x86_64 replay instrumentation.
// Preserves the native function and all arguments; only observes before/after facts.
namespace elite_phase { inline std::string snapshot(uintptr_t,uintptr_t,int32_t); }
namespace elite_special { inline std::string snapshot(uintptr_t,uintptr_t); }
namespace elite_telemetry {
constexpr uintptr_t kDamageRva=0xFBB7C0;
constexpr size_t kCapacity=32768;
constexpr unsigned char kDamageHead[14]={0x55,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x48,0x83,0xec,0x58};
using DamageFn=int32_t(*)(void*,int32_t,int32_t,int32_t,void*,int32_t,void*,void*,uint64_t,int32_t*,uint64_t,uint64_t);
inline DamageFn original=nullptr;
inline uintptr_t battle=0,world=0,module_base=0;
inline std::array<int32_t,6> tower_keys{};
inline uint64_t epoch=0;
inline std::atomic<uint64_t> sequence{0},rejected{0},unattributed{0};
inline bool enabled=false;
struct Event {uint64_t seq=0,caller_rva=0;int32_t tick=0,source=0,target=0,source_side=-1,target_side=-1,target_kind=0,requested=0,actual=0,pre_hp=0,post_hp=0,pre_shield=0,post_shield=0,reported=0,lethal=0,attacker=0,source_present=0,target_tower=0,logic_state=-1;};
struct Slot {std::atomic<uint64_t> committed{0};Event event;};
inline std::array<Slot,kCapacity> ring;
inline void jump(unsigned char* where,uintptr_t target){where[0]=0xff;where[1]=0x25;std::memset(where+2,0,4);std::memcpy(where+6,&target,8);}
inline int32_t damage_hook(void* component,int32_t requested,int32_t m2,int32_t m3,void* source,int32_t m5,void* a6,void* a7,uint64_t a8,int32_t* amount,uint64_t a10,uint64_t a11){
  SafeMemoryReader memory;Event record;uint64_t target=0,object_world=0;
  const uintptr_t comp=reinterpret_cast<uintptr_t>(component),src=reinterpret_cast<uintptr_t>(source);
  bool capture=enabled && memory.read(comp+8,&target) && target && memory.read(target+0x10,&object_world) && object_world==world &&
    memory.read(battle+0x60,&record.tick) && memory.read(target+8,&record.target) && record.target>=5000000 && record.target<6000000 &&
    memory.read(comp+0x10,&record.pre_hp) && memory.read(comp+0x28,&record.pre_shield);
  if(capture){
    const uintptr_t caller=reinterpret_cast<uintptr_t>(__builtin_return_address(0));
    record.caller_rva=caller>=module_base?caller-module_base:0;
    uint64_t logic=0;if(memory.read(battle+0xa8,&logic)&&logic)memory.read(logic+0x18,&record.logic_state);
    record.target_tower=std::find(tower_keys.begin(),tower_keys.end(),record.target)!=tower_keys.end()?1:0;
    record.source_present=src?1:0;
    memory.read(target+0x78,&record.target_side);memory.read(target+0x30,&record.target_kind);
    uint64_t source_world=0;
    if(src && memory.read(src+0x10,&source_world) && source_world==world){memory.read(src+8,&record.source);memory.read(src+0x78,&record.source_side);}
    if(record.source<3000000 || record.source>=6000000)record.source=0;
    record.attacker=record.source;
    if(record.source>=4000000 && record.source<5000000){
      uint64_t owner=0,owner_world=0;int32_t owner_key=0,owner_side=-1;
      if(memory.read(src+0x100,&owner)&&owner&&memory.read(owner+0x10,&owner_world)&&owner_world==world&&
         memory.read(owner+8,&owner_key)&&owner_key>=5000000&&owner_key<6000000&&memory.read(owner+0x78,&owner_side)&&owner_side==record.source_side)record.attacker=owner_key;
    }
  }
  int32_t result;
  {
    // On-damaged and death spawns are owned by this health component's
    // character, not by the outer attack dispatch. Nested native action
    // dispatches still override this scope with their own explicit owner.
    elite_origin::ActionScope damage_response(capture?reinterpret_cast<void*>(target):nullptr);
    result=original(component,requested,m2,m3,source,m5,a6,a7,a8,amount,a10,a11);
  }
  if(!capture)return result;
  if(!memory.read(comp+0x10,&record.post_hp) || !memory.read(comp+0x28,&record.post_shield)){rejected.fetch_add(1);return result;}
  record.requested=requested;record.lethal=result&1;
  record.actual=(record.pre_hp-record.post_hp)+(record.pre_shield-record.post_shield);
  record.reported=record.actual;
  if(amount)memory.read(reinterpret_cast<uintptr_t>(amount),&record.reported);
  if(record.actual==0 && !record.lethal && (record.reported==0 || (record.pre_hp==0 && record.post_hp==0 && record.pre_shield==0 && record.post_shield==0)))return result;
  const bool survival=record.post_hp==1 && record.pre_hp>1 && !record.lethal && record.pre_shield==record.post_shield && record.reported==record.pre_hp && requested>=record.pre_hp;
  // The same native survive-at-one gate reports the capped incoming point on
  // subsequent hits, while actual HP/shield remain unchanged. Retain a zero
  // effective-damage receipt, not a fake lost event or a point of damage.
  const bool survival_floor=record.pre_hp==1 && record.post_hp==1 && !record.lethal && record.pre_shield==record.post_shield && record.reported==1 && requested>=1;
  if(record.actual<0 || (record.actual==0&&!survival_floor) || record.pre_hp<0 || record.post_hp<0 || record.pre_shield<0 || record.post_shield<0 || (record.actual!=record.reported && !survival && !survival_floor)){
    if(rejected.fetch_add(1)<16)std::fprintf(stderr,"ELITE_DAMAGE_REJECT tick=%d target=%d source=%d requested=%d actual=%d reported=%d hp=%d/%d shield=%d/%d lethal=%d\n",record.tick,record.target,record.source,requested,record.actual,record.reported,record.pre_hp,record.post_hp,record.pre_shield,record.post_shield,record.lethal);
    return result;
  }
  if(record.source==0)unattributed.fetch_add(1);
  const uint64_t seq=sequence.fetch_add(1)+1;record.seq=seq;Slot& slot=ring[(seq-1)%kCapacity];
  slot.committed.store(0,std::memory_order_release);slot.event=record;slot.committed.store(seq,std::memory_order_release);
  return result;
}
inline bool install(uintptr_t base,std::string& error){
  if(original)return true;
  SafeMemoryReader memory;unsigned char bytes[14];
  if(!memory.read_bytes(base+kDamageRva,bytes,sizeof(bytes)) || std::memcmp(bytes,kDamageHead,sizeof(bytes))!=0){error="damage function fingerprint mismatch";return false;}
  auto* trampoline=static_cast<unsigned char*>(mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
  if(trampoline==MAP_FAILED){error="trampoline allocation failed";return false;}
  std::memcpy(trampoline,bytes,sizeof(bytes));jump(trampoline+14,base+kDamageRva+14);
  if(mprotect(trampoline,4096,PROT_READ|PROT_EXEC)!=0){munmap(trampoline,4096);error="trampoline executable protection failed";return false;}
  const uintptr_t page=(base+kDamageRva)&~uintptr_t(4095);
  if(mprotect(reinterpret_cast<void*>(page),4096,PROT_READ|PROT_WRITE|PROT_EXEC)!=0){munmap(trampoline,4096);error="native code page protection failed";return false;}
  original=reinterpret_cast<DamageFn>(trampoline);
  jump(reinterpret_cast<unsigned char*>(base+kDamageRva),reinterpret_cast<uintptr_t>(&damage_hook));
  __builtin___clear_cache(reinterpret_cast<char*>(base+kDamageRva),reinterpret_cast<char*>(base+kDamageRva+14));
  if(mprotect(reinterpret_cast<void*>(page),4096,PROT_READ|PROT_EXEC)!=0){error="native code page restore failed";return false;}
  return true;
}
inline bool begin(uintptr_t base,std::string& error){
  module_base=base;
  enabled=false;SafeMemoryReader memory;uint64_t manager=0,state=0,logic=0,registry=0,collection=0,objects=0,first=0;
  if(!memory.read(base+0x1a85978,&manager)||!memory.read(manager+0x20,&state)||!memory.read(state+0x90,&battle)||
     !memory.read(battle+0xa8,&logic)||!memory.read(logic+8,&registry)||!memory.read(registry+0x40,&collection)||
     !memory.read(collection+8,&objects)||!memory.read(objects,&first)||!memory.read(first+0x10,&world)||!world){error="battle unavailable for telemetry epoch";return false;}
  if(!install(base,error))return false;
  capture_episode_state(memory,base,battle);
  if(g_episode.tower_count!=6){error="six crown tower identities required at capture start";return false;}
  for(size_t i=0;i<6;++i){
    if(!memory.read(g_episode.towers[i].id+8,&tower_keys[i]) || tower_keys[i]<5000000 || tower_keys[i]>=6000000){error="invalid tower generation identity";return false;}
  }
  ++epoch;sequence.store(0);rejected.store(0);unattributed.store(0);
  for(auto& slot:ring)slot.committed.store(0);
  enabled=true;return true;
}
inline std::string events_json(uint64_t after){
  const uint64_t last=sequence.load(std::memory_order_acquire),oldest=last>=kCapacity?last-kCapacity+1:1;
  std::string out="{\"schema\":\"native-elite-damage.x86-v1\",\"enabled\":";
  out+=enabled?"true":"false";out+=",\"epoch\":"+std::to_string(epoch)+",\"last_sequence\":"+std::to_string(last);
  out+=",\"rejected\":"+std::to_string(rejected.load())+",\"unattributed\":"+std::to_string(unattributed.load());
  out+=",\"gap\":";out+=(after+1<oldest || after>last)?"true":"false";out+=",\"events\":[";
  bool first=true;
  for(uint64_t seq=std::max(after+1,oldest);seq<=last;++seq){
    const Slot& slot=ring[(seq-1)%kCapacity];if(slot.committed.load(std::memory_order_acquire)!=seq)continue;const Event e=slot.event;
    char row[1024];std::snprintf(row,sizeof(row),"%s{\"sequence\":%llu,\"tick\":%d,\"source_key\":%d,\"attacker_key\":%d,\"source_pointer_present\":%d,\"target_key\":%d,\"source_side\":%d,\"target_side\":%d,\"target_kind\":%d,\"target_is_tower\":%d,\"requested\":%d,\"effective_damage\":%d,\"reported\":%d,\"pre_hp\":%d,\"post_hp\":%d,\"pre_shield\":%d,\"post_shield\":%d,\"lethal\":%d}",first?"":",",static_cast<unsigned long long>(e.seq),e.tick,e.source,e.attacker,e.source_present,e.target,e.source_side,e.target_side,e.target_kind,e.target_tower,e.requested,e.actual,e.reported,e.pre_hp,e.post_hp,e.pre_shield,e.post_shield,e.lethal);
    first=false;out+=row;out.pop_back();out+=",\"caller_rva\":"+std::to_string(e.caller_rva)+",\"native_logic_state\":"+std::to_string(e.logic_state)+"}";
  }
  out+="]}";return out;
}
inline std::string data_name(SafeMemoryReader& memory,uint64_t object){
  int32_t length=0;uint64_t chars=0;
  if(!object || !memory.read(object+0x2c,&length) || length<=0 || length>128)return "";
  if(length<8)chars=object+0x30;else if(!memory.read(object+0x30,&chars))return "";
  char buffer[129]={};if(!memory.read_bytes(chars,buffer,static_cast<size_t>(length)))return "";
  std::string result;
  for(int32_t i=0;i<length;++i){unsigned char ch=static_cast<unsigned char>(buffer[i]);if(ch<32 || ch>=127)return "";if(ch=='"'||ch=='\\')result+='\\';result+=static_cast<char>(ch);}
  return result;
}
inline std::string details_json(uintptr_t base){
  SafeMemoryReader memory;uint64_t manager=0,state=0,current_battle=0,logic=0,registry=0,collection=0,data=0;int32_t count=0,tick=-1;
  if(!memory.read(base+0x1a85978,&manager)||!memory.read(manager+0x20,&state)||!memory.read(state+0x90,&current_battle)||
     !memory.read(current_battle+0x60,&tick)||!memory.read(current_battle+0xa8,&logic)||!memory.read(logic+8,&registry)||
     !memory.read(registry+0x40,&collection)||!memory.read(collection+8,&data)||!memory.read(collection+0x14,&count)||count<0||count>2048)return "{\"valid\":false}";
  std::string result="{\"schema\":\"native-elite-details.x86-v1\",\"tick\":"+std::to_string(tick)+",\"objects\":[";bool first=true;
  for(int32_t i=0;i<count;++i){
    uint64_t object=0,native_data=0,components=0;int32_t key=0,kind=0,side=-1,created=-1,native_id=0,component_count=0;
    if(!memory.read(data+static_cast<uint64_t>(i)*8,&object)||!object||!memory.read(object+8,&key)||key<3000000||key>=6000000)continue;
    memory.read(object+0x30,&kind);memory.read(object+0x78,&side);memory.read(object+0x48,&native_data);memory.read(native_data+0x40,&native_id);
    memory.read(object+0xa8,&created);memory.read(object+0x18,&components);memory.read(object+0x24,&component_count);
    const std::string name=data_name(memory,native_data);
    char row[640];std::snprintf(row,sizeof(row),"%s{\"key\":%d,\"kind\":%d,\"side\":%d,\"native_data_id\":%d,\"native_name\":\"%s\",\"created_tick_candidate\":%d,\"component_count\":%d",first?"":",",key,kind,side,native_id,name.c_str(),created,component_count);first=false;result+=row;
    uint64_t source_object=0,source_world=0,own_world=0;int32_t source_key=0,source_side=-1;
    if(key>=4000000 && key<5000000 && memory.read(object+0x100,&source_object) && source_object && memory.read(source_object+0x10,&source_world) &&
      memory.read(object+0x10,&own_world) && own_world && own_world==source_world){memory.read(source_object+8,&source_key);memory.read(source_object+0x78,&source_side);}
    if(source_key<3000000 || source_key>=6000000)source_key=0;
    result+=",\"source_key\":"+std::to_string(source_key)+",\"source_side\":"+std::to_string(source_side);
    uint64_t hp_component=0,owner=0;int32_t shield=-1,maximum_shield=-1;
    if(component_count>=3 && component_count<=32 && memory.read(components+16,&hp_component) && hp_component && memory.read(hp_component+8,&owner) && owner==object){memory.read(hp_component+0x28,&shield);memory.read(hp_component+0x2c,&maximum_shield);}
    result+=",\"builtin_shield\":"+std::to_string(shield)+",\"max_builtin_shield\":"+std::to_string(maximum_shield);
    uint64_t effect_component=0,effect_owner=0,entries=0;int32_t capacity=-1,size=-1,invisible=-1;
    bool effects_valid=component_count>=4 && component_count<=32 && memory.read(components+24,&effect_component) && effect_component &&
      memory.read(effect_component+8,&effect_owner)&&effect_owner==object && memory.read(effect_component+0x18,&entries)&&
      memory.read(effect_component+0x20,&capacity)&&memory.read(effect_component+0x24,&size)&&size>=0&&capacity>=size&&capacity<=256;
    result+=",\"effects_header_valid\":";result+=effects_valid?"true":"false";result+=",\"active_effects\":[";
    if(effects_valid){
      memory.read(effect_component+0x34,&invisible);
      for(int32_t j=0;j<size;++j){
        uint64_t entry=0,asset=0,source=0;int32_t remaining=-2,effect_id=0,source_key=0;
        if(!memory.read(entries+static_cast<uint64_t>(j)*8,&entry)||!entry||!memory.read(entry+0x18,&asset)){effects_valid=false;break;}
        memory.read(entry+8,&remaining);memory.read(asset+0x40,&effect_id);memory.read(entry+0x38,&source);if(source)memory.read(source+8,&source_key);
        const std::string effect_name=data_name(memory,asset);
        if(effect_name.empty()||remaining< -1){effects_valid=false;break;}
        char effect[384];std::snprintf(effect,sizeof(effect),"%s{\"native_data_id\":%d,\"name\":\"%s\",\"remaining_ms\":%d,\"source_key\":%d}",j?",":"",effect_id,effect_name.c_str(),remaining,source_key);result+=effect;
      }
    }
    result+="],\"effects_valid\":";result+=effects_valid?"true":"false";result+=",\"invisible_count_candidate\":"+std::to_string(invisible);
    // Keep bounded, owner-validated original component words. Their presence
    // is not a claim that every attack/deployment phase has been decoded.
    // Bounds match structures already read by the full native observer.
    result+=",\"component_snapshots\":[";
    const size_t lengths[4]={0x68,0x220,0x30,0x38};
    const char* roles[4]={"attack","movement","health","effects"};
    bool first_component=true;
    for(int slot_index=0;slot_index<4;++slot_index){
      uint64_t component=0,component_owner=0;
      bool valid=component_count>slot_index && component_count<=32 && components &&
        memory.read(components+static_cast<uint64_t>(slot_index)*8,&component) && component &&
        memory.read(component+8,&component_owner) && component_owner==object;
      std::array<unsigned char,0x220> bytes{};
      valid=valid && memory.read_bytes(component,bytes.data(),lengths[slot_index]);
      if(!first_component)result+=",";first_component=false;
      result+="{\"role\":\"";result+=roles[slot_index];result+="\",\"valid\":";result+=valid?"true":"false";
      result+=",\"i32_words\":[";
      if(valid){
        for(size_t offset=0;offset<lengths[slot_index];offset+=4){
          int32_t value=0;std::memcpy(&value,bytes.data()+offset,4);
          if(offset)result+=",";result+=std::to_string(value);
        }
      }
      result+="]}";
    }
    result+="],\"phase_runtime\":"+elite_phase::snapshot(base,object,key)+",\"special_runtime\":"+elite_special::snapshot(base,object)+"}";
  }
  int32_t after=-1;memory.read(current_battle+0x60,&after);result+="],\"tick_after\":"+std::to_string(after)+",\"valid\":";
  result+=after==tick?"true":"false";result+="}";return result;
}
} // namespace elite_telemetry
#include "elite_phase_telemetry.h"
#include "elite_special_snapshots.h"
