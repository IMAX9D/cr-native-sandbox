// Snapshot layouts adapted from FirstLight CR contributors (Apache-2.0),
// commit 28d66cc0. x86_64 vtables independently located through class-name
// registration, data constructors and runtime constructors in 150535029.
namespace elite_special {
inline std::string ints(const std::vector<int32_t>& values){
  std::string s="[";for(size_t i=0;i<values.size();++i){if(i)s+=",";s+=std::to_string(values[i]);}return s+"]";
}
inline bool vector32(SafeMemoryReader& m,uint64_t root,size_t offset,int maximum,std::vector<int32_t>& out){
  uint64_t data=0;int32_t capacity=-1,count=-1;
  if(!m.read(root+offset,&data)||!m.read(root+offset+8,&capacity)||!m.read(root+offset+12,&count)||count<0||capacity<count||capacity>maximum||(count&&!data))return false;
  out.resize(static_cast<size_t>(count));return count==0||m.read_bytes(data,out.data(),out.size()*4);
}
inline bool object_present(SafeMemoryReader& m,int32_t key){
  uint64_t data=0;int32_t count=-1;
  if(!m.read(elite_telemetry::world+8,&data)||!m.read(elite_telemetry::world+0x14,&count)||count<0||count>2048)return false;
  for(int i=0;i<count;++i){uint64_t obj=0;int32_t id=0;if(m.read(data+static_cast<uint64_t>(i)*8,&obj)&&obj&&m.read(obj+8,&id)&&id==key)return true;}
  return false;
}
inline std::string snapshot(uintptr_t base,uintptr_t object){
  SafeMemoryReader m;uint64_t queue=0,array=0;int32_t capacity=0,count=0;
  if(!m.read(object+0x28,&queue))return "{\"valid\":false}";
  if(queue && (!m.read(queue+0x50,&array)||!m.read(queue+0x58,&capacity)||!m.read(queue+0x5c,&count)||count<0||capacity<count||capacity>256||(count&&!array)))return "{\"valid\":false}";
  uint64_t captures=0,relocation=0,modifier=0;
  for(int i=0;i<count;++i){uint64_t r=0,vt=0,owner=0;
    if(!m.read(array+static_cast<uint64_t>(i)*8,&r)||!r)continue;
    if(!m.read(r,&vt))return "{\"valid\":false}";
    uint64_t* slot=vt==base+0x196b1e8?&captures:vt==base+0x196c600?&relocation:vt==base+0x196c2e8?&modifier:nullptr;
    if(slot){if(*slot||!m.read(r+0x20,&owner)||owner!=object)return "{\"valid\":false}";*slot=r;}
  }
  std::string result="{\"valid\":true,\"capture\":";
  if(!captures)result+="{\"present\":false}";
  else{
    uint64_t data=0,vt=0;int32_t limit=0,delay=-1,pause=-1,drag=-1,cooldown=-1,remaining=-1,frequency=0,accumulator=-1;
    std::vector<int32_t> active,completed,elapsed;
    const bool ok=m.read(captures+0x18,&data)&&data&&m.read(data,&vt)&&vt==base+0x1959758&&
      m.read(data+0xf8,&limit)&&m.read(data+0x108,&delay)&&m.read(data+0x168,&pause)&&m.read(data+0x10c,&drag)&&
      m.read(data+0x114,&cooldown)&&m.read(data+0xf4,&frequency)&&m.read(captures+0xc0,&remaining)&&m.read(captures+0xc4,&accumulator)&&
      vector32(m,captures,0x60,1024,active)&&vector32(m,captures,0x70,1024,completed)&&vector32(m,captures,0x80,1024,elapsed)&&
      limit>0&&limit<=1000&&delay>=0&&pause>=0&&drag>=0&&cooldown>=0&&remaining>=0&&remaining<=cooldown&&frequency>0&&accumulator>=0&&
      active.size()==elapsed.size()&&completed.size()<=active.size()&&active.size()<=static_cast<size_t>(limit)&&active.size()<=160;
    if(!ok)result+="{\"present\":true,\"valid\":false}";
    else{
      bool valid=true;
      for(size_t i=0;i<active.size();++i){if(active[i]<=0||elapsed[i]<0)valid=false;for(size_t j=0;j<i;++j)if(active[j]==active[i])valid=false;}
      for(auto id:completed)if(std::find(active.begin(),active.end(),id)==active.end())valid=false;
      for(size_t i=0;i<completed.size();++i)for(size_t j=0;j<i;++j)if(completed[i]==completed[j])valid=false;
      result+="{\"present\":true,\"valid\":";result+=valid?"true":"false";
      result+=",\"cooldown_remaining_ms\":"+std::to_string(remaining)+",\"cooldown_ms\":"+std::to_string(cooldown)+",\"hit_frequency_ms\":"+std::to_string(frequency)+",\"hit_accumulator_ms\":"+std::to_string(accumulator)+",\"targets\":[";
      for(size_t i=0;i<active.size();++i){
        if(i)result+=",";int phase=0,budget=-1;
        if(!object_present(m,active[i]))phase=4;
        else if(std::find(completed.begin(),completed.end(),active[i])!=completed.end())phase=3;
        else if(elapsed[i]<delay){phase=0;budget=delay-elapsed[i];}
        else if(elapsed[i]<delay+pause){phase=1;budget=delay+pause-elapsed[i];}
        else{phase=2;budget=std::max(0,delay+pause+drag-elapsed[i]);}
        result+="{\"key\":"+std::to_string(active[i])+",\"phase\":"+std::to_string(phase)+",\"remaining_ms\":"+std::to_string(budget)+"}";
      }
      result+="]}";
    }
  }
  result+=",\"relocation\":";
  if(!relocation)result+="{\"present\":false}";
  else{
    uint64_t data=0,vt=0;int32_t duration=0,remaining=-1,stage=-1,state=-1;std::vector<int32_t> thresholds;
    bool valid=m.read(relocation+0x18,&data)&&data&&m.read(data,&vt)&&vt==base+0x195be28&&
      m.read(data+0xf4,&duration)&&vector32(m,data,0x148,64,thresholds)&&m.read(relocation+0x60,&remaining)&&m.read(relocation+0x74,&stage)&&m.read(object+0x11c,&state)&&
      duration>0&&remaining>=0&&remaining<=duration&&!thresholds.empty()&&thresholds.size()<=16&&stage>=1&&stage<=static_cast<int>(thresholds.size())*2+1&&state>=0&&state<=16;
    for(size_t i=0;i<thresholds.size();++i)if(thresholds[i]<0||thresholds[i]>100||(i&&thresholds[i]>=thresholds[i-1]))valid=false;
    result+="{\"present\":true,\"valid\":";result+=valid?"true":"false";
    result+=",\"stage\":"+std::to_string(stage)+",\"remaining_ms\":"+std::to_string(remaining)+",\"duration_ms\":"+std::to_string(duration)+",\"owner_state\":"+std::to_string(state)+",\"thresholds\":"+ints(thresholds)+"}";
  }
  result+=",\"periodic_modifier\":";
  if(!modifier)result+="{\"present\":false}";
  else{
    uint64_t data=0,vt=0;int32_t state=-1,period=0,completed=-1,linger=0,remaining=-1,source=0;unsigned char finished=0;
    bool valid=m.read(modifier+0x18,&data)&&data&&m.read(data,&vt)&&vt==base+0x195b9d8&&m.read(modifier+0x10,&finished)&&
      m.read(modifier+0x60,&state)&&m.read(data+0xec,&period)&&m.read(data+0x108,&linger)&&m.read(modifier+0x70,&completed)&&m.read(modifier+0x74,&remaining)&&m.read(modifier+0x78,&source)&&
      finished<=1&&state>=0&&state<=2&&period>0&&completed>=0&&completed<period&&linger>0&&remaining>=0&&remaining<=linger;
    if(valid&&state>0&&!finished)valid=source>0&&((state==1&&remaining==0)||(state==2&&remaining>0));
    if(valid&&(finished||state==0))result+="{\"present\":false}";
    else{
      const bool source_alive=source>0&&object_present(m,source);
      if(valid&&state==1&&!source_alive)remaining=linger;
      result+="{\"present\":true,\"valid\":";result+=valid?"true":"false";
      result+=",\"period\":"+std::to_string(period)+",\"completed\":"+std::to_string(completed)+",\"linger_ms\":"+std::to_string(linger)+",\"remaining_ms\":"+std::to_string(remaining)+",\"source_death_linger\":";
      result+=(state==2||!source_alive)?"true":"false";result+="}";
    }
  }
  result+=",\"extra_spawn\":";
  uint64_t object_vt=0,data=0,ability=0,ability_vt=0;int32_t extra_base=0,extra_limit=0,extra_current=0;
  if(m.read(object,&object_vt)&&object_vt==base+0x19693f8&&m.read(object+0x48,&data)&&data&&
     m.read(data+0x710,&ability)&&ability&&m.read(ability,&ability_vt)&&ability_vt==base+0x194f300&&
     m.read(ability+0x104,&extra_base)&&m.read(ability+0x108,&extra_limit)&&extra_base>0&&extra_limit>extra_base){
    const bool valid=m.read(object+0x1f0,&extra_current)&&extra_current>=0&&extra_current<=extra_limit-extra_base;
    result+="{\"present\":true,\"valid\":";result+=valid?"true":"false";
    result+=",\"current\":"+std::to_string(extra_current)+",\"capacity\":"+std::to_string(extra_limit-extra_base)+"}";
  }else result+="{\"present\":false}";
  result+=",\"projectile\":";
  uint64_t projectile_vt=0,projectile_data=0,homing=0,homing_world=0;int32_t data_id=0,drag_speed=0,homing_key=0;unsigned char terminal=0,drag_stage=0;
  if(m.read(object,&projectile_vt)&&projectile_vt==base+0x1969b38){
    // EXT rows use signed name hashes, including positive hashes, not the
    // traditional 10,000,000 table range. Exact runtime vtable above is the
    // class identity; preserve and separately resolve its actual data name.
    bool valid=m.read(object+0x48,&projectile_data)&&projectile_data&&m.read(projectile_data+0x40,&data_id)&&
      !elite_telemetry::data_name(m,projectile_data).empty()&&
      m.read(object+0x170,&terminal)&&terminal<=1&&m.read(projectile_data+0x1b0,&drag_speed)&&m.read(object+0x110,&homing);
    if(valid&&drag_speed>0)valid=m.read(object+0x189,&drag_stage)&&drag_stage<=1;
    bool homing_known=valid && !homing;
    if(valid&&homing)homing_known=m.read(homing+0x10,&homing_world)&&homing_world==elite_telemetry::world&&m.read(homing+8,&homing_key)&&object_present(m,homing_key);
    result+="{\"present\":true,\"valid\":";result+=valid?"true":"false";
    result+=",\"terminal\":"+std::to_string(terminal)+",\"drag_configured\":";result+=drag_speed>0?"true":"false";
    result+=",\"drag_stage\":"+std::to_string(drag_stage)+",\"homing_known\":";result+=homing_known?"true":"false";
    result+=",\"homing\":";result+=homing?"true":"false";result+="}";
  }else result+="{\"present\":false}";
  uint64_t head=0,variable_count=0;
  bool variables_valid=m.read(object+0xd8,&head)&&m.read(object+0xe0,&variable_count)&&variable_count<=128&&((head==0)==(variable_count==0));
  std::vector<uint64_t> visited;std::vector<int32_t> keys,values;uint64_t node=head;
  if(variables_valid)for(uint64_t i=0;i<variable_count;++i){
    uint64_t next=0;int64_t hash=0;int32_t key=0,value=0;
    if(!node||std::find(visited.begin(),visited.end(),node)!=visited.end()||!m.read(node,&next)||!m.read(node+8,&hash)||!m.read(node+0x10,&key)||!m.read(node+0x14,&value)||
       hash!=static_cast<int64_t>(key)||std::find(keys.begin(),keys.end(),key)!=keys.end()){variables_valid=false;break;}
    visited.push_back(node);keys.push_back(key);values.push_back(value);node=next;
  }
  if(node)variables_valid=false;
  result+=",\"variables_valid\":";result+=variables_valid?"true":"false";
  result+=",\"variable_keys\":"+ints(keys)+",\"variable_values\":"+ints(values)+"}";return result;
}
} // namespace elite_special
