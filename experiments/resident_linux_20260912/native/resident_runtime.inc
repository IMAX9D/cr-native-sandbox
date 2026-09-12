// ISOLATED EXPERIMENT ONLY. One execution lane, four independently constructed
// inner battle managers. No outer manager duplication and no singleton writes.
namespace {
struct ResidentSlot { void* battle=nullptr; void* replay=nullptr; EpisodeState episode{}; };
ResidentSlot resident_slots[4];
int resident_selected=-1;
EpisodeState resident_host_episode{};
void resident_unbind() {
  if(resident_selected>=0) resident_slots[resident_selected].episode=g_episode;
  if(resident_selected>=0) g_episode=resident_host_episode;
  resident_selected=-1;g_resident_observation_battle=0;
}
void resident_bind(int id) {
  resident_unbind();resident_host_episode=g_episode;
  resident_selected=id;g_episode=resident_slots[id].episode;
  g_resident_observation_battle=uintptr_t(resident_slots[id].battle);
}
}
extern "C" JNIEXPORT jstring JNICALL
Java_royale_nativehost_JniHost_nativeResident(JNIEnv* env,jclass,jstring path,jint id,jint op,jstring replay,jint steps) {
  native_profile::Scope timing(op==2?native_profile::STEP:(op==0?native_profile::CREATE:native_profile::BIND));
  native_profile::Context profile_context(op==2?1:0);
  if(id<0 || id>=4 || steps<0 || steps>20000) {throw_state(env,"resident argument range");return nullptr;}
  const char* chars=env->GetStringUTFChars(path,nullptr);
  void* h=dlopen(chars,RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);env->ReleaseStringUTFChars(path,chars);
  Dl_info info{};void* symbol=h?dlsym(h,"JNI_OnLoad"):nullptr;
  if(!symbol || !dladdr(symbol,&info) || uintptr_t(symbol)-uintptr_t(info.dli_fbase)!=kExpectedJniOnLoadRva) {
    if(h)dlclose(h);throw_state(env,"resident runtime guard");return nullptr;
  }
  const uintptr_t base=uintptr_t(info.dli_fbase);SafeMemoryReader memory;
  auto& slot=resident_slots[id];
  std::string out="{\"ok\":true}";
  if(op==0) {
    resident_unbind();uint64_t outer=0,state=0,source=0,a=0,b=0;
    if(!memory.read(base+kManagerGlobalRva,&outer)||!outer||!memory.read(outer+0x20,&state)||!state||
       !memory.read(state+0x90,&source)||!source||!memory.read(source+0x208,&a)||!a||
       !memory.read(source+0x210,&b)||!b||!replay) {
      dlclose(h);throw_state(env,"resident requires bootstrapped host");return nullptr;
    }
    // This fixture uses the already prewarmed location of the bootstrap replay.
    alignas(8) unsigned char text[16]{},key[16]{};
    auto str=reinterpret_cast<NativeStringFromUtf8>(base+kNativeStringFromUtf8Rva);
    auto dstr=reinterpret_cast<NativeStringDestroy>(base+kNativeStringDestroyRva);
    const char* r=env->GetStringUTFChars(replay,nullptr);str(text,r);env->ReleaseStringUTFChars(replay,r);
    void* root=reinterpret_cast<ParseJsonObject>(base+kParseJsonObjectRva)(text,0,100);dstr(text);
    str(key,"battle");void* json=root?reinterpret_cast<JsonGetObject>(base+kJsonGetObjectRva)(root,key,true):nullptr;dstr(key);
    if(!json) {dlclose(h);throw_state(env,"resident replay lacks battle");return nullptr;}
    if(slot.battle) {
      reinterpret_cast<void(*)(void*)>(base+0x10b7a60)(slot.battle);
      reinterpret_cast<void(*)(void*)>(base+0x998ce0)(slot.battle);
      reinterpret_cast<NativeFree>(base+kNativeFreeRva)(slot.battle);
      if(slot.replay) reinterpret_cast<void(*)(void*)>((*reinterpret_cast<void***>(slot.replay))[1])(slot.replay);
      slot={};
    }
    slot.battle=reinterpret_cast<NativeAlloc>(base+kNativeAllocRva)(0x230);slot.replay=root;
    std::fprintf(stderr,"RESIDENT_CREATE slot=%d battle=%p\n",id,slot.battle);std::fflush(stderr);
    reinterpret_cast<void(*)(void*,int,void*,void*,int)>(base+0x10b74c0)(slot.battle,1,reinterpret_cast<void*>(a),reinterpret_cast<void*>(b),0);
    reinterpret_cast<void(*)(void*,int)>(base+0x10b7950)(slot.battle,1);
    reinterpret_cast<BattleLoadReplay>(base+kBattleLoadReplayRva)(slot.battle,root,nullptr,nullptr);
    resident_bind(id);reset_episode_state();
    capture_episode_state(memory,base,uintptr_t(slot.battle));
    out="{\"ok\":true,\"slot\":"+std::to_string(id)+",\"battle\":"+std::to_string(uintptr_t(slot.battle))+",\"episode\":"+episode_json()+"}";
    resident_unbind();
  } else if(op==3) {resident_unbind();}
  else if(!slot.battle) {dlclose(h);throw_state(env,"resident slot not created");return nullptr;}
  else if(op==1) {resident_bind(id);}
  else if(op==2) {
    resident_bind(id);capture_episode_state(memory,base,uintptr_t(slot.battle));
    const int before=g_episode.tick;int done=0;
    // Native update count is not identical to elapsed game ticks at settlement.
    // Resolve a blocked terminal boundary without asking the policy to act on a
    // partial decision interval. Extra calls are bounded and do not change dt.
    for(;!g_episode.terminated && (done<steps ||
          (steps>0 && g_episode.tick<before+steps && g_episode.command_gate_code!=0 && done<steps+8));++done) {
      int old=g_episode.tick;
      if(!g_episode.core_only_terminal_phase) {
        g_episode.tiebreak_start_crowns[0]=episode_crowns(0);
        g_episode.tiebreak_start_crowns[1]=episode_crowns(1);
      }
      {native_profile::Scope core_timing(native_profile::ENGINE);
       reinterpret_cast<void(*)(void*)>(base+0x10b8ae0)(slot.battle);}
      capture_episode_state(memory,base,uintptr_t(slot.battle));
      g_episode.stalled_updates=(g_episode.tick>=100 && g_episode.tick==old)?g_episode.stalled_updates+1:0;
      if(g_episode.tick>=6000 || g_episode.stalled_updates>0) g_episode.core_only_terminal_phase=true;
      if(g_episode.logic_state==1 && (g_episode.logic_substate==6 || g_episode.logic_substate==15)) {
        g_episode.terminated=true;g_episode.termination_reason="native_logic_terminal";
      } else if(g_episode.core_only_terminal_phase && g_episode.tiebreak_start_crowns[0]>=0 &&
                (episode_crowns(0)>g_episode.tiebreak_start_crowns[0] || episode_crowns(1)>g_episode.tiebreak_start_crowns[1])) {
        g_episode.terminated=true;g_episode.termination_reason="native_tiebreak_hp_drain";
      } else if(g_episode.core_only_terminal_phase && g_episode.stalled_updates>=2 &&
                episode_crowns(0)==episode_crowns(1) && episode_lowest_living_tower_hp(0)==episode_lowest_living_tower_hp(1)) {
        g_episode.terminated=true;g_episode.termination_reason="native_tiebreak_exact_draw";
      } else if(g_episode.stalled_updates>=2 && episode_crowns(0)+episode_crowns(1)>0) {
        g_episode.terminated=true;g_episode.termination_reason="native_logic_clock_stopped";
      }
    }
    out="{\"ok\":true,\"tick_before\":"+std::to_string(before)+",\"tick_after\":"+std::to_string(g_episode.tick)+",\"stepped\":"+std::to_string(done)+",\"settlement_updates\":"+std::to_string(std::max(0,done-steps))+",\"episode\":"+episode_json()+"}";
    resident_unbind();
  } else {dlclose(h);throw_state(env,"unknown resident operation");return nullptr;}
  dlclose(h);return env->NewStringUTF(out.c_str());
}
