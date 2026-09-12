#pragma once
#include <time.h>
namespace native_profile {
enum Metric { STEP, ENGINE, CAPTURE_STEP, OBSERVE, CAPTURE_OBS, EPISODE_STEP, EPISODE_OBS, ACTION, GRID, BIND, CREATE, OTHER_CAPTURE, COUNT };
inline const char* names[]={"step_total","engine_update","capture_step","observe_total","capture_observe","episode_json_step","episode_json_observe","action","legality_grid","bind_unbind","create","capture_other"};
struct Counter { std::atomic<uint64_t> calls{0},wall{0},cpu{0}; };
inline Counter counters[COUNT];
inline thread_local int context=0;
inline bool enabled(){static bool value=[] {const char* p=std::getenv("CR_NATIVE_PROFILE_TIMING");return p&&std::strcmp(p,"1")==0;}();return value;}
inline uint64_t clock(clockid_t id){timespec t{};clock_gettime(id,&t);return uint64_t(t.tv_sec)*1000000000ULL+uint64_t(t.tv_nsec);}
struct Context {int prior;explicit Context(int c):prior(context){context=c;}~Context(){context=prior;}};
struct Scope {
  Metric metric;uint64_t wall=0,cpu=0;bool active;
  explicit Scope(Metric m):metric(m),active(enabled()){if(active){wall=clock(CLOCK_MONOTONIC);cpu=clock(CLOCK_THREAD_CPUTIME_ID);}}
  ~Scope(){if(active){const auto c=clock(CLOCK_THREAD_CPUTIME_ID)-cpu,w=clock(CLOCK_MONOTONIC)-wall;
    counters[metric].calls.fetch_add(1,std::memory_order_relaxed);counters[metric].wall.fetch_add(w,std::memory_order_relaxed);counters[metric].cpu.fetch_add(c,std::memory_order_relaxed);}}
};
}
extern "C" JNIEXPORT jlong JNICALL Java_royale_nativehost_JniHost_nativeThreadCpuNanos(JNIEnv*,jclass){return static_cast<jlong>(native_profile::clock(CLOCK_THREAD_CPUTIME_ID));}
extern "C" JNIEXPORT jstring JNICALL Java_royale_nativehost_JniHost_nativeTimingStats(JNIEnv* env,jclass){
  std::string result="{\"enabled\":";result+=native_profile::enabled()?"true":"false";result+=",\"metrics\":{";
  for(int i=0;i<native_profile::COUNT;++i){auto& c=native_profile::counters[i];char buffer[256];
    std::snprintf(buffer,sizeof(buffer),"%s\"%s\":{\"calls\":%llu,\"wall_ns\":%llu,\"cpu_ns\":%llu}",i?",":"",native_profile::names[i],
      static_cast<unsigned long long>(c.calls.load()),static_cast<unsigned long long>(c.wall.load()),static_cast<unsigned long long>(c.cpu.load()));result+=buffer;}
  result+="}}";return env->NewStringUTF(result.c_str());
}
