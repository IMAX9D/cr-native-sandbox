#pragma once
// Task-only observation/action adapter; the native singleton is never changed.
static uint64_t g_resident_observation_battle = 0;
template<class Reader>
bool resident_read_battle(const Reader& reader, uint64_t state, uint64_t* out) {
  if (g_resident_observation_battle) { *out=g_resident_observation_battle; return true; }
  return reader.read(state+0x90,out);
}
// Isolated research exports only; not compiled into the production host.
extern "C" JNIEXPORT jstring JNICALL
Java_royale_nativehost_JniHost_nativeResidentCode(JNIEnv* env,jclass,jstring path,jint rva,jint size) {
  if(rva<0x600000 || rva>=0x1600000 || size<1 || size>2048 || rva>0x1600000-size)return env->NewStringUTF("{\"error\":\"range\"}");
  const char* p=env->GetStringUTFChars(path,nullptr);
  void* handle=dlopen(p,RTLD_NOW|RTLD_NOLOAD|RTLD_LOCAL);env->ReleaseStringUTFChars(path,p);
  Dl_info info{};
  void* jni=handle?dlsym(handle,"JNI_OnLoad"):nullptr;
  if(!jni || !dladdr(jni,&info) || uintptr_t(jni)-uintptr_t(info.dli_fbase)!=0x1458bc0){
    if(handle)dlclose(handle);return env->NewStringUTF("{\"error\":\"runtime\"}");
  }
  const uintptr_t base=reinterpret_cast<uintptr_t>(info.dli_fbase);
  std::vector<unsigned char> bytes(static_cast<size_t>(size));
  const int fd=open("/proc/self/mem",O_RDONLY|O_CLOEXEC);
  const bool ok=fd>=0 && pread(fd,bytes.data(),bytes.size(),static_cast<off_t>(base+rva))==size;
  if(fd>=0)close(fd);dlclose(handle);
  if(!ok)return env->NewStringUTF("{\"error\":\"read\"}");
  const char digits[]="0123456789abcdef";std::string hex;
  for(auto b:bytes){hex+=digits[b>>4];hex+=digits[b&15];}
  const std::string json="{\"rva\":"+std::to_string(rva)+",\"bytes\":\""+hex+"\"}";
  return env->NewStringUTF(json.c_str());
}
