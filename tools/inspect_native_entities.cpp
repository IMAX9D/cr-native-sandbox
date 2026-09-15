// Read-only diagnostics for the owned 150535029 x86_64 replay host.
// The caller must verify the runtime SHA before invoking this tool.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

static int memfd=-1;
template<class T> T read_at(uint64_t p){T v{};if(p && pread(memfd,&v,sizeof(v),static_cast<off_t>(p))==sizeof(v))return v;return {};}
static void words(uint64_t p,int length){
  std::printf("[");for(int i=0;i<length;i+=8)std::printf("%s\"%llx\"",i?",":"",static_cast<unsigned long long>(read_at<uint64_t>(p+i)));std::printf("]");
}
int main(int argc,char**argv){
  if(argc!=2)return 64;
  int pid=std::atoi(argv[1]);char path[80];std::snprintf(path,sizeof(path),"/proc/%d/maps",pid);
  FILE* maps=std::fopen(path,"r");if(!maps)return 1;
  char line[2048];uint64_t base=0;
  while(std::fgets(line,sizeof(line),maps)){
    unsigned long long start=0,offset=0;char perms[8];
    if(std::strstr(line,"/cr-native-direct-") && std::strstr(line,"/libg.so") &&
       std::sscanf(line,"%llx-%*x %7s %llx",&start,perms,&offset)==3 && offset==0){base=start;break;}
  }
  std::fclose(maps);if(!base)return 2;
  std::snprintf(path,sizeof(path),"/proc/%d/mem",pid);memfd=open(path,O_RDONLY|O_CLOEXEC);if(memfd<0)return 3;
  uint64_t manager=read_at<uint64_t>(base+0x1a85978),state=read_at<uint64_t>(manager+0x20),battle=read_at<uint64_t>(state+0x90);
  uint64_t logic=read_at<uint64_t>(battle+0xa8),registry=read_at<uint64_t>(logic+8),collection=read_at<uint64_t>(registry+0x40);
  int count=read_at<int>(collection+0x14);uint64_t data=read_at<uint64_t>(collection+8);
  if(count<0||count>2048)return 4;
  std::printf("{\"base\":\"%llx\",\"tick\":%d,\"entities\":[",static_cast<unsigned long long>(base),read_at<int>(battle+0x60));
  bool first=true;
  for(int i=0;i<count;i++){
    uint64_t e=read_at<uint64_t>(data+i*8);if(!e)continue;
    int id=read_at<int>(e+8);if(id<4000000||id>=6000000)continue;
    std::printf("%s{\"key\":%d,\"object\":\"%llx\",\"kind\":%d,\"side\":%d,\"card\":%d,\"object_words\":",first?"":",",id,static_cast<unsigned long long>(e),read_at<int>(e+0x30),read_at<int>(e+0x78),read_at<int>(e+0xac));first=false;words(e,0x220);
    uint64_t native_data=read_at<uint64_t>(e+0x48);
    std::printf(",\"data\":\"%llx\",\"data_words\":",static_cast<unsigned long long>(native_data));words(native_data,0x80);
    std::printf(",\"components\":[");uint64_t array=read_at<uint64_t>(e+0x18);
    for(int j=0;j<3;j++){
      uint64_t component=read_at<uint64_t>(array+8*j),vt=read_at<uint64_t>(component);
      std::printf("%s{\"address\":\"%llx\",\"vtable_rva\":\"%llx\",\"functions\":[",j?",":"",static_cast<unsigned long long>(component),static_cast<unsigned long long>(vt>=base?vt-base:0));
      for(int k=0;k<16;k++){uint64_t fn=read_at<uint64_t>(vt+k*8);std::printf("%s\"%llx\"",k?",":"",static_cast<unsigned long long>(fn>=base?fn-base:0));}
      std::printf("],\"words\":");words(component,0x100);std::printf("}");
    }
    std::printf("]}");
  }
  std::printf("]}\n");close(memfd);return 0;
}
