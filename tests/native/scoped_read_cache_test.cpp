#include "scoped_read_cache.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>

template<std::size_t BlockBytes> int check_cache() {
  std::array<unsigned char,4096> storage{};
  for(std::size_t i=0;i<storage.size();++i)storage[i]=static_cast<unsigned char>(i);
  int reads=0;bool short_block=false;
  auto direct=[&](uintptr_t address,void* output,std::size_t size){
    ++reads;
    if(address<0x1000 || size>storage.size() || address-0x1000>storage.size()-size)return false;
    if(short_block && size==BlockBytes){std::memcpy(output,storage.data()+address-0x1000,128);return false;}
    std::memcpy(output,storage.data()+address-0x1000,size);return true;
  };
  cr_native::ScopedReadCache<BlockBytes,2> cache;
  unsigned char value=0;
  if(!cache.read(0x1010,&value,1,direct) || value!=16 || reads!=1)return 1;
  if(!cache.begin())return 2;
  if(!cache.read(0x1010,&value,1,direct) || value!=16 || reads!=2)return 3;
  storage[16]=99;
  if(!cache.read(0x1010,&value,1,direct) || value!=16 || reads!=2)return 4;
  cache.end();
  if(!cache.read(0x1010,&value,1,direct) || value!=99 || reads!=3)return 5;
  cache.begin();
  if(!cache.read(0x1010,&value,1,direct) || value!=99 || reads!=4)return 6;
  if(!cache.read(0x1411,&value,1,direct) || value!=17)return 7;
  if(!cache.read(0x1010,&value,1,direct) || value!=99)return 8;
  unsigned char cross[4]{};
  if(!cache.read(0x11ff,cross,4,direct) || cross[0]!=255 || cross[1]!=0)return 9;
  cache.end();short_block=true;cache.begin();
  if(!cache.read(0x1010,&value,1,direct) || value!=99)return 10;
  storage[16]=77;
  if(!cache.read(0x1010,&value,1,direct) || value!=77)return 11;
  const int before=reads;
  if(cache.read(0,nullptr,0,direct) || cache.read(std::numeric_limits<uintptr_t>::max()-1,cross,4,direct) || reads!=before)return 12;
  short_block=false;cache.end();cache.begin();
  cache.read(0x1010,&value,1,direct);storage[16]=55;
  if(cache.begin())return 13;
  if(!cache.read(0x1010,&value,1,direct) || value!=55)return 14;
  cache.end();
  if(!cache.hits || !cache.fills || !cache.fallbacks)return 15;
  cache.begin();
  const auto fills=cache.fills;
  unsigned char body[80]{};
  if(!cache.read(0x1800,body,sizeof(body),direct) || body[0]!=0 || body[79]!=79 || cache.fills!=fills)return 16;
  if(!cache.read(0x1800,&value,1,direct))return 17;
  const int cached_reads=reads;
  if(!cache.read(0x1800,body,sizeof(body),direct) || reads!=cached_reads)return 18;
  cache.end();
  return 0;
}
int main(){
  const int a=check_cache<512>(),b=check_cache<1024>();
  if(a || b){std::printf("FAIL 512=%d 1024=%d\n",a,b);return 1;}
  std::puts("PASS scoped read cache 512/1024: uncached, hits, epoch invalidation, collision, boundary, short-read fallback, overflow, nested scope");
}
