#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int read_exact(int fd,uint64_t a,void*o,size_t s){uint8_t*p=o;size_t n=0;while(n<s){ssize_t v=pread(fd,p+n,s-n,(off_t)(a+n));if(v<=0)return 0;n+=(size_t)v;}return 1;}

int main(int argc,char**argv){
  if(argc!=3)return 2;
  int pid=atoi(argv[1]);uint64_t logic=strtoull(argv[2],NULL,0);
  char path[64];snprintf(path,sizeof(path),"/proc/%d/mem",pid);
  int fd=open(path,O_RDONLY|O_CLOEXEC);if(fd<0)return 3;
  uint8_t raw[0x800];if(!read_exact(fd,logic,raw,sizeof(raw)))return 4;
  printf("{\"event\":\"mumu_private_relocate\",\"pid\":%d,\"logic\":\"0x%" PRIx64 "\",\"results\":[",pid,logic);
  int emitted=0;
  for(int logic_off=0;logic_off<=0x7f8;logic_off+=8){
    uint64_t player=0;memcpy(&player,raw+logic_off,8);
    if(player<0x10000||(player&7))continue;
    for(int delta=-0x80;delta<=0x100;delta+=8){
      uint64_t hand_ptr=0,cycle_ptr=0;
      int32_t elixir=-1,hand_size=-1,cycle_size=-1,deck_count=-1,refill=-1;
      int32_t hand[4]={};
      if(!read_exact(fd,player+0x220+delta,&hand_ptr,8)||
         !read_exact(fd,player+0x22c+delta,&hand_size,4)||
         !read_exact(fd,player+0x230+delta,&cycle_ptr,8)||
         !read_exact(fd,player+0x23c+delta,&cycle_size,4)||
         !read_exact(fd,player+0x240+delta,&deck_count,4)||
         !read_exact(fd,player+0x2f8+delta,&elixir,4)||
         !read_exact(fd,player+0x218+delta,&refill,4))continue;
      if(hand_size!=4||cycle_size<0||cycle_size>8||deck_count!=8||
         elixir<0||elixir>100000||refill<0||refill>10000||
         !hand_ptr||!cycle_ptr||!read_exact(fd,hand_ptr,hand,sizeof(hand)))continue;
      int good=1;for(int i=0;i<4;++i)if(hand[i]<-1||hand[i]>7)good=0;
      if(!good)continue;
      if(emitted++)putchar(',');
      printf("{\"logic_offset\":%d,\"player\":\"0x%" PRIx64 "\",\"layout_delta\":%d,\"elixir_raw\":%d,\"refill_timer\":%d,\"hand\":[%d,%d,%d,%d],\"cycle_size\":%d}",logic_off,player,delta,elixir,refill,hand[0],hand[1],hand[2],hand[3],cycle_size);
    }
  }
  printf("]}\n");close(fd);return 0;
}
