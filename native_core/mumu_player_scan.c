#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_MAPS 2048
#define MAX_RESULTS 128

typedef struct{uint64_t start,end;int readable,writable;char path[256];}Map;
static int read_exact(int fd,uint64_t a,void*o,size_t s){uint8_t*p=o;size_t n=0;while(n<s){ssize_t v=pread(fd,p+n,s-n,(off_t)(a+n));if(v<=0)return 0;n+=(size_t)v;}return 1;}
static int maps(int pid,Map*m){char f[64],line[1024];snprintf(f,sizeof(f),"/proc/%d/maps",pid);FILE*h=fopen(f,"r");if(!h)return-1;int c=0;while(c<MAX_MAPS&&fgets(line,sizeof(line),h)){unsigned long long s,e,o;unsigned int a,b;unsigned long ino;char p[8];int u=0;if(sscanf(line,"%llx-%llx %7s %llx %x:%x %lu %n",&s,&e,p,&o,&a,&b,&ino,&u)<7)continue;Map*x=&m[c++];memset(x,0,sizeof(*x));x->start=s;x->end=e;x->readable=p[0]=='r';x->writable=p[1]=='w';char*n=line+u;while(*n==' '||*n=='\t')++n;size_t l=strcspn(n,"\r\n");if(l>=sizeof(x->path))l=sizeof(x->path)-1;memcpy(x->path,n,l);}fclose(h);return c;}
static int valid_hand(int fd,uint64_t ptr,int32_t out[4]){if(!ptr||!read_exact(fd,ptr,out,16))return 0;for(int i=0;i<4;++i)if(out[i]<-1||out[i]>7)return 0;return 1;}
int main(int argc,char**argv){
  if(argc!=3)return 2;
  int pid=atoi(argv[1]);int delta=(int)strtol(argv[2],NULL,0);
  Map mm[MAX_MAPS];int mc=maps(pid,mm);char mp[64];
  snprintf(mp,sizeof(mp),"/proc/%d/mem",pid);int fd=open(mp,O_RDONLY|O_CLOEXEC);if(fd<0)return 3;
  printf("{\"event\":\"mumu_player_scan\",\"pid\":%d,\"layout_delta\":%d,\"results\":[",pid,delta);
  int emitted=0;
  for(int mi=0;mi<mc&&emitted<MAX_RESULTS;++mi){
    Map*m=&mm[mi];int scudo=strstr(m->path,"scudo:")!=NULL;
    int low=m->end<0x100000000ULL&&(m->path[0]==0||strstr(m->path,"Mem_"));
    if(!m->readable||!m->writable||(!scudo&&!low)||(delta==9999&&!scudo))continue;
    for(uint64_t start=m->start;start<m->end&&emitted<MAX_RESULTS;start+=0x10000){
      size_t size=(size_t)((m->end-start)<0x10000?(m->end-start):0x10000);
      uint8_t*b=malloc(size);if(!b||!read_exact(fd,start,b,size)){free(b);continue;}
      for(size_t off=0;off+0x300<=size&&emitted<MAX_RESULTS;off+=8){
        uint64_t base=start+off;int32_t hs=0,cs=0,dc=0,el=0,rf=0;uint64_t hp=0,cp=0;
        if(delta==8888){
          uint64_t vt=0;int32_t hc=0,cc=0;
          memcpy(&vt,b+off,8);memcpy(&hp,b+off+0x210,8);memcpy(&hc,b+off+0x218,4);memcpy(&hs,b+off+0x21c,4);
          memcpy(&cp,b+off+0x220,8);memcpy(&cc,b+off+0x228,4);memcpy(&cs,b+off+0x22c,4);memcpy(&dc,b+off+0x230,4);memcpy(&el,b+off+0x2f8,4);
          if(vt<0x03000000ULL||vt>=0x05000000ULL||hc<4||hc>16||hs!=4||cc<1||cc>16||cs<1||cs>8||cs>cc||hp<0x100000000ULL||cp<0x100000000ULL||el<0||el>100000)continue;
          int32_t hand[4],cycle[8];if(!valid_hand(fd,hp,hand)||!read_exact(fd,cp,cycle,(size_t)cs*4))continue;int good=1;for(int i=0;i<cs;++i)if(cycle[i]<0||cycle[i]>7)good=0;if(!good)continue;
          if(emitted++)putchar(',');printf("{\"player\":\"0x%" PRIx64 "\",\"vtable\":\"0x%" PRIx64 "\",\"elixir_raw\":%d,\"hand_capacity\":%d,\"hand\":[%d,%d,%d,%d],\"cycle_capacity\":%d,\"cycle_size\":%d,\"cycle\":[",base,vt,el,hc,hand[0],hand[1],hand[2],hand[3],cc,cs);for(int i=0;i<cs;++i){if(i)putchar(',');printf("%d",cycle[i]);}printf("],\"deck_count_candidate\":%d,\"map\":\"%s\"}",dc,m->path);continue;
        }
        if(delta==9999){
          uint64_t vt=0,p10=0,p48=0;int32_t index=-1;
          memcpy(&vt,b+off,8);memcpy(&p10,b+off+0x10,8);memcpy(&p48,b+off+0x48,8);
          memcpy(&index,b+off+0x78,4);memcpy(&el,b+off+0x2f8,4);
          if(vt<0x03000000ULL||vt>=0x05000000ULL||p10<0x100000000ULL||p48<0x100000000ULL||index<0||index>100||el<0||el>100000)continue;
          if(emitted++)putchar(',');
          printf("{\"player\":\"0x%" PRIx64 "\",\"vtable\":\"0x%" PRIx64 "\",\"index\":%d,\"elixir_raw\":%d,\"p10\":\"0x%" PRIx64 "\",\"p48\":\"0x%" PRIx64 "\",\"map\":\"%s\"}",base,vt,index,el,p10,p48,m->path);
          continue;
        }
        memcpy(&hp,b+off+0x220+delta,8);memcpy(&hs,b+off+0x22c+delta,4);
        memcpy(&cp,b+off+0x230+delta,8);memcpy(&cs,b+off+0x23c+delta,4);
        memcpy(&dc,b+off+0x240+delta,4);memcpy(&rf,b+off+0x218+delta,4);memcpy(&el,b+off+0x2f8+delta,4);
        if(hs!=4||cs<0||cs>8||dc!=8||el<0||el>100000||rf<0||rf>10000||!cp)continue;
        int32_t hand[4];if(!valid_hand(fd,hp,hand))continue;if(emitted++)putchar(',');
        printf("{\"player\":\"0x%" PRIx64 "\",\"elixir_raw\":%d,\"refill_timer\":%d,\"hand\":[%d,%d,%d,%d],\"cycle_size\":%d,\"map\":\"%s\"}",base,el,rf,hand[0],hand[1],hand[2],hand[3],cs,m->path);
      }
      free(b);
    }
  }
  printf("]}\n");close(fd);return 0;
}
