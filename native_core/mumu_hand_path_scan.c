#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define MAX_NODES 8192
typedef struct{uint64_t a;int d,p0,p1,p2;}Node;
static int rd(int f,uint64_t a,void*o,size_t s){uint8_t*p=o;size_t n=0;while(n<s){ssize_t v=pread(f,p+n,s-n,(off_t)(a+n));if(v<=0)return 0;n+=v;}return 1;}
static int seen(Node*n,int c,uint64_t a){for(int i=0;i<c;++i)if(n[i].a==a)return 1;return 0;}
static int hand(const int32_t*v){int mask=0,distinct=0;for(int i=0;i<4;++i){if(v[i]<-1||v[i]>7)return 0;int bit=1<<(v[i]+1);if(!(mask&bit)){mask|=bit;distinct++;}}return distinct>=3;}
int main(int ac,char**av){if(ac!=3)return 2;int pid=atoi(av[1]);uint64_t root=strtoull(av[2],0,0);char p[64];snprintf(p,sizeof(p),"/proc/%d/mem",pid);int f=open(p,O_RDONLY|O_CLOEXEC);if(f<0)return 3;Node nodes[MAX_NODES];int count=1;nodes[0]=(Node){root,0,-1,-1,-1};printf("{\"event\":\"mumu_hand_path_scan\",\"root\":\"0x%" PRIx64 "\",\"results\":[",root);int out=0;for(int c=0;c<count&&c<MAX_NODES;++c){Node n=nodes[c];uint8_t b[0x800];if(!rd(f,n.a,b,sizeof(b)))continue;for(int o=0;o<=0x7f0;o+=4){int32_t v[4];memcpy(v,b+o,16);if(hand(v)){if(out++)putchar(',');printf("{\"node\":\"0x%" PRIx64 "\",\"depth\":%d,\"path\":[%d,%d,%d],\"field\":%d,\"kind\":\"inline\",\"values\":[%d,%d,%d,%d]}",n.a,n.d,n.p0,n.p1,n.p2,o,v[0],v[1],v[2],v[3]);}}
for(int o=0;o<=0x7f8;o+=8){uint64_t q=0;memcpy(&q,b+o,8);if(q<0x100000000ULL||(q&7))continue;int32_t v[4];if(rd(f,q,v,16)&&hand(v)){if(out++)putchar(',');printf("{\"node\":\"0x%" PRIx64 "\",\"depth\":%d,\"path\":[%d,%d,%d],\"field\":%d,\"kind\":\"pointer\",\"array\":\"0x%" PRIx64 "\",\"values\":[%d,%d,%d,%d]}",n.a,n.d,n.p0,n.p1,n.p2,o,q,v[0],v[1],v[2],v[3]);}if(n.d<3&&count<MAX_NODES){uint64_t probe=0;if(rd(f,q,&probe,8)&&!seen(nodes,count,q))nodes[count++]=(Node){q,n.d+1,n.d==0?o:n.p0,n.d==1?o:n.p1,n.d==2?o:n.p2};}}}
printf("],\"nodes\":%d}\n",count);close(f);return 0;}
