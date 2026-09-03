#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define MAX_MAPS 2048
#define MAX_VECTORS 8192
typedef struct{uint64_t s,e;int r,w;char p[128];}Map;
typedef struct{uint64_t h,d;int32_t v[4];}Vec;
static int rd(int f,uint64_t a,void*o,size_t s){uint8_t*p=o;size_t n=0;while(n<s){ssize_t v=pread(f,p+n,s-n,(off_t)(a+n));if(v<=0)return 0;n+=v;}return 1;}
static int maps(int pid,Map*m){char f[64],l[1024];snprintf(f,sizeof(f),"/proc/%d/maps",pid);FILE*h=fopen(f,"r");if(!h)return-1;int c=0;while(c<MAX_MAPS&&fgets(l,sizeof(l),h)){unsigned long long s,e,o;unsigned int a,b;unsigned long ino;char q[8];int u=0;if(sscanf(l,"%llx-%llx %7s %llx %x:%x %lu %n",&s,&e,q,&o,&a,&b,&ino,&u)<7)continue;Map*x=&m[c++];memset(x,0,sizeof(*x));x->s=s;x->e=e;x->r=q[0]=='r';x->w=q[1]=='w';char*n=l+u;while(*n==' '||*n=='\t')++n;size_t z=strcspn(n,"\r\n");if(z>=sizeof(x->p))z=sizeof(x->p)-1;memcpy(x->p,n,z);}fclose(h);return c;}
static int four(const int32_t*v){int mask=0;for(int i=0;i<4;++i){if(v[i]<0||v[i]>7||(mask&(1<<v[i])))return 0;mask|=1<<v[i];}return mask;}
int main(int ac,char**av){if(ac!=2)return 2;int pid=atoi(av[1]);Map mm[MAX_MAPS];int mc=maps(pid,mm);char p[64];snprintf(p,sizeof(p),"/proc/%d/mem",pid);int f=open(p,O_RDONLY|O_CLOEXEC);if(f<0)return 3;Vec*vv=calloc(MAX_VECTORS,sizeof(Vec));int vc=0;for(int mi=0;mi<mc&&vc<MAX_VECTORS;++mi){Map*m=&mm[mi];if(!m->r||!m->w||!strstr(m->p,"scudo:"))continue;for(uint64_t s=m->s;s<m->e&&vc<MAX_VECTORS;s+=0x1000){size_t n=(size_t)((m->e-s)<0x1000?(m->e-s):0x1000);uint8_t*b=malloc(n);if(!b||!rd(f,s,b,n)){free(b);continue;}for(size_t o=0;o+16<=n&&vc<MAX_VECTORS;o+=8){uint64_t data=0;int32_t cap=0,size=0;memcpy(&data,b+o,8);memcpy(&cap,b+o+8,4);memcpy(&size,b+o+12,4);if(size!=4||cap<4||cap>8||data<0x100000000ULL)continue;int32_t v[4];if(!rd(f,data,v,16)||!four(v))continue;vv[vc].h=s+o;vv[vc].d=data;memcpy(vv[vc].v,v,16);vc++;}free(b);}}
printf("{\"event\":\"mumu_cycle_vector_scan\",\"pid\":%d,\"vector_count\":%d,\"pairs\":[",pid,vc);int out=0;for(int i=0;i<vc;++i)for(int j=i+1;j<vc;++j){uint64_t dist=vv[i].h>vv[j].h?vv[i].h-vv[j].h:vv[j].h-vv[i].h;if(dist>0x200)continue;int a=four(vv[i].v),b=four(vv[j].v);if((a&b)||((a|b)!=0xff))continue;if(out++)putchar(',');printf("{\"header0\":\"0x%" PRIx64 "\",\"data0\":\"0x%" PRIx64 "\",\"values0\":[%d,%d,%d,%d],\"header1\":\"0x%" PRIx64 "\",\"data1\":\"0x%" PRIx64 "\",\"values1\":[%d,%d,%d,%d],\"distance\":%" PRIu64 "}",vv[i].h,vv[i].d,vv[i].v[0],vv[i].v[1],vv[i].v[2],vv[i].v[3],vv[j].h,vv[j].d,vv[j].v[0],vv[j].v[1],vv[j].v[2],vv[j].v[3],dist);}printf("]}\n");free(vv);close(f);return 0;}
