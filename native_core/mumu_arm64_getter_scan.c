#define _GNU_SOURCE
#include <fcntl.h>
#include <elf.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_MAPS 2048
#define MAX_RESULTS 8192

typedef struct { uint64_t start,end,file_offset; int readable,writable,executable; char path[512]; } MapRange;

static int read_exact(int fd,uint64_t address,void *output,size_t size){uint8_t *p=output;size_t n=0;while(n<size){ssize_t v=pread(fd,p+n,size-n,(off_t)(address+n));if(v<=0)return 0;n+=(size_t)v;}return 1;}

static int parse_maps(int pid,MapRange maps[MAX_MAPS]){char f[64],line[2048];snprintf(f,sizeof(f),"/proc/%d/maps",pid);FILE*h=fopen(f,"r");if(!h)return-1;int c=0;while(c<MAX_MAPS&&fgets(line,sizeof(line),h)){unsigned long long s=0,e=0,o=0;unsigned int ma=0,mi=0;unsigned long ino=0;char p[8]={0};int used=0;if(sscanf(line,"%llx-%llx %7s %llx %x:%x %lu %n",&s,&e,p,&o,&ma,&mi,&ino,&used)<7)continue;MapRange*out=&maps[c++];memset(out,0,sizeof(*out));out->start=s;out->end=e;out->file_offset=o;out->readable=p[0]=='r';out->writable=p[1]=='w';out->executable=p[2]=='x';char*n=line+used;while(*n==' '||*n=='\t')++n;size_t l=strcspn(n,"\r\n");if(l>=sizeof(out->path))l=sizeof(out->path)-1;memcpy(out->path,n,l);}fclose(h);return c;}

static const MapRange *find_map(const MapRange*maps,int count,uint64_t address,size_t size){for(int i=0;i<count;++i)if(maps[i].readable&&address>=maps[i].start&&address+size<=maps[i].end)return&maps[i];return NULL;}

static int64_t sign_extend(uint64_t value,int bits){uint64_t sign=1ULL<<(bits-1);return(int64_t)((value^sign)-sign);}

static uint64_t runtime_for_rva(int fd,const MapRange*maps,int map_count,uint64_t module_min,uint64_t rva){
  Elf64_Ehdr header; if(!read_exact(fd,module_min,&header,sizeof(header))||memcmp(header.e_ident,ELFMAG,SELFMAG)!=0||header.e_phnum>64)return 0;
  Elf64_Phdr phdrs[64]; if(!read_exact(fd,module_min+header.e_phoff,phdrs,(size_t)header.e_phnum*sizeof(Elf64_Phdr)))return 0;
  for(int p=0;p<header.e_phnum;++p){Elf64_Phdr*ph=&phdrs[p];if(ph->p_type!=PT_LOAD)continue;uint64_t vpage=ph->p_vaddr&~0xfffULL;uint64_t opage=ph->p_offset&~0xfffULL;uint64_t vend=(ph->p_vaddr+ph->p_memsz+0xfffULL)&~0xfffULL;if(rva<vpage||rva>=vend)continue;for(int m=0;m<map_count;++m)if(strstr(maps[m].path,"/libg.so")&&maps[m].file_offset==opage)return maps[m].start+(rva-vpage);}
  return 0;
}

int main(int argc,char**argv){if(argc!=2)return 2;int pid=atoi(argv[1]);MapRange maps[MAX_MAPS];int count=parse_maps(pid,maps);if(count<=0)return 3;uint64_t module_min=UINT64_MAX,module_max=0;for(int i=0;i<count;++i)if(strstr(maps[i].path,"/libg.so")){if(maps[i].start<module_min)module_min=maps[i].start;if(maps[i].end>module_max)module_max=maps[i].end;}char mem[64];snprintf(mem,sizeof(mem),"/proc/%d/mem",pid);int fd=open(mem,O_RDONLY|O_CLOEXEC);if(fd<0)return 4;printf("{\"event\":\"mumu_arm64_getter_scan\",\"pid\":%d,\"module_min\":\"0x%" PRIx64 "\",\"results\":[",pid,module_min);int emitted=0;
  for(int m=0;m<count&&emitted<MAX_RESULTS;++m){MapRange*map=&maps[m];if(!map->readable||!map->executable||!strstr(map->path,"/libg.so"))continue;size_t size=(size_t)(map->end-map->start);uint8_t*buf=malloc(size);if(!buf||!read_exact(fd,map->start,buf,size)){free(buf);continue;}for(size_t off=0;off+12<=size&&emitted<MAX_RESULTS;off+=4){uint32_t a=0,l=0,r=0;memcpy(&a,buf+off,4);memcpy(&l,buf+off+4,4);memcpy(&r,buf+off+8,4);if((a&0x9F000000U)!=0x90000000U||(l&0xFFC00000U)!=0xF9400000U||r!=0xD65F03C0U)continue;int rd=(int)(a&31),rn=(int)((l>>5)&31),rt=(int)(l&31);if(rn!=rd||rt!=0)continue;uint64_t immlo=(a>>29)&3,immhi=(a>>5)&0x7ffff;int64_t pages=sign_extend((immhi<<2)|immlo,21);uint64_t function_rva=map->file_offset+off;int64_t target_page=(int64_t)(function_rva&~0xfffULL)+pages*4096LL;uint64_t global_rva=(uint64_t)(target_page+(int64_t)(((l>>10)&0xfff)*8ULL));uint64_t global=runtime_for_rva(fd,maps,count,module_min,global_rva);const MapRange*gm=find_map(maps,count,global,8);if(!global||!gm)continue;uint64_t root=0;if(!read_exact(fd,global,&root,8)||!root||!find_map(maps,count,root,0x20))continue;if(emitted++)putchar(',');uint64_t pc=map->start+off;printf("{\"function_address\":\"0x%" PRIx64 "\",\"function_rva_guess\":\"0x%" PRIx64 "\",\"global_address\":\"0x%" PRIx64 "\",\"global_rva_guess\":\"0x%" PRIx64 "\",\"root\":\"0x%" PRIx64 "\",\"pointer_fields\":[",pc,function_rva,global,global_rva,root);int first=1;for(int f=0;f<=0x100;f+=8){uint64_t value=0;if(read_exact(fd,root+f,&value,8)&&value&&find_map(maps,count,value,8)){if(!first)putchar(',');first=0;printf("[%d,\"0x%" PRIx64 "\"]",f,value);}}printf("],\"small_fields\":[");first=1;for(int f=0;f<=0x100;f+=4){int32_t value=0;if(read_exact(fd,root+f,&value,4)&&value>=0&&value<=16){if(!first)putchar(',');first=0;printf("[%d,%d]",f,value);}}printf("]}");}free(buf);}printf("]}\n");close(fd);return 0;}
