#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_MAPS 2048
#define MAX_RESULTS 4096

typedef struct {
  uint64_t start, end, file_offset;
  int readable, writable;
  char path[512];
} MapRange;

static int read_exact(int fd, uint64_t address, void *output, size_t size) {
  uint8_t *cursor = output; size_t done = 0;
  while (done < size) {
    ssize_t value = pread(fd, cursor + done, size - done, (off_t)(address + done));
    if (value <= 0) return 0;
    done += (size_t)value;
  }
  return 1;
}

static int parse_maps(int pid, MapRange maps[MAX_MAPS]) {
  char filename[64], line[2048];
  snprintf(filename, sizeof(filename), "/proc/%d/maps", pid);
  FILE *handle = fopen(filename, "r"); if (!handle) return -1;
  int count = 0;
  while (count < MAX_MAPS && fgets(line, sizeof(line), handle)) {
    unsigned long long start=0,end=0,offset=0; unsigned int ma=0,mi=0;
    unsigned long inode=0; char perms[8]={0}; int consumed=0;
    if (sscanf(line,"%llx-%llx %7s %llx %x:%x %lu %n",&start,&end,perms,&offset,&ma,&mi,&inode,&consumed)<7) continue;
    MapRange *out=&maps[count++]; memset(out,0,sizeof(*out));
    out->start=start; out->end=end; out->file_offset=offset;
    out->readable=perms[0]=='r'; out->writable=perms[1]=='w';
    char *name=line+consumed; while(*name==' '||*name=='\t') ++name;
    size_t length=strcspn(name,"\r\n"); if(length>=sizeof(out->path)) length=sizeof(out->path)-1;
    memcpy(out->path,name,length);
  }
  fclose(handle); return count;
}

static int in_range(const MapRange *maps,int count,uint64_t address,size_t size,int writable){
  if(address<0x10000||address+size<address) return 0;
  for(int i=0;i<count;++i) if(maps[i].readable&&(!writable||maps[i].writable)&&address>=maps[i].start&&address+size<=maps[i].end) return 1;
  return 0;
}

static int object_like(int fd,const MapRange *maps,int count,uint64_t pointer,uint64_t libg_min,uint64_t libg_max){
  return (pointer&7)==0 && pointer>libg_max &&
      in_range(maps,count,pointer,0x100,0);
}

int main(int argc,char **argv){
  if(argc!=2) return 2; int pid=atoi(argv[1]); if(pid<=0) return 2;
  MapRange maps[MAX_MAPS]; int map_count=parse_maps(pid,maps); if(map_count<=0) return 3;
  uint64_t libg_min=UINT64_MAX,libg_max=0;
  for(int i=0;i<map_count;++i) if(strstr(maps[i].path,"/libg.so")){
    if(maps[i].start<libg_min) libg_min=maps[i].start;
    if(maps[i].end>libg_max) libg_max=maps[i].end;
  }
  char mem_path[64]; snprintf(mem_path,sizeof(mem_path),"/proc/%d/mem",pid);
  int fd=open(mem_path,O_RDONLY|O_CLOEXEC); if(fd<0) return 4;
  printf("{\"event\":\"mumu_manager_scan\",\"pid\":%d,\"libg_min\":\"0x%" PRIx64 "\",\"libg_max\":\"0x%" PRIx64 "\",\"results\":[",pid,libg_min,libg_max);
  int emitted=0;
  for(int mi=0;mi<map_count && emitted<MAX_RESULTS;++mi){
    MapRange *map=&maps[mi];
    if(!map->readable||map->start<libg_min||map->end>libg_max) continue;
    size_t size=(size_t)(map->end-map->start); uint8_t *buffer=malloc(size); if(!buffer) continue;
    if(!read_exact(fd,map->start,buffer,size)){free(buffer);continue;}
    for(size_t off=0;off+8<=size && emitted<MAX_RESULTS;off+=8){
      uint64_t root=0; memcpy(&root,buffer+off,8);
      if(!object_like(fd,maps,map_count,root,libg_min,libg_max)) continue;
      for(int state_off=0x20;state_off<=0x20 && emitted<MAX_RESULTS;state_off+=8){
        uint64_t state=0; if(!read_exact(fd,root+state_off,&state,8)||!object_like(fd,maps,map_count,state,libg_min,libg_max)) continue;
        int32_t current_type=-1;
        if(!read_exact(fd,root+0x30,&current_type,4)||current_type<0||current_type>10) continue;
        int small_count=0; int small_offsets[16],small_values[16];
        for(int field=0;field<=0x100&&small_count<16;field+=4){
          int32_t value=0; if(read_exact(fd,root+field,&value,4)&&value>=0&&value<=10){small_offsets[small_count]=field;small_values[small_count++]=value;}
        }
        if(emitted++) putchar(',');
        printf("{\"global_address\":\"0x%" PRIx64 "\",\"global_rva_guess\":\"0x%" PRIx64 "\",\"root\":\"0x%" PRIx64 "\",\"state_offset\":%d,\"state\":\"0x%" PRIx64 "\",\"current_type\":%d,\"small_fields\":[",map->start+off,map->file_offset+off,root,state_off,state,current_type);
        for(int s=0;s<small_count;++s){if(s)putchar(',');printf("[%d,%d]",small_offsets[s],small_values[s]);}
        printf("]}");
      }
    }
    free(buffer);
  }
  printf("]}\n"); close(fd); return 0;
}
