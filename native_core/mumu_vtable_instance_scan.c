#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_MAPS 2048
#define MAX_RESULTS 256

typedef struct { uint64_t start, end; int readable, writable; char path[256]; } Map;

static int read_exact(int fd, uint64_t address, void *output, size_t size) {
  uint8_t *cursor = output;
  size_t done = 0;
  while (done < size) {
    ssize_t value = pread(fd, cursor + done, size - done, (off_t)(address + done));
    if (value <= 0) return 0;
    done += (size_t)value;
  }
  return 1;
}

static int load_maps(int pid, Map *maps) {
  char path[64], line[1024];
  snprintf(path, sizeof(path), "/proc/%d/maps", pid);
  FILE *handle = fopen(path, "r");
  if (!handle) return -1;
  int count = 0;
  while (count < MAX_MAPS && fgets(line, sizeof(line), handle)) {
    unsigned long long start, end, offset;
    unsigned int major, minor;
    unsigned long inode;
    char permissions[8];
    int consumed = 0;
    if (sscanf(line, "%llx-%llx %7s %llx %x:%x %lu %n", &start, &end,
               permissions, &offset, &major, &minor, &inode, &consumed) < 7)
      continue;
    Map *map = &maps[count++];
    memset(map, 0, sizeof(*map));
    map->start = start;
    map->end = end;
    map->readable = permissions[0] == 'r';
    map->writable = permissions[1] == 'w';
    char *name = line + consumed;
    while (*name == ' ' || *name == '\t') ++name;
    size_t length = strcspn(name, "\r\n");
    if (length >= sizeof(map->path)) length = sizeof(map->path) - 1;
    memcpy(map->path, name, length);
  }
  fclose(handle);
  return count;
}

static int scan_map(const Map *map) {
  if (!map->readable || !map->writable) return 0;
  return strstr(map->path, "scudo:") != NULL ||
         strstr(map->path, "[anon:") != NULL ||
         map->end < 0x100000000ULL;
}

static void emit_values(int fd, uint64_t pointer, int count) {
  int32_t values[8] = {0};
  if (!pointer || count < 0 || count > 8 ||
      !read_exact(fd, pointer, values, (size_t)count * sizeof(values[0]))) {
    printf("null");
    return;
  }
  putchar('[');
  for (int i = 0; i < count; ++i) {
    if (i) putchar(',');
    printf("%d", values[i]);
  }
  putchar(']');
}

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  int pid = atoi(argv[1]);
  uint64_t target = strtoull(argv[2], NULL, 0);
  Map maps[MAX_MAPS];
  int map_count = load_maps(pid, maps);
  char memory_path[64];
  snprintf(memory_path, sizeof(memory_path), "/proc/%d/mem", pid);
  int fd = open(memory_path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 3;
  printf("{\"event\":\"mumu_vtable_instance_scan\",\"target\":\"0x%" PRIx64
         "\",\"instances\":[", target);
  int emitted = 0;
  for (int map_index = 0; map_index < map_count && emitted < MAX_RESULTS; ++map_index) {
    const Map *map = &maps[map_index];
    if (!scan_map(map)) continue;
    for (uint64_t start = map->start; start < map->end && emitted < MAX_RESULTS;
         start += 0x10000) {
      size_t size = (size_t)((map->end - start) < 0x10000
                                 ? (map->end - start)
                                 : 0x10000);
      uint8_t *buffer = malloc(size);
      if (!buffer || !read_exact(fd, start, buffer, size)) {
        free(buffer);
        continue;
      }
      for (size_t offset = 0; offset + 0x308 <= size && emitted < MAX_RESULTS;
           offset += 8) {
        uint64_t vtable = 0;
        memcpy(&vtable, buffer + offset, 8);
        if (vtable != target) continue;
        uint64_t address = start + offset, hand = 0, cycle = 0;
        int32_t hand_capacity = -1, hand_size = -1, cycle_capacity = -1;
        int32_t cycle_size = -1, deck_count = -1, elixir = -1;
        memcpy(&hand, buffer + offset + 0x210, 8);
        memcpy(&hand_capacity, buffer + offset + 0x218, 4);
        memcpy(&hand_size, buffer + offset + 0x21c, 4);
        memcpy(&cycle, buffer + offset + 0x220, 8);
        memcpy(&cycle_capacity, buffer + offset + 0x228, 4);
        memcpy(&cycle_size, buffer + offset + 0x22c, 4);
        memcpy(&deck_count, buffer + offset + 0x230, 4);
        memcpy(&elixir, buffer + offset + 0x2f8, 4);
        if (emitted++) putchar(',');
        printf("{\"address\":\"0x%" PRIx64
               "\",\"hand_ptr\":\"0x%" PRIx64
               "\",\"hand_capacity\":%d,\"hand_size\":%d,\"hand\":",
               address, hand, hand_capacity, hand_size);
        emit_values(fd, hand, hand_size);
        printf(",\"cycle_ptr\":\"0x%" PRIx64
               "\",\"cycle_capacity\":%d,\"cycle_size\":%d,\"cycle\":",
               cycle, cycle_capacity, cycle_size);
        emit_values(fd, cycle, cycle_size);
        printf(",\"deck_count\":%d,\"elixir_raw\":%d,\"map\":\"%s\"}",
               deck_count, elixir, map->path);
      }
      free(buffer);
    }
  }
  printf("]}\n");
  close(fd);
  return 0;
}
