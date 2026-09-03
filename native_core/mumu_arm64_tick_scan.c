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
#define MAX_CHUNKS 8192
#define CHUNK_SIZE (64 * 1024)

typedef struct {
  uint64_t start, end;
  int readable, writable;
  char path[512];
} MapRange;

typedef struct {
  uint64_t address, object, vtable;
  int32_t before, after;
  int offset;
  char map[96];
} Result;

typedef struct {
  uint64_t start;
  size_t size;
  uint8_t *before;
  char map[96];
} Chunk;

static int read_exact(int fd, uint64_t address, void *output, size_t size) {
  uint8_t *cursor = output;
  size_t done = 0;
  while (done < size) {
    ssize_t value = pread(fd, cursor + done, size - done,
                          (off_t)(address + done));
    if (value <= 0) return 0;
    done += (size_t)value;
  }
  return 1;
}

static int parse_maps(int pid, MapRange maps[MAX_MAPS]) {
  char filename[64], line[2048];
  snprintf(filename, sizeof(filename), "/proc/%d/maps", pid);
  FILE *handle = fopen(filename, "r");
  if (!handle) return -1;
  int count = 0;
  while (count < MAX_MAPS && fgets(line, sizeof(line), handle)) {
    unsigned long long start = 0, end = 0, offset = 0;
    unsigned int major = 0, minor = 0;
    unsigned long inode = 0;
    char permissions[8] = {0};
    int consumed = 0;
    if (sscanf(line, "%llx-%llx %7s %llx %x:%x %lu %n", &start, &end,
               permissions, &offset, &major, &minor, &inode, &consumed) < 7)
      continue;
    MapRange *out = &maps[count++];
    memset(out, 0, sizeof(*out));
    out->start = start;
    out->end = end;
    out->readable = permissions[0] == 'r';
    out->writable = permissions[1] == 'w';
    char *name = line + consumed;
    while (*name == ' ' || *name == '\t') ++name;
    size_t length = strcspn(name, "\r\n");
    if (length >= sizeof(out->path)) length = sizeof(out->path) - 1;
    memcpy(out->path, name, length);
  }
  fclose(handle);
  return count;
}

static int duplicate(const Result *results, int count, uint64_t address,
                     uint64_t object) {
  for (int index = 0; index < count; ++index)
    if (results[index].address == address && results[index].object == object)
      return 1;
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: mumu-arm64-tick-scan PID DELAY_MS\n");
    return 2;
  }
  int pid = atoi(argv[1]);
  int delay_ms = atoi(argv[2]);
  if (pid <= 0 || delay_ms < 100 || delay_ms > 1000) return 2;
  MapRange maps[MAX_MAPS];
  int map_count = parse_maps(pid, maps);
  if (map_count <= 0) return 3;
  uint64_t libg_min = UINT64_MAX, libg_max = 0;
  for (int index = 0; index < map_count; ++index) {
    if (strstr(maps[index].path, "/libg.so")) {
      if (maps[index].start < libg_min) libg_min = maps[index].start;
      if (maps[index].end > libg_max) libg_max = maps[index].end;
    }
  }
  char memory_path[64];
  snprintf(memory_path, sizeof(memory_path), "/proc/%d/mem", pid);
  int fd = open(memory_path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 4;
  Result results[MAX_RESULTS];
  Chunk chunks[MAX_CHUNKS];
  int result_count = 0, chunk_count = 0;
  uint64_t bytes_scanned = 0;
  for (int map_index = 0; map_index < map_count; ++map_index) {
    MapRange *map = &maps[map_index];
    int scudo = strstr(map->path, "scudo:") != NULL;
    int low_native = map->end < 0x100000000ULL &&
        (map->path[0] == 0 || strstr(map->path, "Mem_"));
    if (!map->readable || !map->writable || (!scudo && !low_native) ||
        map->end <= map->start)
      continue;
    for (uint64_t start = map->start; start < map->end &&
         chunk_count < MAX_CHUNKS; start += CHUNK_SIZE) {
      size_t size = (size_t)((map->end - start) < CHUNK_SIZE
          ? (map->end - start) : CHUNK_SIZE);
      uint8_t *before = malloc(size);
      if (!before || !read_exact(fd, start, before, size)) {
        free(before);
        continue;
      }
      Chunk *chunk = &chunks[chunk_count++];
      chunk->start = start;
      chunk->size = size;
      chunk->before = before;
      snprintf(chunk->map, sizeof(chunk->map), "%s", map->path);
    }
  }
  usleep((useconds_t)delay_ms * 1000U);
  for (int chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
    Chunk *chunk = &chunks[chunk_index];
    size_t size = chunk->size;
    uint8_t *before = chunk->before;
    uint8_t *after = malloc(size);
    if (!after || !read_exact(fd, chunk->start, after, size)) {
      free(after); free(before); chunk->before = NULL; continue;
    }
    bytes_scanned += size;
    for (size_t offset = 0; offset + 4 <= size; offset += 4) {
      int32_t old = 0, now = 0;
      memcpy(&old, before + offset, 4);
      memcpy(&now, after + offset, 4);
      int32_t delta = now - old;
      if (old < 0 || old > 100000000 || delta < 150 || delta > 400) continue;
      uint64_t address = chunk->start + offset;
      int matched_object = 0;
      for (int field_offset = 0; field_offset <= 0x300; field_offset += 4) {
        if (address < (uint64_t)field_offset) continue;
        uint64_t object = address - (uint64_t)field_offset;
        if ((object & 7) != 0 || object < chunk->start ||
            object + 8 > chunk->start + size)
          continue;
        uint64_t vtable = 0;
        memcpy(&vtable, before + (object - chunk->start), 8);
        if (vtable < libg_min || vtable >= libg_max) continue;
        matched_object = 1;
        if (duplicate(results, result_count, address, object)) continue;
        if (result_count >= MAX_RESULTS) break;
        Result *out = &results[result_count++];
        memset(out, 0, sizeof(*out));
        out->address = address;
        out->object = object;
        out->vtable = vtable;
        out->before = old;
        out->after = now;
        out->offset = field_offset;
        snprintf(out->map, sizeof(out->map), "%s", chunk->map);
      }
      if (!matched_object && result_count < MAX_RESULTS) {
        Result *out = &results[result_count++];
        memset(out, 0, sizeof(*out));
        out->address = address;
        out->before = old;
        out->after = now;
        out->offset = -1;
        snprintf(out->map, sizeof(out->map), "%s", chunk->map);
      }
    }
    free(before);
    free(after);
    chunk->before = NULL;
  }
  printf("{\"event\":\"mumu_arm64_tick_scan\",\"pid\":%d,"
         "\"delay_ms\":%d,\"bytes_scanned\":%" PRIu64
         ",\"result_count\":%d,\"results\":[",
         pid, delay_ms, bytes_scanned, result_count);
  for (int index = 0; index < result_count; ++index) {
    Result *item = &results[index];
    if (index) putchar(',');
    printf("{\"address\":\"0x%" PRIx64 "\",\"object\":\"0x%" PRIx64
           "\",\"vtable\":\"0x%" PRIx64
           "\",\"field_offset\":%d,\"before\":%d,\"after\":%d,"
           "\"delta\":%d,\"map\":\"%s\"}",
           item->address, item->object, item->vtable, item->offset,
           item->before, item->after, item->after - item->before, item->map);
  }
  printf("]}\n");
  close(fd);
  return 0;
}
