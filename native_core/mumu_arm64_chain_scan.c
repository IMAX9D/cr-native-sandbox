#define _GNU_SOURCE

#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_MAPS 2048
#define MAX_CHAINS 32768

typedef struct {
  uint64_t start;
  uint64_t end;
  uint64_t file_offset;
  int readable;
  int writable;
  char path[512];
} MapRange;

typedef struct {
  uint64_t global_address;
  uint64_t global_rva_guess;
  uint64_t root;
  uint64_t context;
  uint64_t battle;
  int root_context_offset;
  int context_battle_offset;
} Chain;

static int read_exact(int fd, uint64_t address, void *output, size_t size) {
  uint8_t *cursor = (uint8_t *)output;
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
  char path[64];
  char line[2048];
  snprintf(path, sizeof(path), "/proc/%d/maps", pid);
  FILE *handle = fopen(path, "r");
  if (!handle) return -1;
  int count = 0;
  while (count < MAX_MAPS && fgets(line, sizeof(line), handle)) {
    unsigned long long start = 0, end = 0, offset = 0;
    char permissions[8] = {0};
    unsigned int major = 0, minor = 0;
    unsigned long inode = 0;
    int consumed = 0;
    int fields = sscanf(line, "%llx-%llx %7s %llx %x:%x %lu %n",
                        &start, &end, permissions, &offset,
                        &major, &minor, &inode, &consumed);
    if (fields < 7) continue;
    MapRange *out = &maps[count++];
    memset(out, 0, sizeof(*out));
    out->start = (uint64_t)start;
    out->end = (uint64_t)end;
    out->file_offset = (uint64_t)offset;
    out->readable = permissions[0] == 'r';
    out->writable = permissions[1] == 'w';
    char *name = line + consumed;
    while (*name == ' ' || *name == '\t') ++name;
    size_t length = strcspn(name, "\r\n");
    if (length >= sizeof(out->path)) length = sizeof(out->path) - 1;
    memcpy(out->path, name, length);
    out->path[length] = 0;
  }
  fclose(handle);
  return count;
}

static int in_readable(const MapRange *maps, int count, uint64_t address,
                       size_t size) {
  if (address < 0x10000 || address + size < address) return 0;
  for (int index = 0; index < count; ++index) {
    if (maps[index].readable && address >= maps[index].start &&
        address + size <= maps[index].end)
      return 1;
  }
  return 0;
}

static int in_writable(const MapRange *maps, int count, uint64_t address,
                       size_t size) {
  if (address < 0x10000 || address + size < address) return 0;
  for (int index = 0; index < count; ++index) {
    if (maps[index].readable && maps[index].writable &&
        address >= maps[index].start && address + size <= maps[index].end)
      return 1;
  }
  return 0;
}

static int pointer_object(int fd, const MapRange *maps, int map_count,
                          uint64_t pointer) {
  uint64_t first = 0;
  if ((pointer & 7) != 0 || !in_readable(maps, map_count, pointer, 0x100) ||
      !read_exact(fd, pointer, &first, 8))
    return 0;
  return first != 0 && in_readable(maps, map_count, first, 4);
}

static int duplicate_chain(const Chain *chains, int count, uint64_t global,
                           uint64_t battle, int root_offset,
                           int battle_offset) {
  for (int index = 0; index < count; ++index)
    if (chains[index].global_address == global &&
        chains[index].battle == battle &&
        chains[index].root_context_offset == root_offset &&
        chains[index].context_battle_offset == battle_offset)
      return 1;
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: mumu-arm64-chain-scan PID DELAY_MS\n");
    return 2;
  }
  int pid = atoi(argv[1]);
  int delay_ms = atoi(argv[2]);
  if (pid <= 0 || delay_ms < 50 || delay_ms > 2000) return 2;
  MapRange maps[MAX_MAPS];
  int map_count = parse_maps(pid, maps);
  if (map_count <= 0) return 3;
  char memory_path[64];
  snprintf(memory_path, sizeof(memory_path), "/proc/%d/mem", pid);
  int fd = open(memory_path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 4;
  Chain *chains = calloc(MAX_CHAINS, sizeof(*chains));
  if (!chains) return 5;
  int chain_count = 0;
  uint64_t libg_min = UINT64_MAX, libg_max = 0;
  for (int index = 0; index < map_count; ++index) {
    if (strstr(maps[index].path, "/libg.so")) {
      if (maps[index].start < libg_min) libg_min = maps[index].start;
      if (maps[index].end > libg_max) libg_max = maps[index].end;
    }
  }
  for (int map_index = 0; map_index < map_count; ++map_index) {
    const MapRange *map = &maps[map_index];
    if (!map->readable || !map->writable || map->start < libg_min ||
        map->end > libg_max)
      continue;
    size_t size = (size_t)(map->end - map->start);
    uint8_t *buffer = malloc(size);
    if (!buffer) continue;
    if (!read_exact(fd, map->start, buffer, size)) {
      free(buffer);
      continue;
    }
    for (size_t offset = 0; offset + 8 <= size; offset += 8) {
      uint64_t root = 0;
      memcpy(&root, buffer + offset, 8);
      if (root <= libg_max || !in_writable(maps, map_count, root, 0x100) ||
          !pointer_object(fd, maps, map_count, root))
        continue;
      for (int root_offset = 0x20; root_offset <= 0x20; root_offset += 8) {
        uint64_t context = 0;
        if (!read_exact(fd, root + root_offset, &context, 8) ||
            context <= libg_max ||
            !in_writable(maps, map_count, context, 0x100) ||
            !pointer_object(fd, maps, map_count, context))
          continue;
        for (int battle_offset = 0x90; battle_offset <= 0x90;
             battle_offset += 8) {
          uint64_t battle = 0;
          if (!read_exact(fd, context + battle_offset, &battle, 8) ||
              battle <= libg_max || battle == root || battle == context ||
              !in_writable(maps, map_count, battle, 0x300) ||
              !pointer_object(fd, maps, map_count, battle))
            continue;
          uint64_t global = map->start + offset;
          if (duplicate_chain(chains, chain_count, global, battle,
                              root_offset, battle_offset))
            continue;
          if (chain_count >= MAX_CHAINS) break;
          Chain *out = &chains[chain_count++];
          out->global_address = global;
          out->global_rva_guess = map->file_offset + offset;
          out->root = root;
          out->context = context;
          out->battle = battle;
          out->root_context_offset = root_offset;
          out->context_battle_offset = battle_offset;
        }
      }
    }
    free(buffer);
  }

  const int scan_slots = 193;
  int32_t *before = calloc((size_t)chain_count * scan_slots, sizeof(*before));
  if (!before) return 6;
  for (int index = 0; index < chain_count; ++index)
    for (int slot = 0; slot < scan_slots; ++slot)
      read_exact(fd, chains[index].battle + slot * 4,
                 &before[index * scan_slots + slot], 4);
  usleep((useconds_t)delay_ms * 1000U);

  printf("{\"event\":\"mumu_arm64_chain_scan\",\"pid\":%d,"
         "\"delay_ms\":%d,\"raw_chain_count\":%d,\"raw_chains\":[",
         pid, delay_ms, chain_count);
  for (int index = 0; index < chain_count && index < 64; ++index) {
    const Chain *item = &chains[index];
    if (index) putchar(',');
    printf("{\"global_address\":\"0x%" PRIx64
           "\",\"global_rva_guess\":\"0x%" PRIx64
           "\",\"root\":\"0x%" PRIx64
           "\",\"context\":\"0x%" PRIx64
           "\",\"battle\":\"0x%" PRIx64 "\"}",
           item->global_address, item->global_rva_guess, item->root,
           item->context, item->battle);
  }
  printf("],\"candidates\":[");
  int emitted = 0;
  for (int index = 0; index < chain_count; ++index) {
    for (int slot = 0; slot < scan_slots; ++slot) {
      int32_t after = 0;
      if (!read_exact(fd, chains[index].battle + 0x10 + slot * 4,
                      &after, 4))
        continue;
      int32_t old = before[index * scan_slots + slot];
      int32_t delta = after - old;
      if (old < 50 || old > 10000000 || delta < 1 || delta > 20) continue;
      const Chain *item = &chains[index];
      if (emitted++) putchar(',');
      printf("{\"global_address\":\"0x%" PRIx64
             "\",\"global_rva_guess\":\"0x%" PRIx64
             "\",\"root\":\"0x%" PRIx64
             "\",\"context\":\"0x%" PRIx64
             "\",\"battle\":\"0x%" PRIx64
             "\",\"root_context_offset\":%d,"
             "\"context_battle_offset\":%d,\"tick_offset\":%d,"
             "\"before\":%d,\"after\":%d,\"delta\":%d}",
             item->global_address, item->global_rva_guess, item->root,
             item->context, item->battle, item->root_context_offset,
             item->context_battle_offset, slot * 4,
             old, after, delta);
    }
  }
  printf("]}\n");
  free(before);
  free(chains);
  close(fd);
  return 0;
}
