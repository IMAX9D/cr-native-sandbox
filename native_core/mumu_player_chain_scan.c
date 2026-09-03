#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_MAPS 2048
#define MAX_NODES 20000
#define MAX_DEPTH 5

typedef struct { uint64_t start, end; int readable; char path[256]; } Map;
typedef struct { uint64_t address; int depth; uint16_t path[MAX_DEPTH]; } Node;
typedef struct {
  uint64_t address, hand_pointer, cycle_pointer;
  int32_t hand_capacity, hand_size, hand[4];
  int32_t cycle_capacity, cycle_size, cycle[8], deck_count, elixir_raw;
} Player;

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
    char *name = line + consumed;
    while (*name == ' ' || *name == '\t') ++name;
    size_t length = strcspn(name, "\r\n");
    if (length >= sizeof(map->path)) length = sizeof(map->path) - 1;
    memcpy(map->path, name, length);
  }
  fclose(handle);
  return count;
}

static int readable(const Map *maps, int count, uint64_t address, size_t size) {
  if (address < 0x10000 || address + size < address) return 0;
  for (int i = 0; i < count; ++i)
    if (maps[i].readable && address >= maps[i].start &&
        address + size <= maps[i].end)
      return 1;
  return 0;
}

static uint64_t libg_base(const Map *maps, int count) {
  uint64_t best = UINT64_MAX;
  for (int i = 0; i < count; ++i)
    if (strstr(maps[i].path, "/libg.so") && maps[i].start < best)
      best = maps[i].start;
  return best == UINT64_MAX ? 0 : best;
}

static int seen(const Node *nodes, int count, uint64_t address) {
  for (int i = 0; i < count; ++i)
    if (nodes[i].address == address) return 1;
  return 0;
}

static int valid_player(int fd, const Map *maps, int map_count,
                        uint64_t address, Player *out) {
  memset(out, 0, sizeof(*out));
  out->address = address;
  if (!readable(maps, map_count, address, 0x300) ||
      !read_exact(fd, address + 0x210, &out->hand_pointer, 8) ||
      !read_exact(fd, address + 0x218, &out->hand_capacity, 4) ||
      !read_exact(fd, address + 0x21c, &out->hand_size, 4) ||
      !read_exact(fd, address + 0x220, &out->cycle_pointer, 8) ||
      !read_exact(fd, address + 0x228, &out->cycle_capacity, 4) ||
      !read_exact(fd, address + 0x22c, &out->cycle_size, 4) ||
      !read_exact(fd, address + 0x230, &out->deck_count, 4) ||
      !read_exact(fd, address + 0x2f8, &out->elixir_raw, 4))
    return 0;
  if (out->hand_capacity < 4 || out->hand_capacity > 8 ||
      out->hand_size != 4 || out->cycle_capacity < 1 ||
      out->cycle_capacity > 8 || out->cycle_size < 1 ||
      out->cycle_size > out->cycle_capacity || out->deck_count != 8 ||
      out->elixir_raw < 0 || out->elixir_raw > 100000 ||
      !readable(maps, map_count, out->hand_pointer, 16) ||
      !readable(maps, map_count, out->cycle_pointer,
                (size_t)out->cycle_size * sizeof(int32_t)) ||
      !read_exact(fd, out->hand_pointer, out->hand, 16) ||
      !read_exact(fd, out->cycle_pointer, out->cycle,
                  (size_t)out->cycle_size * sizeof(int32_t)))
    return 0;
  for (int index = 0; index < 4; ++index)
    if (out->hand[index] < -1 || out->hand[index] >= 8) return 0;
  for (int index = 0; index < out->cycle_size; ++index)
    if (out->cycle[index] < 0 || out->cycle[index] >= 8) return 0;
  return 1;
}

static int player_pair(int fd, const Map *maps, int map_count, uint64_t state,
                       Player players[2]) {
  uint64_t addresses[2] = {0};
  return readable(maps, map_count, state, 0xf0) &&
         read_exact(fd, state + 0xe0, addresses, sizeof(addresses)) &&
         valid_player(fd, maps, map_count, addresses[0], &players[0]) &&
         valid_player(fd, maps, map_count, addresses[1], &players[1]);
}

static void emit_path(const Node *node) {
  putchar('[');
  for (int index = 0; index < node->depth; ++index) {
    if (index) putchar(',');
    printf("%u", node->path[index]);
  }
  putchar(']');
}

static void emit_player(const Player *player) {
  printf("{\"address\":\"0x%" PRIx64
         "\",\"elixir_raw\":%d,\"hand\":[%d,%d,%d,%d],\"cycle\":[",
         player->address, player->elixir_raw, player->hand[0], player->hand[1],
         player->hand[2], player->hand[3]);
  for (int index = 0; index < player->cycle_size; ++index) {
    if (index) putchar(',');
    printf("%d", player->cycle[index]);
  }
  printf("]}");
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  int pid = atoi(argv[1]);
  Map maps[MAX_MAPS];
  int map_count = load_maps(pid, maps);
  char memory_path[64];
  snprintf(memory_path, sizeof(memory_path), "/proc/%d/mem", pid);
  int fd = open(memory_path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 3;
  uint64_t base = libg_base(maps, map_count), manager = 0, state = 0, battle = 0;
  if (!base || !read_exact(fd, base + 0x1a569a8, &manager, 8) || !manager ||
      !read_exact(fd, manager + 0x28, &state, 8) || !state ||
      !read_exact(fd, state + 0x90, &battle, 8) || !battle)
    return 4;

  Node *nodes = calloc(MAX_NODES, sizeof(Node));
  int count = 1, emitted = 0;
  nodes[0].address = battle;
  printf("{\"event\":\"mumu_player_chain_scan\",\"battle\":\"0x%" PRIx64
         "\",\"results\":[", battle);
  for (int cursor = 0; cursor < count && cursor < MAX_NODES; ++cursor) {
    Node node = nodes[cursor];
    uint8_t raw[0x600];
    if (!read_exact(fd, node.address, raw, sizeof(raw))) continue;
    Player players[2];
    if (player_pair(fd, maps, map_count, node.address, players)) {
      if (emitted++) putchar(',');
      printf("{\"kind\":\"direct\",\"node\":\"0x%" PRIx64
             "\",\"path\":", node.address);
      emit_path(&node);
      printf(",\"players\":["); emit_player(&players[0]); putchar(',');
      emit_player(&players[1]); printf("]}");
    }
    uint64_t context = 0, indirect_state = 0;
    if (read_exact(fd, node.address + 0x10, &context, 8) && context &&
        read_exact(fd, context + 0x98, &indirect_state, 8) && indirect_state &&
        player_pair(fd, maps, map_count, indirect_state, players)) {
      if (emitted++) putchar(',');
      printf("{\"kind\":\"outer_context\",\"node\":\"0x%" PRIx64
             "\",\"context\":\"0x%" PRIx64
             "\",\"player_state\":\"0x%" PRIx64 "\",\"path\":",
             node.address, context, indirect_state);
      emit_path(&node);
      printf(",\"players\":["); emit_player(&players[0]); putchar(',');
      emit_player(&players[1]); printf("]}");
    }
    if (node.depth >= MAX_DEPTH) continue;
    for (int offset = 0; offset <= 0x5f8 && count < MAX_NODES; offset += 8) {
      uint64_t child = 0, probe = 0;
      memcpy(&child, raw + offset, 8);
      if ((child & 7) || !readable(maps, map_count, child, 8) ||
          !read_exact(fd, child, &probe, 8) || seen(nodes, count, child))
        continue;
      nodes[count] = node;
      nodes[count].address = child;
      nodes[count].path[node.depth] = (uint16_t)offset;
      nodes[count].depth = node.depth + 1;
      ++count;
    }
  }
  printf("],\"nodes_scanned\":%d}\n", count);
  free(nodes);
  close(fd);
  return 0;
}
