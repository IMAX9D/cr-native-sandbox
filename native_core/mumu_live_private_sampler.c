#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
  BATTLE_TICK = 0x60,
  PLAYER_TABLE = 0xE0,
  PLAYER_HAND_VECTOR = 0x210,
  PLAYER_HAND_CAPACITY = 0x218,
  PLAYER_HAND_SIZE = 0x21C,
  PLAYER_CYCLE_VECTOR = 0x220,
  PLAYER_CYCLE_CAPACITY = 0x228,
  PLAYER_CYCLE_SIZE = 0x22C,
  PLAYER_DECK_COUNT = 0x230,
  PLAYER_ELIXIR_RAW = 0x2F8,
  MAX_DISCOVERY_NODES = 20000,
  MAX_DISCOVERY_DEPTH = 5,
};

typedef struct {
  uint64_t address;
  int32_t elixir_raw;
  int32_t refill_timer;
  int32_t next_deck_index;
  int32_t hand[4];
  int32_t hand_size;
  int32_t cycle[8];
  int32_t cycle_size;
  int32_t deck_count;
  int32_t deck_card_ids[8];
  int32_t deck_form_flags[8];
  int32_t deck_visible;
} PlayerFrame;

typedef struct {
  uint64_t address;
  int depth;
} DiscoveryNode;

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

static uint64_t monotonic_us(void) {
  struct timespec value;
  clock_gettime(CLOCK_MONOTONIC, &value);
  return (uint64_t)value.tv_sec * 1000000ULL +
         (uint64_t)value.tv_nsec / 1000ULL;
}

static uint64_t find_libg_base(int pid) {
  char maps_path[64];
  char line[1024];
  uint64_t best = UINT64_MAX;
  snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", pid);
  FILE *maps = fopen(maps_path, "r");
  if (!maps) return 0;
  while (fgets(line, sizeof(line), maps)) {
    unsigned long long start = 0;
    unsigned long long offset = 0;
    char permissions[8] = {0};
    if (!strstr(line, "/libg.so")) continue;
    if (sscanf(line, "%llx-%*llx %7s %llx", &start, permissions, &offset) !=
        3)
      continue;
    if ((uint64_t)start >= (uint64_t)offset &&
        (uint64_t)start - (uint64_t)offset < best)
      best = (uint64_t)start - (uint64_t)offset;
  }
  fclose(maps);
  return best == UINT64_MAX ? 0 : best;
}

static int valid_index(int32_t value, int32_t deck_count, int allow_empty) {
  if (allow_empty && value == -1) return 1;
  return value >= 0 && value < deck_count;
}

static int read_player(int fd, uint64_t player, PlayerFrame *out) {
  uint64_t hand_vector = 0, cycle_vector = 0;
  int32_t hand_capacity = -1, cycle_capacity = -1;
  memset(out, 0, sizeof(*out));
  out->address = player;
  out->refill_timer = 0;
  out->deck_count = 8;
  out->next_deck_index = -1;
  for (int index = 0; index < 4; ++index) out->hand[index] = -1;
  if (!player ||
      !read_exact(fd, player + PLAYER_ELIXIR_RAW, &out->elixir_raw, 4) ||
      !read_exact(fd, player + PLAYER_HAND_VECTOR, &hand_vector, 8) ||
      !read_exact(fd, player + PLAYER_HAND_CAPACITY, &hand_capacity, 4) ||
      !read_exact(fd, player + PLAYER_HAND_SIZE, &out->hand_size, 4) ||
      !read_exact(fd, player + PLAYER_CYCLE_VECTOR, &cycle_vector, 8) ||
      !read_exact(fd, player + PLAYER_CYCLE_CAPACITY, &cycle_capacity, 4) ||
      !read_exact(fd, player + PLAYER_CYCLE_SIZE, &out->cycle_size, 4))
    return 0;
  if (out->elixir_raw < 0 || out->elixir_raw > 100000 ||
      hand_capacity < 4 || hand_capacity > 8 || out->hand_size != 4 ||
      cycle_capacity < 0 || cycle_capacity > 8 ||
      out->cycle_size < 0 || out->cycle_size > cycle_capacity ||
      (out->hand_size && !hand_vector) ||
      (out->cycle_size && !cycle_vector))
    return 0;
  if (out->hand_size &&
      !read_exact(fd, hand_vector, out->hand,
                  (size_t)out->hand_size * sizeof(out->hand[0])))
    return 0;
  if (out->cycle_size &&
      !read_exact(fd, cycle_vector, out->cycle,
                  (size_t)out->cycle_size * sizeof(out->cycle[0])))
    return 0;
  for (int index = 0; index < out->hand_size; ++index)
    if (!valid_index(out->hand[index], out->deck_count, 1)) return 0;
  for (int index = 0; index < out->cycle_size; ++index)
    if (!valid_index(out->cycle[index], out->deck_count, 0)) return 0;
  if (out->cycle_size > 0) out->next_deck_index = out->cycle[0];
  return 1;
}

static int read_visible_deck(int fd, PlayerFrame *player) {
  int visible = 0;
  for (int index = 0; index < 4; ++index)
    if (player->hand[index] >= 0) visible = 1;
  if (!visible) return 1;
  uint64_t context = 0, root = 0, avatar = 0, owner = 0, entries = 0;
  int32_t side = -1, count = -1, identity = -1;
  int32_t account_hi = 0, account_lo = 0, entry_count = -1;
  if (!read_exact(fd, player->address + 0x10, &context, 8) || !context ||
      !read_exact(fd, context + 0x98, &root, 8) || !root ||
      !read_exact(fd, player->address + 0x78, &side, 4) ||
      side < 0 || side > 5 ||
      !read_exact(fd, root + 0x30 + (uint64_t)side * 8, &avatar, 8) ||
      !avatar || !read_exact(fd, avatar, &account_hi, 4) ||
      !read_exact(fd, avatar + 4, &account_lo, 4) ||
      !read_exact(fd, root + 0x60, &count, 4) || count < 1 || count > 6)
    return 0;
  for (int index = 0; index < count; ++index) {
    uint64_t candidate = 0;
    int32_t hi = 0, lo = 0;
    if (!read_exact(fd, root + 0x30 + (uint64_t)index * 8,
                    &candidate, 8) || !candidate ||
        !read_exact(fd, candidate, &hi, 4) ||
        !read_exact(fd, candidate + 4, &lo, 4))
      continue;
    if (hi == account_hi && lo == account_lo) {
      identity = index;
      break;
    }
  }
  if (identity < 0 ||
      !read_exact(fd, root + 0x88 + (uint64_t)identity * 8, &owner, 8) ||
      !owner || !read_exact(fd, owner + 0x20, &entries, 8) || !entries ||
      !read_exact(fd, owner + 0x2c, &entry_count, 4) || entry_count != 8)
    return 0;
  for (int index = 0; index < 8; ++index) {
    uint64_t entry = 0, data = 0;
    if (!read_exact(fd, entries + (uint64_t)index * 8, &entry, 8) || !entry ||
        !read_exact(fd, entry + 0x10, &data, 8) || !data ||
        !read_exact(fd, data + 0x40, &player->deck_card_ids[index], 4) ||
        !read_exact(fd, entry + 0x1c, &player->deck_form_flags[index], 4))
      return 0;
    int32_t card_id = player->deck_card_ids[index];
    int32_t form = player->deck_form_flags[index];
    if (card_id < 25000000 || card_id > 29999999 || form < 0 || form > 2)
      return 0;
  }
  player->deck_visible = 1;
  return 1;
}

static int read_player_pair(int fd, uint64_t player_state,
                            PlayerFrame players[2]) {
  uint64_t addresses[2] = {0, 0};
  return player_state &&
         read_exact(fd, player_state + PLAYER_TABLE, addresses,
                    sizeof(addresses)) &&
         read_player(fd, addresses[0], &players[0]) &&
         read_player(fd, addresses[1], &players[1]) &&
         read_visible_deck(fd, &players[0]) &&
         read_visible_deck(fd, &players[1]);
}

static int seen_insert(uint64_t *table, size_t capacity, uint64_t value) {
  size_t index = (size_t)((value >> 3) * 11400714819323198485ull) &
                 (capacity - 1);
  for (size_t attempt = 0; attempt < capacity; ++attempt) {
    if (table[index] == value) return 0;
    if (table[index] == 0) {
      table[index] = value;
      return 1;
    }
    index = (index + 1) & (capacity - 1);
  }
  return 0;
}

static uint64_t discover_player_state(int fd, uint64_t battle,
                                      PlayerFrame players[2]) {
  enum { SEEN_CAPACITY = 32768 };
  DiscoveryNode *nodes = calloc(MAX_DISCOVERY_NODES, sizeof(*nodes));
  uint64_t *seen = calloc(SEEN_CAPACITY, sizeof(*seen));
  if (!nodes || !seen) {
    free(nodes);
    free(seen);
    return 0;
  }
  int count = 1;
  nodes[0].address = battle;
  seen_insert(seen, SEEN_CAPACITY, battle);
  for (int cursor = 0; cursor < count; ++cursor) {
    DiscoveryNode node = nodes[cursor];
    if (read_player_pair(fd, node.address, players)) {
      free(nodes);
      free(seen);
      return node.address;
    }
    uint64_t context = 0, indirect_state = 0;
    if (read_exact(fd, node.address + 0x10, &context, 8) && context &&
        read_exact(fd, context + 0x98, &indirect_state, 8) &&
        read_player_pair(fd, indirect_state, players)) {
      free(nodes);
      free(seen);
      return indirect_state;
    }
    if (node.depth >= MAX_DISCOVERY_DEPTH) continue;
    uint8_t raw[0x600];
    if (!read_exact(fd, node.address, raw, sizeof(raw))) continue;
    for (size_t offset = 0;
         offset + sizeof(uint64_t) <= sizeof(raw) &&
         count < MAX_DISCOVERY_NODES;
         offset += sizeof(uint64_t)) {
      uint64_t child = 0, probe = 0;
      memcpy(&child, raw + offset, sizeof(child));
      if (child < 0x100000000ull || (child & 7) ||
          !seen_insert(seen, SEEN_CAPACITY, child) ||
          !read_exact(fd, child, &probe, sizeof(probe)))
        continue;
      nodes[count].address = child;
      nodes[count].depth = node.depth + 1;
      ++count;
    }
  }
  free(nodes);
  free(seen);
  return 0;
}

static int read_frame(int fd, uint64_t libg, uint64_t manager_rva,
                      uint64_t root_context_offset, int32_t *tick,
                      int *coherent, PlayerFrame players[2],
                      uint64_t *cached_battle, uint64_t *cached_player_state,
                      uint64_t *last_discovery_us) {
  uint64_t root = 0, context = 0, battle = 0;
  int32_t before = -1, after = -1;
  *tick = -1;
  *coherent = 1;
  if (!read_exact(fd, libg + manager_rva, &root, 8) || !root ||
      !read_exact(fd, root + root_context_offset, &context, 8) || !context ||
      !read_exact(fd, context + 0x90, &battle, 8) || !battle ||
      !read_exact(fd, battle + BATTLE_TICK, &before, 4) || before < 0)
    return 0;
  if (*cached_battle != battle) {
    *cached_battle = battle;
    *cached_player_state = 0;
    *last_discovery_us = 0;
  }
  if (!read_player_pair(fd, *cached_player_state, players)) {
    *cached_player_state = 0;
    uint64_t now = monotonic_us();
    if (*last_discovery_us && now - *last_discovery_us < 500000) return 0;
    *last_discovery_us = now;
    *cached_player_state = discover_player_state(fd, battle, players);
    if (!*cached_player_state) return 0;
  }
  if (!read_exact(fd, battle + BATTLE_TICK, &after, 4)) return 0;
  *tick = after;
  *coherent = before == after;
  return 1;
}

static void emit_player(const PlayerFrame *player, int side) {
  printf("{\"side\":%d,\"address\":\"0x%" PRIx64
         "\",\"elixir_raw\":%d,\"refill_timer\":%d,"
         "\"next_deck_index\":%d,\"hand_deck_indices\":[%d,%d,%d,%d],"
         "\"cycle_deck_indices\":[",
         side, player->address, player->elixir_raw, player->refill_timer,
         player->next_deck_index, player->hand[0], player->hand[1],
         player->hand[2], player->hand[3]);
  for (int index = 0; index < player->cycle_size; ++index) {
    if (index) putchar(',');
    printf("%d", player->cycle[index]);
  }
  printf("],\"deck_card_ids\":[");
  if (player->deck_visible) {
    for (int index = 0; index < 8; ++index) {
      if (index) putchar(',');
      printf("%d", player->deck_card_ids[index]);
    }
  }
  printf("],\"deck_form_flags\":[");
  if (player->deck_visible) {
    for (int index = 0; index < 8; ++index) {
      if (index) putchar(',');
      printf("%d", player->deck_form_flags[index]);
    }
  }
  printf("]}");
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr,
            "usage: mumu-live-private PID INTERVAL_MS MANAGER_RVA "
            "ROOT_CONTEXT_OFFSET\n");
    return 2;
  }
  int pid = atoi(argv[1]);
  int interval_ms = atoi(argv[2]);
  uint64_t manager_rva = strtoull(argv[3], NULL, 0);
  uint64_t root_context_offset = strtoull(argv[4], NULL, 0);
  if (pid <= 0 || interval_ms < 20 || interval_ms > 5000 || !manager_rva ||
      !root_context_offset)
    return 2;
  uint64_t libg = find_libg_base(pid);
  if (!libg) return 3;
  char memory_path[64];
  snprintf(memory_path, sizeof(memory_path), "/proc/%d/mem", pid);
  int fd = open(memory_path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 4;
  setvbuf(stdout, NULL, _IONBF, 0);
  uint64_t sequence = 0;
  uint64_t cached_battle = 0, cached_player_state = 0;
  uint64_t last_discovery_us = 0;
  PlayerFrame players[2];
  while (kill(pid, 0) == 0 || errno == EPERM) {
    uint64_t started = monotonic_us();
    int32_t tick = -1;
    int coherent = 1;
    int active = 0;
    for (int attempt = 0; attempt < 3; ++attempt) {
      active = read_frame(fd, libg, manager_rva, root_context_offset, &tick,
                          &coherent, players, &cached_battle,
                          &cached_player_state, &last_discovery_us);
      if (!active || coherent) break;
    }
    printf("{\"event\":\"mumu_live_private\",\"sequence\":%" PRIu64
           ",\"battle_active\":%s,\"coherent\":%s,\"game_tick\":%d,"
           "\"players\":[",
           sequence++, active ? "true" : "false",
           coherent ? "true" : "false", tick);
    if (active) {
      emit_player(&players[0], 0);
      putchar(',');
      emit_player(&players[1], 1);
    }
    printf("],\"read_us\":%" PRIu64 "}\n", monotonic_us() - started);
    uint64_t elapsed = monotonic_us() - started;
    uint64_t target = (uint64_t)interval_ms * 1000ULL;
    if (elapsed < target) usleep((useconds_t)(target - elapsed));
  }
  close(fd);
  return 0;
}
