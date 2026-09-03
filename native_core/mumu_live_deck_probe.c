#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  int pid = atoi(argv[1]);
  uint64_t player = strtoull(argv[2], NULL, 0);
  char memory_path[64];
  snprintf(memory_path, sizeof(memory_path), "/proc/%d/mem", pid);
  int fd = open(memory_path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 3;
  uint64_t context = 0, root = 0, avatar = 0, owner = 0, vtable = 0;
  uint64_t virtual_deck_entry = 0;
  int32_t side = -1, count = -1, identity_index = -1;
  int32_t account_hi = 0, account_lo = 0;
  if (!read_exact(fd, player + 0x10, &context, 8) || !context ||
      !read_exact(fd, context + 0x98, &root, 8) || !root ||
      !read_exact(fd, player + 0x78, &side, 4) || side < 0 || side > 5 ||
      !read_exact(fd, root + 0x30 + (uint64_t)side * 8, &avatar, 8) || !avatar ||
      !read_exact(fd, avatar, &account_hi, 4) ||
      !read_exact(fd, avatar + 4, &account_lo, 4) ||
      !read_exact(fd, root + 0x60, &count, 4) || count < 1 || count > 6)
    return 4;
  for (int32_t index = 0; index < count; ++index) {
    uint64_t candidate = 0;
    int32_t hi = 0, lo = 0;
    if (!read_exact(fd, root + 0x30 + (uint64_t)index * 8, &candidate, 8) ||
        !candidate || !read_exact(fd, candidate, &hi, 4) ||
        !read_exact(fd, candidate + 4, &lo, 4))
      continue;
    if (hi == account_hi && lo == account_lo) {
      identity_index = index;
      break;
    }
  }
  if (identity_index < 0 ||
      !read_exact(fd, root + 0x88 + (uint64_t)identity_index * 8, &owner, 8) ||
      !owner || !read_exact(fd, owner, &vtable, 8) || !vtable ||
      !read_exact(fd, vtable + 0x38, &virtual_deck_entry, 8) ||
      !virtual_deck_entry)
    return 5;
  uint64_t fields[128] = {0};
  if (!read_exact(fd, owner, fields, sizeof(fields))) return 6;
  printf("{\"event\":\"mumu_live_deck_probe\",\"player\":\"0x%" PRIx64
         "\",\"side\":%d,\"context\":\"0x%" PRIx64
         "\",\"root\":\"0x%" PRIx64
         "\",\"account\":[%d,%d],\"identity_index\":%d,"
         "\"owner\":\"0x%" PRIx64 "\",\"vtable\":\"0x%" PRIx64
         "\",\"virtual_deck_entry\":\"0x%" PRIx64 "\",\"fields\":[",
         player, side, context, root, account_hi, account_lo, identity_index,
         owner, vtable, virtual_deck_entry);
  for (int index = 0; index < 128; ++index) {
    if (index) putchar(',');
    printf("\"0x%" PRIx64 "\"", fields[index]);
  }
  printf("]}\n");
  close(fd);
  return 0;
}
