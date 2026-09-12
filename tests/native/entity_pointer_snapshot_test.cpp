#include "entity_pointer_snapshot.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

struct FakeReader {
  mutable int scalar_calls = 0;
  mutable int bulk_calls = 0;
  bool short_bulk = false;
  uint64_t values[3] = {0x2000, 0x3000, 0x4000};

  bool read(uintptr_t address, uint64_t* value) const {
    ++scalar_calls;
    if (address < 0x1000 || address >= 0x1018 || (address - 0x1000) % 8) return false;
    *value = values[(address - 0x1000) / 8];
    return true;
  }

  bool read_bytes(uintptr_t address, void* output, std::size_t size) const {
    ++bulk_calls;
    if (address != 0x1000 || size != sizeof(values)) return false;
    std::memcpy(output, values, short_bulk ? sizeof(uint64_t) : size);
    return !short_bulk;
  }
};

int main() {
  using Table = cr_native::EntityPointerSnapshot<FakeReader, 4>;
  for (bool enabled : {false, true}) {
    FakeReader reader;
    Table table(reader, 0x1000, 3, enabled);
    for (int32_t i = 0; i < 3; ++i) {
      uint64_t value = 0;
      if (!table.read(i, &value) || value != reader.values[i]) return 1;
    }
    if (reader.scalar_calls != (enabled ? 0 : 3) || reader.bulk_calls != (enabled ? 1 : 0)) return 2;
    uint64_t value = 0;
    if (table.read(-1, &value) || table.read(3, &value) || table.read(0, nullptr)) return 3;
  }
  {
    FakeReader reader;
    reader.short_bulk = true;
    Table table(reader, 0x1000, 3, true);
    if (!table.bulk_attempted() || table.bulk_copied()) return 4;
    for (int32_t i = 0; i < 3; ++i) {
      uint64_t value = 0;
      if (!table.read(i, &value) || value != reader.values[i]) return 5;
    }
    if (reader.bulk_calls != 1 || reader.scalar_calls != 3) return 6;
  }
  for (int32_t count : {-1, 5}) {
    FakeReader reader;
    Table table(reader, 0x1000, count, true);
    uint64_t value = 0;
    if (table.bulk_attempted() || table.read(0, &value) || reader.bulk_calls || reader.scalar_calls) return 7;
  }
  {
    FakeReader reader;
    Table table(reader, 0, 0, true);
    uint64_t value = 0;
    if (table.bulk_attempted() || table.read(0, &value) || reader.bulk_calls || reader.scalar_calls) return 8;
  }
  {
    FakeReader reader;
    Table table(reader, 0x1000, 3, true);
    reader.values[1] = 0x9000;
    uint64_t value = 0;
    if (!table.read(1, &value) || value != 0x3000) return 9;
  }
  std::puts("entity pointer snapshot: scalar/bulk/short-read/bounds/empty/snapshot tests passed");
  return 0;
}
