#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace cr_native {

// A bounded read-only snapshot of the contiguous pointer table. Entity bodies
// still use the existing checked reader. A partial bulk read is never exposed:
// the caller falls back to the original scalar read for each entry.
template <typename Reader, std::size_t Capacity>
class EntityPointerSnapshot {
 public:
  EntityPointerSnapshot(const Reader& reader, uintptr_t data, int32_t count,
                        bool bulk_enabled)
      : reader_(reader), data_(data), count_(count),
        valid_(count >= 0 && static_cast<std::size_t>(count) <= Capacity &&
               (count == 0 || data != 0)),
        attempted_(valid_ && bulk_enabled && count > 0),
        copied_(attempted_ && reader.read_bytes(
            data, pointers_.data(), static_cast<std::size_t>(count) * sizeof(uint64_t))) {}

  bool read(int32_t index, uint64_t* output) const {
    if (!valid_ || output == nullptr || index < 0 || index >= count_) return false;
    if (copied_) {
      *output = pointers_[static_cast<std::size_t>(index)];
      return true;
    }
    return reader_.read(data_ + static_cast<uintptr_t>(index) * sizeof(uint64_t), output);
  }

  bool bulk_attempted() const { return attempted_; }
  bool bulk_copied() const { return copied_; }

 private:
  const Reader& reader_;
  uintptr_t data_;
  int32_t count_;
  bool valid_;
  std::array<uint64_t, Capacity> pointers_;
  bool attempted_;
  bool copied_;
};

}  // namespace cr_native
