#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace cr_native {
// A snapshot only during a known read-only section. Never spans native mutation.
// Blocks are fetched using the original checked reader, not raw dereferencing.
template <std::size_t BlockBytes = 1024, std::size_t Slots = 64>
class ScopedReadCache {
  static_assert(BlockBytes && (BlockBytes & (BlockBytes - 1)) == 0);
  static_assert(Slots && (Slots & (Slots - 1)) == 0);
  struct Block {
    uintptr_t address = 0;
    uint64_t epoch = 0;
    std::array<unsigned char, BlockBytes> bytes;
  };
 public:
  ScopedReadCache() {}  // Leave payload bytes uninitialized; epoch guards every read.
  uint64_t hits = 0, fills = 0, fallbacks = 0, epochs = 0;
  bool begin() {
    // Nested/unclear scopes fail back to uncached reads, rather than sharing
    // a potentially stale outer snapshot.
    if (active_) { active_ = false; return false; }
    if (++epoch_ == 0) {
      for (auto& block : blocks_) block.epoch = 0;
      epoch_ = 1;
    }
    active_ = true; ++epochs; return true;
  }
  void end() { active_ = false; }
  template<class Reader>
  bool read(uintptr_t address, void* output, std::size_t size, Reader direct) {
    if (!output || !address || !size || address > std::numeric_limits<uintptr_t>::max() - size) return false;
    if (!active_) return direct(address, output, size);
    const uintptr_t base = address & ~uintptr_t(BlockBytes - 1);
    const std::size_t offset = static_cast<std::size_t>(address - base);
    if (size > BlockBytes - offset) { ++fallbacks; return direct(address, output, size); }
    constexpr std::size_t Ways = Slots < 4 ? Slots : 4;
    constexpr std::size_t Sets = Slots / Ways;
    // Native allocators often separate size classes into far-apart arenas.
    // Mix high address bits so equal low offsets do not thrash one cache set.
    const uintptr_t key=base/BlockBytes;
    const std::size_t set=(key^(key>>11)^(key>>23))&(Sets-1);
    Block* selected=nullptr;
    for(std::size_t way=0;way<Ways;++way){
      auto& candidate=blocks_[set*Ways+way];
      if(candidate.epoch==epoch_ && candidate.address==base){selected=&candidate;break;}
    }
    if(selected){++hits;} else {
      // Full entity prefixes are already contiguous reads. Do not amplify a
      // 292-byte body read into a 1KiB prefetch unless the block is already hot.
      if(size>64){++fallbacks;return direct(address,output,size);}
      for(std::size_t way=0;way<Ways;++way){
        auto& candidate=blocks_[set*Ways+way];
        if(candidate.epoch!=epoch_){selected=&candidate;break;}
      }
      if(!selected)selected=&blocks_[set*Ways+(victim_++%Ways)];
      auto& block=*selected;
      block.epoch = 0;
      if (!direct(base, block.bytes.data(), BlockBytes)) {
        ++fallbacks;
        return direct(address, output, size);
      }
      block.address = base; block.epoch = epoch_; ++fills;
    }
    std::memcpy(output, selected->bytes.data() + offset, size);
    return true;
  }
 private:
  bool active_ = false;
  uint64_t epoch_ = 0;
  std::size_t victim_ = 0;
  std::array<Block, Slots> blocks_;
};
}  // namespace cr_native
