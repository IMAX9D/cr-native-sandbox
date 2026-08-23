#include <dlfcn.h>
#include <fcntl.h>
#include <jni.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr uintptr_t kExpectedJniOnLoadRva = 0x1458BC0;
constexpr uintptr_t kCreateGameMainRva = 0x1458E00;
constexpr uintptr_t kThreadOptionsMapRva = 0x1AB9910;
constexpr uintptr_t kManagerGlobalRva = 0x1A85978;
constexpr uintptr_t kInitManagerRva = 0xCE65B0;
constexpr uintptr_t kSetReplayDataRva = 0xCE7C40;
constexpr uintptr_t kGameStateManagerUpdateRva = 0xCE7810;
constexpr uintptr_t kBattleReplayControllerRva = 0x10B8AD0;
constexpr uintptr_t kSubmitReplayToControllerRva = 0x11ABC50;
constexpr uintptr_t kBattleStateUpdateRva = 0xCE26D0;
constexpr uintptr_t kBattleCoreUpdateRva = 0xCE2CC0;
constexpr uintptr_t kSkipCoreAndPresentationFlagRva = 0x1A85930;
constexpr uintptr_t kDoSpellCommandCtorRva = 0xD8D4D0;
constexpr uintptr_t kDoSpellCommandExecuteRva = 0xD8D520;
constexpr uintptr_t kBuildCanonicalSelectionRva = 0x1048170;
constexpr uintptr_t kResolveCanonicalSelectionRva = 0xE85D40;
constexpr uintptr_t kValidateDeploymentRva = 0xD5B770;
constexpr uintptr_t kBattleIdentityIndexRva = 0xD4E180;
constexpr uintptr_t kBattlePlayerAtIndexRva = 0xD4FFE0;
constexpr uintptr_t kDeckIndexToHandRva = 0xF96360;
constexpr uintptr_t kHandEntryRva = 0xF8FD20;
constexpr uintptr_t kPlayerElixirRva = 0xF93EA0;
constexpr uintptr_t kNextDeckIndexRva = 0xF98120;
constexpr size_t kMaxObservedEntities = 2048;
constexpr size_t kMaxPathNodes = 115;
constexpr uintptr_t kProjectileVtableRva = 0x1969B38;
constexpr jint kTraceSchemaVersion = 1;
constexpr jint kMaxTraceSteps = 64;
constexpr jint kMinTraceResponseBytes = 64 * 1024;
constexpr jint kMaxTraceResponseBytes = 32 * 1024 * 1024;
constexpr uintptr_t kNativeAllocRva = 0x18B1600;
constexpr uintptr_t kNativeFreeRva = 0x18B1630;
constexpr uintptr_t kNativeStringFromUtf8Rva = 0x140F8F0;
constexpr uintptr_t kNativeStringDestroyRva = 0x140F7D0;
constexpr uintptr_t kParseJsonObjectRva = 0x127C450;

using CreateGameMain = jstring (*)(
    JNIEnv*, jclass, jobject, jstring, jstring, jstring, jlong, jint, jint,
    jint, jfloat, jfloat, jint, jstring, jobject);
using NativeAlloc = void* (*)(size_t);
using NativeFree = void (*)(void*);
using NativeStringFromUtf8 = void* (*)(void*, const char*);
using NativeStringDestroy = void (*)(void*);
using ParseJsonObject = void* (*)(void*, int, int);
using BattleReplayController = void* (*)(void*);
using SubmitReplayToController = void (*)(void*, void*);
using SetReplayData = void (*)(void*, void*);
using GameStateManagerUpdate = void (*)(void*, float);
using InitManager = void (*)();
using BattleStateUpdate = void (*)(void*, float);
using DoSpellCommandCtor = void (*)(void*, void*);
using DoSpellCommandExecute = int32_t (*)(void*, void*, int32_t, int32_t);
using BuildCanonicalSelection = void* (*)(void*, void*, void*, int32_t);
using ResolveCanonicalSelection = void* (*)(void*);
using ValidateDeployment = int32_t (*)(
    void*, int32_t, int32_t, void*, int32_t);
using BattleIdentityIndex = int32_t (*)(void*, int32_t, int32_t);
using BattlePlayerAtIndex = void* (*)(void*, int32_t);
using DeckIndexToHand = int32_t (*)(void*, int32_t);
using HandEntry = void* (*)(void*, int32_t);
using PlayerElixir = int32_t (*)(void*);
using NextDeckIndex = int32_t (*)(void*);

class SafeMemoryReader {
 public:
  SafeMemoryReader() : fd_(open("/proc/self/mem", O_RDONLY | O_CLOEXEC)) {}
  ~SafeMemoryReader() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  template <typename T>
  bool read(uintptr_t address, T* value) const {
    if (fd_ < 0 || address == 0 || value == nullptr) {
      return false;
    }
    return pread(fd_, value, sizeof(T), static_cast<off_t>(address)) ==
           static_cast<ssize_t>(sizeof(T));
  }

  bool read_bytes(uintptr_t address, void* output, size_t size) const {
    if (fd_ < 0 || address == 0 || output == nullptr || size == 0) {
      return false;
    }
    return pread(fd_, output, size, static_cast<off_t>(address)) ==
           static_cast<ssize_t>(size);
  }

 private:
  int fd_;
};

struct CrownTowerState {
  uint64_t id = 0;
  int32_t side = -1;
  int32_t x = 0;
  int32_t y = 0;
  int32_t hp = -1;
  int32_t max_hp = -1;
  bool king = false;
  bool seen_now = false;
};

struct EpisodeState {
  uint64_t battle = 0;
  int32_t tick = -1;
  int32_t battle_phase = -1;
  int32_t logic_state = -1;
  int32_t logic_substate = -1;
  int32_t battle_flag_1e9 = -1;
  CrownTowerState towers[6] = {};
  size_t tower_count = 0;
  bool initialized = false;
  bool terminated = false;
  bool core_only_terminal_phase = false;
  int32_t stalled_updates = 0;
  const char* termination_reason = nullptr;
};

EpisodeState g_episode;

void reset_episode_state() {
  g_episode = EpisodeState{};
}

bool read_entity_hp(const SafeMemoryReader& memory,
                    const unsigned char* raw, int32_t* hp,
                    int32_t* max_hp) {
  uint64_t component_array = 0;
  std::memcpy(&component_array, raw + 0x18, sizeof(component_array));
  uint64_t components[3] = {};
  if (component_array == 0 ||
      !memory.read_bytes(component_array, components, sizeof(components)) ||
      components[2] == 0) {
    return false;
  }
  int32_t hp_pair[2] = {-1, -1};
  if (!memory.read_bytes(components[2] + 0x10, hp_pair, sizeof(hp_pair)) ||
      hp_pair[0] < 0 || hp_pair[1] < hp_pair[0] || hp_pair[1] > 100000) {
    return false;
  }
  *hp = hp_pair[0];
  *max_hp = hp_pair[1];
  return true;
}

bool capture_episode_state(const SafeMemoryReader& memory, uint64_t battle) {
  uint64_t logic = 0, registry = 0, collection = 0, data = 0;
  int32_t tick = -1, count = -1;
  if (battle == 0 || !memory.read(battle + 0x60, &tick) ||
      !memory.read(battle + 0xA8, &logic) || logic == 0 ||
      !memory.read(logic + 0x08, &registry) || registry == 0 ||
      !memory.read(registry + 0x40, &collection) || collection == 0 ||
      !memory.read(collection + 0x08, &data) || data == 0 ||
      !memory.read(collection + 0x14, &count) || count < 0 ||
      count > static_cast<int32_t>(kMaxObservedEntities)) {
    return false;
  }
  if (g_episode.battle != battle) {
    reset_episode_state();
    g_episode.battle = battle;
  }
  g_episode.tick = tick;
  memory.read(battle + 0x24, &g_episode.battle_phase);
  memory.read(logic + 0x18, &g_episode.logic_state);
  unsigned char flag_1e9 = 0;
  if (memory.read(battle + 0x1E9, &flag_1e9)) {
    g_episode.battle_flag_1e9 = static_cast<int32_t>(flag_1e9);
  }
  uint64_t logic_substate = 0;
  if (memory.read(logic + 0x1B0, &logic_substate) && logic_substate != 0) {
    memory.read(logic_substate + 0x08, &g_episode.logic_substate);
  }
  for (size_t index = 0; index < g_episode.tower_count; ++index) {
    g_episode.towers[index].seen_now = false;
  }
  for (int32_t index = 0; index < count; ++index) {
    uint64_t entity = 0;
    unsigned char raw[0x124] = {};
    if (!memory.read(data + static_cast<uintptr_t>(index) * 8, &entity) ||
        entity == 0 || !memory.read_bytes(entity, raw, sizeof(raw))) {
      continue;
    }
    auto raw_i32 = [&raw](size_t offset) {
      int32_t value = 0;
      std::memcpy(&value, raw + offset, sizeof(value));
      return value;
    };
    const int32_t kind = raw_i32(0x30);
    const int32_t side = raw_i32(0x78);
    const int32_t x = raw_i32(0x7C);
    const int32_t y = raw_i32(0x80);
    const int32_t card_id = raw_i32(0xAC);
    int32_t hp = -1, max_hp = -1;
    // In the verified standard 1v1 registry, original crown-tower entities
    // are the card-less buildings. Deployed buildings retain their card ID.
    if ((kind != 12 && kind != 13) || card_id != -1 ||
        (side != 0 && side != 1) ||
        x < 0 || x > 18000 || y < 0 || y > 32000 ||
        !read_entity_hp(memory, raw, &hp, &max_hp)) {
      continue;
    }
    CrownTowerState* tower = nullptr;
    for (size_t tower_index = 0; tower_index < g_episode.tower_count;
         ++tower_index) {
      if (g_episode.towers[tower_index].id == entity) {
        tower = &g_episode.towers[tower_index];
        break;
      }
    }
    if (tower == nullptr && g_episode.tower_count < 6) {
      tower = &g_episode.towers[g_episode.tower_count++];
      tower->id = entity;
      tower->side = side;
      tower->x = x;
      tower->y = y;
      tower->king = std::abs(x - 9000) <= 1500;
    }
    if (tower != nullptr) {
      tower->hp = hp;
      tower->max_hp = max_hp;
      tower->seen_now = true;
    }
  }
  size_t princess_count = 0;
  int princess_by_side[2] = {0, 0};
  for (size_t index = 0; index < g_episode.tower_count; ++index) {
    const CrownTowerState& tower = g_episode.towers[index];
    if (!tower.king) {
      ++princess_count;
      ++princess_by_side[tower.side];
    }
  }
  // Sleeping King towers can be absent from the active entity registry until
  // their activation. Four Princess towers are therefore the complete minimum
  // standard-1v1 outcome snapshot; King towers are added if/when libg exposes
  // them and remain tracked through destruction.
  if (princess_count == 4 && princess_by_side[0] == 2 &&
      princess_by_side[1] == 2) {
    g_episode.initialized = true;
  }
  if (g_episode.initialized) {
    for (size_t index = 0; index < g_episode.tower_count; ++index) {
      if (!g_episode.towers[index].seen_now) {
        g_episode.towers[index].hp = 0;
      }
    }
  }
  return g_episode.initialized;
}

int episode_crowns(int32_t side) {
  int crowns = 0;
  for (size_t index = 0; index < g_episode.tower_count; ++index) {
    const CrownTowerState& tower = g_episode.towers[index];
    // A side earns crowns by destroying the opposing side's towers.
    if (tower.side != 1 - side || tower.hp > 0) {
      continue;
    }
    if (tower.king) {
      return 3;
    }
    ++crowns;
  }
  return crowns;
}

std::string episode_json() {
  const int crowns0 = episode_crowns(0);
  const int crowns1 = episode_crowns(1);
  int winner = -1;
  if (g_episode.terminated && crowns0 != crowns1) {
    winner = crowns0 > crowns1 ? 0 : 1;
  }
  const double reward0 = winner < 0 ? 0.0 : (winner == 0 ? 1.0 : -1.0);
  const char* outcome = !g_episode.terminated
      ? "ongoing"
      : (winner < 0 ? "draw" : (winner == 0 ? "side0_win" : "side1_win"));
  std::string result;
  result.reserve(1600);
  char header[768];
  std::snprintf(
      header, sizeof(header),
      "{\"terminated\":%s,\"truncated\":false,\"outcome\":\"%s\",\"winner\":%s,"
      "\"crowns\":[%d,%d],\"rewards\":[%.1f,%.1f],"
      "\"reward_definition\":\"zero_sum_from_native_winner\","
      "\"result_source\":\"native_logic_terminal_and_crown_tower_entities\","
      "\"terminal_tick\":%d,\"native_phase\":{"
      "\"battle\":%d,\"logic\":%d,\"logic_substate\":%d,\"flag_1e9\":%d},"
      "\"termination_reason\":%s,"
      "\"tower_snapshot_complete\":%s,\"crown_towers\":[",
      g_episode.terminated ? "true" : "false",
      outcome,
      winner < 0 ? "null" : (winner == 0 ? "0" : "1"), crowns0, crowns1,
      reward0, -reward0, g_episode.tick, g_episode.battle_phase,
      g_episode.logic_state, g_episode.logic_substate,
      g_episode.battle_flag_1e9,
      g_episode.termination_reason == nullptr
          ? "null"
          : (std::strcmp(g_episode.termination_reason,
                         "native_logic_terminal") == 0
                 ? "\"native_logic_terminal\""
                 : (std::strcmp(g_episode.termination_reason,
                                "native_logic_clock_stopped") == 0
                        ? "\"native_logic_clock_stopped\""
                        : "\"native_battle_state_transition\"")),
      g_episode.initialized ? "true" : "false");
  result.append(header);
  for (size_t index = 0; index < g_episode.tower_count; ++index) {
    const CrownTowerState& tower = g_episode.towers[index];
    char row[320];
    std::snprintf(
        row, sizeof(row),
        "%s{\"id\":\"0x%llx\",\"side\":%d,\"type\":\"%s\","
        "\"lane\":%s,\"x\":%d,\"y\":%d,\"hp\":%d,\"max_hp\":%d,"
        "\"destroyed\":%s}",
        index == 0 ? "" : ",",
        static_cast<unsigned long long>(tower.id), tower.side,
        tower.king ? "king" : "princess",
        tower.king ? "null" : (tower.x < 9000 ? "\"left\"" : "\"right\""),
        tower.x, tower.y, tower.hp, tower.max_hp,
        tower.hp <= 0 ? "true" : "false");
    result.append(row);
  }
  result.append("]}");
  return result;
}

bool read_native_string(const SafeMemoryReader& memory, uintptr_t address,
                        char* output, size_t capacity) {
  if (output == nullptr || capacity == 0) {
    return false;
  }
  output[0] = '\0';
  int32_t length = -1;
  if (!memory.read(address + 4, &length) || length < 0 ||
      static_cast<size_t>(length) >= capacity) {
    return false;
  }
  uintptr_t data = address + 8;
  if (length >= 8 && !memory.read(address + 8, &data)) {
    return false;
  }
  for (int32_t index = 0; index < length; ++index) {
    unsigned char value = 0;
    if (!memory.read(data + static_cast<uintptr_t>(index), &value) ||
        value < 0x20 || value >= 0x7f) {
      output[0] = '\0';
      return false;
    }
    output[index] = static_cast<char>(value);
  }
  output[length] = '\0';
  return true;
}

void throw_state(JNIEnv* env, const std::string& message) {
  jclass type = env->FindClass("java/lang/IllegalStateException");
  if (type != nullptr) {
    env->ThrowNew(type, message.c_str());
  }
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_royale_nativehost_JniHost_nativeAct(
    JNIEnv* env, jclass, jstring libg_path, jint side, jint deck_index,
    jint x, jint y, jint account_hi, jint account_lo, jboolean dry_run) {
  if (side < 0 || side > 1 || deck_index < 0 || deck_index > 7 ||
      x < 0 || x > 18000 || y < 0 || y > 32000) {
    throw_state(env, "action fields are outside the native arena/deck range");
    return nullptr;
  }
  const char* path_chars = env->GetStringUTFChars(libg_path, nullptr);
  if (path_chars == nullptr) {
    return nullptr;
  }
  void* handle = dlopen(path_chars, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  env->ReleaseStringUTFChars(libg_path, path_chars);
  if (handle == nullptr) {
    throw_state(env, "libg is not loaded for native action execution");
    return nullptr;
  }
  void* exported = dlsym(handle, "JNI_OnLoad");
  Dl_info info{};
  if (exported == nullptr || dladdr(exported, &info) == 0 ||
      info.dli_fbase == nullptr) {
    dlclose(handle);
    throw_state(env, "cannot resolve libg base for native action execution");
    return nullptr;
  }
  const auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
  if (reinterpret_cast<uintptr_t>(exported) - base !=
      kExpectedJniOnLoadRva) {
    dlclose(handle);
    throw_state(env, "libg version guard rejected native action execution");
    return nullptr;
  }

  SafeMemoryReader memory;
  uint64_t manager = 0, state = 0, battle = 0, command_context = 0;
  uint64_t battle_logic = 0;
  int32_t current_type = -1, tick = -1;
  if (!memory.read(base + kManagerGlobalRva, &manager) || manager == 0 ||
      !memory.read(manager + 0x20, &state) || state == 0 ||
      !memory.read(manager + 0x30, &current_type) || current_type != 4 ||
      !memory.read(state + 0x90, &battle) || battle == 0 ||
      !memory.read(battle + 0xA8, &battle_logic) || battle_logic == 0 ||
      !memory.read(battle + 0x208, &command_context) || command_context == 0 ||
      !memory.read(battle + 0x60, &tick)) {
    dlclose(handle);
    throw_state(env, "native battle is not ready for action execution");
    return nullptr;
  }

  auto native_alloc = reinterpret_cast<NativeAlloc>(base + kNativeAllocRva);
  auto identity_index = reinterpret_cast<BattleIdentityIndex>(
      base + kBattleIdentityIndexRva);
  auto player_at_index = reinterpret_cast<BattlePlayerAtIndex>(
      base + kBattlePlayerAtIndexRva);
  auto deck_index_to_hand = reinterpret_cast<DeckIndexToHand>(
      base + kDeckIndexToHandRva);
  auto hand_entry = reinterpret_cast<HandEntry>(base + kHandEntryRva);
  const int32_t player_index = identity_index(
      reinterpret_cast<void*>(battle_logic), account_hi, account_lo);
  void* player = player_index < 0
      ? nullptr
      : player_at_index(reinterpret_cast<void*>(battle_logic), player_index);
  const int32_t hand_index = player == nullptr
      ? -1
      : deck_index_to_hand(player, deck_index);
  void* entry = hand_index < 0 || hand_index > 3
      ? nullptr
      : hand_entry(player, hand_index);
  if (entry == nullptr) {
    char payload[256];
    std::snprintf(
        payload, sizeof(payload),
        "{\"accepted\":false,\"result_code\":9,\"tick\":%d,"
        "\"side\":%d,\"deck_index\":%d,\"hand_index\":%d,"
        "\"reason\":\"card_not_in_hand\"}",
        tick, side, deck_index, hand_index);
    dlclose(handle);
    return env->NewStringUTF(payload);
  }

  void* command = native_alloc(0x58);
  if (command == nullptr) {
    dlclose(handle);
    throw_state(env, "libg could not allocate native action command");
    return nullptr;
  }
  auto construct = reinterpret_cast<DoSpellCommandCtor>(
      base + kDoSpellCommandCtorRva);
  auto execute = reinterpret_cast<DoSpellCommandExecute>(
      base + kDoSpellCommandExecuteRva);
  auto build_selection = reinterpret_cast<BuildCanonicalSelection>(
      base + kBuildCanonicalSelectionRva);
  construct(command, reinterpret_cast<void*>(command_context));
  *reinterpret_cast<int32_t*>(reinterpret_cast<uintptr_t>(command) + 0x14) =
      account_hi;
  *reinterpret_cast<int32_t*>(reinterpret_cast<uintptr_t>(command) + 0x18) =
      account_lo;
  *reinterpret_cast<int32_t*>(reinterpret_cast<uintptr_t>(command) + 0x28) = x;
  *reinterpret_cast<int32_t*>(reinterpret_cast<uintptr_t>(command) + 0x2C) = y;
  *reinterpret_cast<int32_t*>(reinterpret_cast<uintptr_t>(command) + 0x30) = -1;
  build_selection(
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(command) + 0x38),
      entry, player, 0);

  int32_t packed_selection = 0;
  uint64_t original_spell = 0;
  memory.read(reinterpret_cast<uintptr_t>(command) + 0x38, &original_spell);
  memory.read(reinterpret_cast<uintptr_t>(command) + 0x48, &packed_selection);
  auto resolve_selection = reinterpret_cast<ResolveCanonicalSelection>(
      base + kResolveCanonicalSelectionRva);
  auto validate_deployment = reinterpret_cast<ValidateDeployment>(
      base + kValidateDeploymentRva);
  void* resolved_selection = resolve_selection(
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(command) + 0x38));
  const int32_t placement_code = resolved_selection == nullptr
      ? -1
      : validate_deployment(
            reinterpret_cast<void*>(battle_logic), x, y,
            resolved_selection, 0);
  // The native command queue invokes vtable slot 3 with flags=3 and
  // feedback=false (D8CACC). Reuse those exact arguments so the command
  // performs the authoritative hand/elixir mutation and deployment.
  constexpr int32_t kCommandExecutionFlags = 3;
  const int32_t result_code = dry_run
      ? (placement_code == 0 ? 0 : placement_code < 0 ? 12
                                                   : placement_code + 14)
      : execute(
            command, reinterpret_cast<void*>(battle),
            kCommandExecutionFlags, 0);

  auto** command_vtable = *reinterpret_cast<void***>(command);
  reinterpret_cast<void (*)(void*)>(command_vtable[1])(command);
  const char* placement_reason = placement_code == 0
      ? "valid"
      : placement_code == 1
      ? "x_below_native_arena"
      : placement_code == 2
      ? "y_below_deploy_band"
      : placement_code == 3
      ? "x_above_native_arena"
      : placement_code == 4
      ? "y_above_deploy_band"
      : placement_code == 5
      ? "blocked_tile_or_card_constraint"
      : "selection_unavailable";
  char payload[640];
  std::snprintf(
      payload, sizeof(payload),
      "{\"accepted\":%s,\"result_code\":%d,\"tick\":%d,"
      "\"side\":%d,\"deck_index\":%d,\"hand_index\":%d,"
      "\"x\":%d,\"y\":%d,"
      "\"dry_run\":%s,\"placement_valid\":%s,"
      "\"placement_code\":%d,\"placement_reason\":\"%s\","
      "\"packed_selection\":%d,\"execution_flags\":%d,"
      "\"original_spell\":\"0x%llx\","
      "\"command_rva\":\"0x%llx\",\"execute_rva\":\"0x%llx\"}",
      result_code == 0 ? "true" : "false", result_code, tick, side,
      deck_index, hand_index, x, y, dry_run ? "true" : "false",
      placement_code == 0 ? "true" : "false", placement_code,
      placement_reason, packed_selection, kCommandExecutionFlags,
      static_cast<unsigned long long>(original_spell),
      static_cast<unsigned long long>(kDoSpellCommandCtorRva),
      static_cast<unsigned long long>(kDoSpellCommandExecuteRva));
  dlclose(handle);
  return env->NewStringUTF(payload);
}

extern "C" JNIEXPORT jstring JNICALL
Java_royale_nativehost_JniHost_nativeProbeGrid(
    JNIEnv* env, jclass, jstring libg_path, jint side, jint deck_index,
    jint account_hi, jint account_lo) {
  if (side < 0 || side > 1 || deck_index < 0 || deck_index > 7) {
    throw_state(env, "grid probe fields are outside the native deck range");
    return nullptr;
  }
  const char* path_chars = env->GetStringUTFChars(libg_path, nullptr);
  if (path_chars == nullptr) {
    return nullptr;
  }
  void* handle = dlopen(path_chars, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  env->ReleaseStringUTFChars(libg_path, path_chars);
  if (handle == nullptr) {
    throw_state(env, "libg is not loaded for native grid probe");
    return nullptr;
  }
  void* exported = dlsym(handle, "JNI_OnLoad");
  Dl_info info{};
  if (exported == nullptr || dladdr(exported, &info) == 0 ||
      info.dli_fbase == nullptr) {
    dlclose(handle);
    throw_state(env, "cannot resolve libg base for native grid probe");
    return nullptr;
  }
  const auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
  if (reinterpret_cast<uintptr_t>(exported) - base !=
      kExpectedJniOnLoadRva) {
    dlclose(handle);
    throw_state(env, "libg version guard rejected native grid probe");
    return nullptr;
  }

  SafeMemoryReader memory;
  uint64_t manager = 0, state = 0, battle = 0, command_context = 0;
  uint64_t battle_logic = 0;
  int32_t current_type = -1;
  if (!memory.read(base + kManagerGlobalRva, &manager) || manager == 0 ||
      !memory.read(manager + 0x20, &state) || state == 0 ||
      !memory.read(manager + 0x30, &current_type) || current_type != 4 ||
      !memory.read(state + 0x90, &battle) || battle == 0 ||
      !memory.read(battle + 0xA8, &battle_logic) || battle_logic == 0 ||
      !memory.read(battle + 0x208, &command_context) || command_context == 0) {
    dlclose(handle);
    throw_state(env, "native battle is not ready for grid probe");
    return nullptr;
  }

  auto identity_index = reinterpret_cast<BattleIdentityIndex>(
      base + kBattleIdentityIndexRva);
  auto player_at_index = reinterpret_cast<BattlePlayerAtIndex>(
      base + kBattlePlayerAtIndexRva);
  auto deck_index_to_hand = reinterpret_cast<DeckIndexToHand>(
      base + kDeckIndexToHandRva);
  auto hand_entry = reinterpret_cast<HandEntry>(base + kHandEntryRva);
  const int32_t player_index = identity_index(
      reinterpret_cast<void*>(battle_logic), account_hi, account_lo);
  void* player = player_index < 0
      ? nullptr
      : player_at_index(reinterpret_cast<void*>(battle_logic), player_index);
  const int32_t hand_index = player == nullptr
      ? -1
      : deck_index_to_hand(player, deck_index);
  void* entry = hand_index < 0 || hand_index > 3
      ? nullptr
      : hand_entry(player, hand_index);
  if (entry == nullptr) {
    dlclose(handle);
    throw_state(env, "grid probe card is not in the native hand");
    return nullptr;
  }

  auto native_alloc = reinterpret_cast<NativeAlloc>(base + kNativeAllocRva);
  void* command = native_alloc(0x58);
  if (command == nullptr) {
    dlclose(handle);
    throw_state(env, "libg could not allocate native grid probe command");
    return nullptr;
  }
  auto construct = reinterpret_cast<DoSpellCommandCtor>(
      base + kDoSpellCommandCtorRva);
  auto build_selection = reinterpret_cast<BuildCanonicalSelection>(
      base + kBuildCanonicalSelectionRva);
  auto resolve_selection = reinterpret_cast<ResolveCanonicalSelection>(
      base + kResolveCanonicalSelectionRva);
  auto validate_deployment = reinterpret_cast<ValidateDeployment>(
      base + kValidateDeploymentRva);
  construct(command, reinterpret_cast<void*>(command_context));
  *reinterpret_cast<int32_t*>(reinterpret_cast<uintptr_t>(command) + 0x14) =
      account_hi;
  *reinterpret_cast<int32_t*>(reinterpret_cast<uintptr_t>(command) + 0x18) =
      account_lo;
  *reinterpret_cast<int32_t*>(reinterpret_cast<uintptr_t>(command) + 0x30) = -1;
  build_selection(
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(command) + 0x38),
      entry, player, 0);
  void* resolved_selection = resolve_selection(
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(command) + 0x38));
  if (resolved_selection == nullptr) {
    auto** command_vtable = *reinterpret_cast<void***>(command);
    reinterpret_cast<void (*)(void*)>(command_vtable[1])(command);
    dlclose(handle);
    throw_state(env, "native grid probe selection is unavailable");
    return nullptr;
  }

  std::string rows;
  int32_t valid_cells = 0;
  for (int32_t row = 0; row < 32; ++row) {
    if (!rows.empty()) {
      rows += ',';
    }
    rows += '"';
    for (int32_t column = 0; column < 18; ++column) {
      const int32_t code = validate_deployment(
          reinterpret_cast<void*>(battle_logic), column * 1000 + 500,
          row * 1000 + 500, resolved_selection, 0);
      rows += code == 0 ? '1' : '0';
      if (code == 0) {
        ++valid_cells;
      }
    }
    rows += '"';
  }
  auto** command_vtable = *reinterpret_cast<void***>(command);
  reinterpret_cast<void (*)(void*)>(command_vtable[1])(command);

  std::string payload =
      "{\"width\":18,\"height\":32,\"cell_size\":1000,";
  payload += "\"valid_cells\":" + std::to_string(valid_cells);
  payload += ",\"rows\":[" + rows + "]}";
  dlclose(handle);
  return env->NewStringUTF(payload.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_royale_nativehost_JniHost_nativeObserve(
    JNIEnv* env, jclass, jstring libg_path) {
  const char* path_chars = env->GetStringUTFChars(libg_path, nullptr);
  if (path_chars == nullptr) {
    return nullptr;
  }
  void* handle = dlopen(path_chars, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  env->ReleaseStringUTFChars(libg_path, path_chars);
  if (handle == nullptr) {
    throw_state(env, "libg is not loaded for observation");
    return nullptr;
  }
  void* exported = dlsym(handle, "JNI_OnLoad");
  Dl_info info{};
  if (exported == nullptr || dladdr(exported, &info) == 0 ||
      info.dli_fbase == nullptr) {
    dlclose(handle);
    throw_state(env, "cannot resolve libg base for observation");
    return nullptr;
  }
  const auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
  if (reinterpret_cast<uintptr_t>(exported) - base !=
      kExpectedJniOnLoadRva) {
    dlclose(handle);
    throw_state(env, "libg version guard rejected observation");
    return nullptr;
  }

  SafeMemoryReader memory;
  uint64_t manager = 0, state = 0, battle = 0, hp_state = 0;
  uint64_t registry = 0, collection = 0, data = 0;
  int32_t tick_before = -1, tick_after = -1, replay_tick = -1, count = -1;
  if (!memory.read(base + kManagerGlobalRva, &manager) || manager == 0 ||
      !memory.read(manager + 0x20, &state) || state == 0 ||
      !memory.read(state + 0x90, &battle) || battle == 0 ||
      !memory.read(battle + 0x60, &tick_before) ||
      !memory.read(battle + 0x1BC, &replay_tick) ||
      !memory.read(battle + 0xA8, &hp_state) || hp_state == 0 ||
      !memory.read(hp_state + 0x08, &registry) || registry == 0 ||
      !memory.read(registry + 0x40, &collection) || collection == 0 ||
      !memory.read(collection + 0x08, &data) || data == 0 ||
      !memory.read(collection + 0x14, &count) || count < 0 ||
      count > static_cast<int32_t>(kMaxObservedEntities)) {
    dlclose(handle);
    throw_state(env, "native battle registry is not ready for observation");
    return nullptr;
  }
  capture_episode_state(memory, battle);

  std::string result;
  result.reserve(512 + static_cast<size_t>(count) * 256);
  char header[320];
  std::snprintf(
      header, sizeof(header),
      "{\"schema_version\":1,\"kind\":\"libg_native_state\","
      "\"tick\":%d,\"applied_replay_tick\":%d,\"entities\":[",
      tick_before, replay_tick);
  result.append(header);
  uint64_t state_hash = 1469598103934665603ULL;
  auto hash_value = [&state_hash](uint64_t value) {
    for (size_t index = 0; index < sizeof(value); ++index) {
      state_hash ^= static_cast<unsigned char>(value >> (index * 8));
      state_hash *= 1099511628211ULL;
    }
  };
  hash_value(static_cast<uint32_t>(tick_before));

  struct ObservedEntity {
    uint64_t id = 0;
    uint64_t target = 0;
    int32_t category = 0;
    int32_t kind = 0;
    int32_t side = 0;
    int32_t x = 0;
    int32_t y = 0;
    int32_t x2 = 0;
    int32_t y2 = 0;
    int32_t card_id = 0;
    int32_t level = 0;
    int32_t hp = 0;
    int32_t max_hp = 0;
    int32_t behavior = 0;
    int32_t pending_damage = -1;
    int32_t event_timer_ms = -1;
    int32_t target_previous_x = 0;
    int32_t target_previous_y = 0;
    int32_t attack_progress_ms = 0;
    int32_t attack_load_timer_ms = 0;
    int32_t direction_x = 0;
    int32_t direction_y = 0;
    int32_t collision_accumulator_x = 0;
    int32_t collision_accumulator_y = 0;
    int32_t collision_count = 0;
    int32_t avoidance_offset = 0;
    int32_t path_node_count = -1;
    int32_t path_segment_direction_x = 0;
    int32_t path_segment_direction_y = 0;
    unsigned char path_node_consumed = 0;
    bool attack_component_valid = false;
    bool move_component_valid = false;
    std::array<int32_t, kMaxPathNodes> path_nodes = {};
    std::array<int32_t, 11> target_key = {};
  };
  struct ObservedEffect {
    uint64_t id = 0;
    uint64_t vtable_rva = 0;
    uint64_t source = 0;
    uint64_t target = 0;
    uint64_t attached_owner = 0;
    int32_t category = 0;
    int32_t kind = 0;
    int32_t side = 0;
    int32_t x = 0;
    int32_t y = 0;
    int32_t x2 = 0;
    int32_t y2 = 0;
    int32_t card_id = 0;
    int32_t projectile_x_candidate = 0;
    int32_t projectile_y_candidate = 0;
  };
  std::vector<ObservedEntity> observed;
  std::vector<ObservedEffect> observed_effects;
  observed.reserve(static_cast<size_t>(count));
  observed_effects.reserve(static_cast<size_t>(count));
  for (int32_t index = 0; index < count; ++index) {
    uint64_t entity = 0;
    unsigned char raw[0x128] = {};
    if (!memory.read(data + static_cast<uintptr_t>(index) * 8, &entity) ||
        entity == 0 || !memory.read_bytes(entity, raw, sizeof(raw))) {
      continue;
    }
    auto raw_i32 = [&raw](size_t offset) {
      int32_t value = 0;
      std::memcpy(&value, raw + offset, sizeof(value));
      return value;
    };
    auto raw_u64 = [&raw](size_t offset) {
      uint64_t value = 0;
      std::memcpy(&value, raw + offset, sizeof(value));
      return value;
    };
    const int32_t category = raw_i32(0x08);
    const int32_t kind = raw_i32(0x30);
    const int32_t side = raw_i32(0x78);
    const int32_t x = raw_i32(0x7C);
    const int32_t y = raw_i32(0x80);
    const int32_t x2 = raw_i32(0x84);
    const int32_t y2 = raw_i32(0x88);
    const int32_t card_id = raw_i32(0xAC);
    const int32_t behavior = raw_i32(0x11C);
    const int32_t pending_damage = raw_i32(0x114);
    const int32_t event_timer_ms = raw_i32(0x118);
    const int32_t direction_x = raw_i32(0xF8);
    const int32_t direction_y = raw_i32(0xFC);
    const int32_t level = raw_i32(0x120) + 1;
    if (category >= 4000000 && category < 5000000) {
      const uint64_t vtable = raw_u64(0x00);
      const uint64_t vtable_rva = vtable >= base ? vtable - base : 0;
      if (vtable_rva == 0 || vtable_rva >= 0x3000000 ||
          (side != 0 && side != 1) ||
          (vtable_rva != kProjectileVtableRva &&
           (x < 0 || x > 18000 || y < 0 || y > 32000))) {
        continue;
      }
      observed_effects.push_back({
          entity, vtable_rva, raw_u64(0x100), raw_u64(0x108),
          raw_u64(0x118), category, kind, side, x, y, x2, y2,
          card_id, raw_i32(0x120), raw_i32(0x124)});
      continue;
    }
    if (category < 5000000 || category >= 6000000 || kind < 10 ||
        kind > 20 || (side != 0 && side != 1) || x < 0 || x > 18000 ||
        y < 0 || y > 32000 || level < 1 || level > 17 ||
        (card_id != -1 && (card_id < 20000000 || card_id >= 1000000000))) {
      continue;
    }
    int32_t hp = -1, max_hp = -1;
    uint64_t components[3] = {};
    const uint64_t component_array = raw_u64(0x18);
    if (component_array != 0 &&
        memory.read_bytes(component_array, components, sizeof(components)) &&
        components[2] != 0) {
      int32_t hp_pair[2] = {-1, -1};
      if (memory.read_bytes(components[2] + 0x10, hp_pair, sizeof(hp_pair)) &&
          hp_pair[0] >= 0 && hp_pair[1] >= hp_pair[0] &&
          hp_pair[1] <= 100000) {
        hp = hp_pair[0];
        max_hp = hp_pair[1];
      }
    }
    uint64_t target = 0;
    int32_t target_previous_x = 0, target_previous_y = 0;
    int32_t attack_progress_ms = 0, attack_load_timer_ms = 0;
    int32_t collision_accumulator_x = 0, collision_accumulator_y = 0;
    int32_t collision_count = 0, avoidance_offset = 0;
    int32_t path_node_count = -1;
    int32_t path_segment_direction_x = 0, path_segment_direction_y = 0;
    unsigned char path_node_consumed = 0;
    bool attack_component_valid = false, move_component_valid = false;
    std::array<int32_t, kMaxPathNodes> path_nodes = {};
    if (components[0] != 0) {
      unsigned char attack_raw[0x68] = {};
      if (memory.read_bytes(
              components[0], attack_raw, sizeof(attack_raw))) {
        auto attack_i32 = [&attack_raw](size_t offset) {
          int32_t value = 0;
          std::memcpy(&value, attack_raw + offset, sizeof(value));
          return value;
        };
        uint64_t actor = 0;
        std::memcpy(&actor, attack_raw + 0x08, sizeof(actor));
        if (actor == entity) {
          attack_component_valid = true;
          std::memcpy(&target, attack_raw + 0x10, sizeof(target));
          attack_progress_ms = attack_i32(0x24);
          attack_load_timer_ms = attack_i32(0x28);
          target_previous_x = attack_i32(0x60);
          target_previous_y = attack_i32(0x64);
        }
      }
    }
    if (components[1] != 0) {
      unsigned char move_raw[0x220] = {};
      if (memory.read_bytes(components[1], move_raw, sizeof(move_raw))) {
        auto move_i32 = [&move_raw](size_t offset) {
          int32_t value = 0;
          std::memcpy(&value, move_raw + offset, sizeof(value));
          return value;
        };
        const int32_t candidate_count = move_i32(0x38);
        if (candidate_count >= 0 &&
            candidate_count <= static_cast<int32_t>(kMaxPathNodes)) {
          move_component_valid = true;
          path_node_count = candidate_count;
          path_segment_direction_x = move_i32(0x30);
          path_segment_direction_y = move_i32(0x34);
          for (int32_t node = 0; node < path_node_count; ++node) {
            path_nodes[static_cast<size_t>(node)] =
                move_i32(0x3C + static_cast<size_t>(node) * 4);
          }
          collision_accumulator_x = move_i32(0x1D4);
          collision_accumulator_y = move_i32(0x1D8);
          collision_count = move_i32(0x1DC);
          avoidance_offset = move_i32(0x1F4);
          path_node_consumed = move_raw[0x1FF];
        }
      }
    }
    observed.push_back({
        entity, target, category, kind, side, x, y, x2, y2, card_id,
        level, hp, max_hp, behavior, pending_damage, event_timer_ms,
        target_previous_x, target_previous_y, attack_progress_ms,
        attack_load_timer_ms, direction_x, direction_y,
        collision_accumulator_x, collision_accumulator_y, collision_count,
        avoidance_offset, path_node_count, path_segment_direction_x,
        path_segment_direction_y, path_node_consumed,
        attack_component_valid, move_component_valid, path_nodes, {}});
  }
  for (ObservedEntity& row : observed) {
    row.target_key.fill(0);
    if (row.target == 0) {
      continue;
    }
    row.target_key[0] = 2;  // non-null but outside the public entity set
    for (const ObservedEntity& candidate : observed) {
      if (candidate.id != row.target) {
        continue;
      }
      row.target_key = {
          1, candidate.category, candidate.kind, candidate.side,
          candidate.x, candidate.y, candidate.x2, candidate.y2,
          candidate.card_id, candidate.level, candidate.behavior};
      break;
    }
  }
  // Registry order and raw pointers vary across otherwise identical process
  // runs.  Serialize and hash the public multiset in a stable tuple order.
  std::sort(
      observed.begin(), observed.end(),
      [](const ObservedEntity& left, const ObservedEntity& right) {
        return std::tie(
                   left.category, left.kind, left.side, left.x, left.y,
                   left.x2, left.y2, left.card_id, left.level, left.hp,
                   left.max_hp, left.behavior, left.target_key) <
               std::tie(
                   right.category, right.kind, right.side, right.x, right.y,
                   right.x2, right.y2, right.card_id, right.level, right.hp,
                   right.max_hp, right.behavior, right.target_key);
      });
  for (size_t index = 1; index < observed.size(); ++index) {
    if (observed[index - 1].category == observed[index].category) {
      dlclose(handle);
      throw_state(env, "native Character generation key is duplicated");
      return nullptr;
    }
  }
  size_t emitted = 0;
  for (const ObservedEntity& entity : observed) {
    char target_json[32];
    if (entity.target == 0) {
      std::snprintf(target_json, sizeof(target_json), "null");
    } else {
      std::snprintf(
          target_json, sizeof(target_json), "\"0x%llx\"",
          static_cast<unsigned long long>(entity.target));
    }
    char pending_json[32], event_json[32], attack_progress_json[32];
    char attack_load_json[32];
    std::snprintf(
        pending_json, sizeof(pending_json),
        entity.pending_damage < 0 ? "null" : "%d", entity.pending_damage);
    std::snprintf(
        event_json, sizeof(event_json),
        entity.event_timer_ms < 0 ? "null" : "%d", entity.event_timer_ms);
    std::snprintf(
        attack_progress_json, sizeof(attack_progress_json),
        entity.attack_component_valid ? "%d" : "null",
        entity.attack_progress_ms);
    std::snprintf(
        attack_load_json, sizeof(attack_load_json),
        entity.attack_component_valid ? "%d" : "null",
        entity.attack_load_timer_ms);
    char row[1536];
    std::snprintf(
        row, sizeof(row),
        "%s{\"id\":\"0x%llx\",\"generation_key\":%d,"
        "\"creation_ordinal\":%d,\"category\":%d,\"kind\":%d,"
        "\"side\":%d,\"x\":%d,\"y\":%d,\"x2\":%d,\"y2\":%d,"
        "\"card_id\":%d,\"level\":%d,\"hp\":%d,\"max_hp\":%d,"
        "\"behavior_state\":%d,\"pending_damage\":%s,"
        "\"event_timer_ms\":%s,\"target\":%s,"
        "\"target_previous_x\":%d,\"target_previous_y\":%d,"
        "\"attack_progress_ms\":%s,\"attack_load_timer_ms\":%s,"
        "\"movement_direction_x\":%d,\"movement_direction_y\":%d,"
        "\"collision_accumulator_x\":%d,"
        "\"collision_accumulator_y\":%d,\"collision_count\":%d,"
        "\"avoidance_offset\":%d,\"path_segment_direction_x\":%d,"
        "\"path_segment_direction_y\":%d,\"path_node_consumed\":%d,"
        "\"path_nodes\":",
        emitted == 0 ? "" : ",", static_cast<unsigned long long>(entity.id),
        entity.category, entity.category - 5000000,
        entity.category, entity.kind, entity.side, entity.x, entity.y,
        entity.x2, entity.y2, entity.card_id, entity.level, entity.hp,
        entity.max_hp, entity.behavior, pending_json, event_json, target_json,
        entity.target_previous_x, entity.target_previous_y,
        attack_progress_json, attack_load_json, entity.direction_x,
        entity.direction_y, entity.move_component_valid
            ? entity.collision_accumulator_x : 0,
        entity.move_component_valid ? entity.collision_accumulator_y : 0,
        entity.move_component_valid ? entity.collision_count : 0,
        entity.move_component_valid ? entity.avoidance_offset : 0,
        entity.move_component_valid ? entity.path_segment_direction_x : 0,
        entity.move_component_valid ? entity.path_segment_direction_y : 0,
        entity.move_component_valid
            ? static_cast<int32_t>(entity.path_node_consumed) : 0);
    result.append(row);
    if (!entity.move_component_valid) {
      result.append("null}");
    } else {
      result.push_back('[');
      for (int32_t node = 0; node < entity.path_node_count; ++node) {
        char value[32];
        std::snprintf(
            value, sizeof(value), "%s%d", node == 0 ? "" : ",",
            entity.path_nodes[static_cast<size_t>(node)]);
        result.append(value);
      }
      result.append("]}");
    }
    ++emitted;
    for (uint64_t value : {
             static_cast<uint64_t>(static_cast<uint32_t>(entity.category)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.kind)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.side)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.x)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.y)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.x2)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.y2)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.card_id)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.level)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.hp)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.max_hp)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.behavior)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.pending_damage)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.event_timer_ms)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.attack_progress_ms)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.attack_load_timer_ms)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.direction_x)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.direction_y)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.collision_accumulator_x)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.collision_accumulator_y)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.collision_count)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.avoidance_offset)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.path_node_count)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.path_segment_direction_x)),
             static_cast<uint64_t>(static_cast<uint32_t>(entity.path_segment_direction_y)),
             static_cast<uint64_t>(entity.path_node_consumed),
             static_cast<uint64_t>(entity.attack_component_valid),
             static_cast<uint64_t>(entity.move_component_valid)}) {
      hash_value(value);
    }
    for (int32_t node = 0; node < entity.path_node_count; ++node) {
      hash_value(static_cast<uint64_t>(static_cast<uint32_t>(
          entity.path_nodes[static_cast<size_t>(node)])));
    }
    for (int32_t value : entity.target_key) {
      hash_value(static_cast<uint64_t>(static_cast<uint32_t>(value)));
    }
  }
  hash_value(static_cast<uint64_t>(emitted));
  result.append("],\"effects\":[");
  std::sort(
      observed_effects.begin(), observed_effects.end(),
      [](const ObservedEffect& left, const ObservedEffect& right) {
        return std::tie(
                   left.category, left.kind, left.side, left.x, left.y,
                   left.x2, left.y2, left.card_id, left.vtable_rva) <
               std::tie(
                   right.category, right.kind, right.side, right.x, right.y,
                   right.x2, right.y2, right.card_id, right.vtable_rva);
      });
  for (size_t index = 0; index < observed_effects.size(); ++index) {
    const ObservedEffect& effect = observed_effects[index];
    auto pointer_json = [](uint64_t pointer, char* output, size_t size) {
      if (pointer == 0) {
        std::snprintf(output, size, "null");
      } else {
        std::snprintf(
            output, size, "\"0x%llx\"",
            static_cast<unsigned long long>(pointer));
      }
    };
    char source_json[32], target_json[32], attached_json[32];
    pointer_json(effect.source, source_json, sizeof(source_json));
    pointer_json(effect.target, target_json, sizeof(target_json));
    pointer_json(effect.attached_owner, attached_json, sizeof(attached_json));
    char row[768];
    std::snprintf(
        row, sizeof(row),
        "%s{\"id\":\"0x%llx\",\"vtable_rva\":\"0x%llx\","
        "\"category\":%d,\"kind\":%d,\"side\":%d,"
        "\"x\":%d,\"y\":%d,\"x2\":%d,\"y2\":%d,"
        "\"card_id\":%d,\"source\":%s,\"target\":%s,"
        "\"attached_owner\":%s,\"projectile_x_candidate\":%d,"
        "\"projectile_y_candidate\":%d}",
        index == 0 ? "" : ",",
        static_cast<unsigned long long>(effect.id),
        static_cast<unsigned long long>(effect.vtable_rva), effect.category,
        effect.kind, effect.side, effect.x, effect.y, effect.x2, effect.y2,
        effect.card_id, source_json, target_json, attached_json,
        effect.projectile_x_candidate, effect.projectile_y_candidate);
    result.append(row);
  }
  result.append("],\"effect_count\":");
  result.append(std::to_string(observed_effects.size()));
  size_t projectile_count = 0;
  for (const ObservedEffect& effect : observed_effects) {
    if (effect.vtable_rva == kProjectileVtableRva) {
      ++projectile_count;
    }
  }
  result.append(",\"projectiles\":[");
  size_t emitted_projectiles = 0;
  for (const ObservedEffect& effect : observed_effects) {
    if (effect.vtable_rva != kProjectileVtableRva) {
      continue;
    }
    auto pointer_json = [](uint64_t pointer, char* output, size_t size) {
      if (pointer == 0) {
        std::snprintf(output, size, "null");
      } else {
        std::snprintf(
            output, size, "\"0x%llx\"",
            static_cast<unsigned long long>(pointer));
      }
    };
    char source_json[32], target_json[32], attached_json[32];
    pointer_json(effect.source, source_json, sizeof(source_json));
    pointer_json(effect.target, target_json, sizeof(target_json));
    pointer_json(effect.attached_owner, attached_json, sizeof(attached_json));
    char row[768];
    std::snprintf(
        row, sizeof(row),
        "%s{\"id\":\"0x%llx\",\"generation_key\":%d,"
        "\"vtable_rva\":\"0x%llx\",\"side\":%d,"
        "\"x\":%d,\"y\":%d,\"x2\":%d,\"y2\":%d,"
        "\"card_id\":%d,\"source\":%s,\"target\":%s,"
        "\"attached_owner\":%s,\"target_x\":%d,\"target_y\":%d}",
        emitted_projectiles == 0 ? "" : ",",
        static_cast<unsigned long long>(effect.id), effect.category,
        static_cast<unsigned long long>(effect.vtable_rva), effect.side,
        effect.x, effect.y, effect.x2, effect.y2, effect.card_id,
        source_json, target_json, attached_json,
        effect.projectile_x_candidate, effect.projectile_y_candidate);
    result.append(row);
    ++emitted_projectiles;
    for (uint64_t value : {
             static_cast<uint64_t>(static_cast<uint32_t>(effect.category)),
             static_cast<uint64_t>(effect.vtable_rva),
             static_cast<uint64_t>(static_cast<uint32_t>(effect.side)),
             static_cast<uint64_t>(static_cast<uint32_t>(effect.x)),
             static_cast<uint64_t>(static_cast<uint32_t>(effect.y)),
             static_cast<uint64_t>(static_cast<uint32_t>(effect.x2)),
             static_cast<uint64_t>(static_cast<uint32_t>(effect.y2)),
             static_cast<uint64_t>(static_cast<uint32_t>(effect.card_id)),
             static_cast<uint64_t>(static_cast<uint32_t>(
                 effect.projectile_x_candidate)),
             static_cast<uint64_t>(static_cast<uint32_t>(
                 effect.projectile_y_candidate))}) {
      hash_value(value);
    }
  }
  hash_value(static_cast<uint64_t>(projectile_count));
  result.append("],\"projectile_count\":");
  result.append(std::to_string(projectile_count));
  result.append(",\"unclassified_effect_count\":");
  result.append(std::to_string(observed_effects.size() - projectile_count));
  result.append(",\"effects_classified\":");
  result.append(
      observed_effects.size() == projectile_count ? "true" : "false");
  result.append(",\"players\":[");
  auto player_at_index = reinterpret_cast<BattlePlayerAtIndex>(
      base + kBattlePlayerAtIndexRva);
  auto deck_index_to_hand = reinterpret_cast<DeckIndexToHand>(
      base + kDeckIndexToHandRva);
  auto player_elixir = reinterpret_cast<PlayerElixir>(
      base + kPlayerElixirRva);
  auto next_deck_index = reinterpret_cast<NextDeckIndex>(
      base + kNextDeckIndexRva);
  uint32_t battle_rng_state = 0;
  bool battle_rng_state_valid = false;
  for (int32_t side = 0; side < 2; ++side) {
    const int32_t player_index = side;
    void* player = player_index < 0
        ? nullptr
        : player_at_index(reinterpret_cast<void*>(hp_state), player_index);
    if (player == nullptr) {
      dlclose(handle);
      throw_state(env, "native battle player is unavailable for observation");
      return nullptr;
    }
    const uintptr_t player_address = reinterpret_cast<uintptr_t>(player);
    uint64_t player_context = 0, battle_rng_owner = 0, battle_rng = 0;
    uint32_t current_rng_state = 0;
    if (!memory.read(player_address + 0x10, &player_context) ||
        player_context == 0 ||
        !memory.read(player_context + 0x98, &battle_rng_owner) ||
        battle_rng_owner == 0 ||
        !memory.read(battle_rng_owner + 0x08, &battle_rng) ||
        battle_rng == 0 ||
        !memory.read(battle_rng + 0xA0, &current_rng_state) ||
        (battle_rng_state_valid && current_rng_state != battle_rng_state)) {
      dlclose(handle);
      throw_state(env, "native battle RNG path is unavailable");
      return nullptr;
    }
    battle_rng_state = current_rng_state;
    battle_rng_state_valid = true;
    int32_t refill_timer = -1;
    int32_t elixir_raw = -1;
    uint64_t hand_vector = 0;
    int32_t hand_capacity = -1, hand_size = -1;
    uint64_t cycle_vector = 0;
    int32_t cycle_capacity = -1, cycle_size = -1, deck_count = -1;
    if (!memory.read(player_address + 0x218, &refill_timer) ||
        !memory.read(player_address + 0x2F8, &elixir_raw) ||
        !memory.read(player_address + 0x220, &hand_vector) ||
        !memory.read(player_address + 0x228, &hand_capacity) ||
        !memory.read(player_address + 0x22C, &hand_size) ||
        !memory.read(player_address + 0x230, &cycle_vector) ||
        !memory.read(player_address + 0x238, &cycle_capacity) ||
        !memory.read(player_address + 0x23C, &cycle_size) ||
        !memory.read(player_address + 0x240, &deck_count) ||
        hand_capacity < 0 || hand_capacity > 8 || hand_size < 0 ||
        hand_size > hand_capacity || cycle_capacity < 0 ||
        cycle_capacity > 8 || cycle_size < 0 ||
        cycle_size > cycle_capacity || deck_count < 0 || deck_count > 8 ||
        (hand_size > 0 && hand_vector == 0) ||
        (cycle_size > 0 && cycle_vector == 0) ||
        elixir_raw < 0 || elixir_raw > 100000) {
      dlclose(handle);
      throw_state(env, "native player cycle vector failed strict bounds");
      return nullptr;
    }
    int32_t native_hand[8] = {};
    int32_t cycle_deck_indices[8] = {};
    if ((hand_size > 0 && !memory.read_bytes(
             hand_vector, native_hand,
             static_cast<size_t>(hand_size) * sizeof(int32_t))) ||
        (cycle_size > 0 && !memory.read_bytes(
             cycle_vector, cycle_deck_indices,
             static_cast<size_t>(cycle_size) * sizeof(int32_t)))) {
      dlclose(handle);
      throw_state(env, "native player cycle vector is unreadable");
      return nullptr;
    }
    for (int32_t index = 0; index < hand_size; ++index) {
      // libg deliberately writes -1 into the played hand slot while the
      // refill timer is active.  It is observable game state, not corrupt
      // deck data, and must survive a lossless per-Tick trace.
      if (native_hand[index] < -1 || native_hand[index] >= deck_count) {
        dlclose(handle);
        throw_state(env, "native hand vector contains invalid deck index");
        return nullptr;
      }
    }
    for (int32_t index = 0; index < cycle_size; ++index) {
      if (cycle_deck_indices[index] < 0 ||
          cycle_deck_indices[index] >= deck_count) {
        dlclose(handle);
        throw_state(env, "native cycle vector contains invalid deck index");
        return nullptr;
      }
    }
    const int32_t next_index = cycle_size == 0
        ? -1
        : next_deck_index(
              reinterpret_cast<void*>(player_address + 0x210));
    if (next_index < -1 || next_index >= deck_count ||
        (cycle_size > 0 && next_index < 0)) {
      dlclose(handle);
      throw_state(env, "native next deck index failed strict bounds");
      return nullptr;
    }
    int32_t deck_to_hand[8];
    int32_t hand_deck_indices[4] = {-1, -1, -1, -1};
    for (int32_t deck_index = 0; deck_index < 8; ++deck_index) {
      deck_to_hand[deck_index] = player == nullptr
          ? -1
          : deck_index_to_hand(player, deck_index);
      if (deck_to_hand[deck_index] >= 0 && deck_to_hand[deck_index] < 4) {
        hand_deck_indices[deck_to_hand[deck_index]] = deck_index;
      }
    }
    const int32_t elixir = player_elixir(player);
    char player_header[384];
    std::snprintf(
        player_header, sizeof(player_header),
        "%s{\"side\":%d,\"player_index\":%d,"
        "\"elixir\":%d,\"elixir_raw\":%d,\"refill_timer\":%d,"
        "\"next_deck_index\":%d,\"deck_to_hand\":[",
        side == 0 ? "" : ",", side, player_index, elixir, elixir_raw,
        refill_timer, next_index);
    result.append(player_header);
    for (int32_t deck_index = 0; deck_index < 8; ++deck_index) {
      char value[32];
      std::snprintf(
          value, sizeof(value), "%s%d", deck_index == 0 ? "" : ",",
          deck_to_hand[deck_index]);
      result.append(value);
    }
    result.append("],\"hand_deck_indices\":[");
    for (int32_t hand_index = 0; hand_index < 4; ++hand_index) {
      char value[32];
      std::snprintf(
          value, sizeof(value), "%s%d",
          hand_index == 0 ? "" : ",",
          hand_deck_indices[hand_index]);
      result.append(value);
    }
    result.append("],\"cycle_deck_indices\":[");
    for (int32_t cycle_index = 0; cycle_index < cycle_size; ++cycle_index) {
      char value[32];
      std::snprintf(
          value, sizeof(value), "%s%d", cycle_index == 0 ? "" : ",",
          cycle_deck_indices[cycle_index]);
      result.append(value);
    }
    result.append("]}");
    for (uint64_t value : {
             static_cast<uint64_t>(static_cast<uint32_t>(side)),
             static_cast<uint64_t>(static_cast<uint32_t>(elixir)),
             static_cast<uint64_t>(static_cast<uint32_t>(elixir_raw)),
             static_cast<uint64_t>(static_cast<uint32_t>(refill_timer)),
             static_cast<uint64_t>(static_cast<uint32_t>(next_index)),
             static_cast<uint64_t>(static_cast<uint32_t>(hand_size)),
             static_cast<uint64_t>(static_cast<uint32_t>(cycle_size)),
             static_cast<uint64_t>(static_cast<uint32_t>(deck_count))}) {
      hash_value(value);
    }
    for (int32_t value : hand_deck_indices) {
      hash_value(static_cast<uint64_t>(static_cast<uint32_t>(value)));
    }
    for (int32_t index = 0; index < cycle_size; ++index) {
      hash_value(static_cast<uint64_t>(
          static_cast<uint32_t>(cycle_deck_indices[index])));
    }
  }
  hash_value(static_cast<uint64_t>(battle_rng_state));
  memory.read(battle + 0x60, &tick_after);
  result.append("],\"entity_count\":");
  result.append(std::to_string(emitted));
  result.append(",\"coherent\":");
  result.append(tick_before == tick_after ? "true" : "false");
  result.append(",\"tick_after\":");
  result.append(std::to_string(tick_after));
  result.append(",\"rng_algorithm\":\"libg_xorshift32_v150535029\"");
  result.append(",\"rng_state\":");
  result.append(std::to_string(battle_rng_state));
  char hash_json[32];
  std::snprintf(
      hash_json, sizeof(hash_json), "\"%016llx\"",
      static_cast<unsigned long long>(state_hash));
  result.append(",\"state_hash\":");
  result.append(hash_json);
  result.append(",\"state_hash_scope\":\"public-observe-v5\"");
  result.append(",\"state_hash_certificate\":false,\"episode\":");
  result.append(episode_json());
  result.push_back('}');
  dlclose(handle);
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_royale_nativehost_JniHost_nativeStep(
    JNIEnv* env, jclass, jstring libg_path, jint steps) {
  if (steps < 0 || steps > 1000000) {
    throw_state(env, "step count must be in 0..1000000");
    return nullptr;
  }
  const char* path_chars = env->GetStringUTFChars(libg_path, nullptr);
  if (path_chars == nullptr) {
    return nullptr;
  }
  void* handle = dlopen(path_chars, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  env->ReleaseStringUTFChars(libg_path, path_chars);
  if (handle == nullptr) {
    throw_state(env, "libg is not loaded for stepping");
    return nullptr;
  }
  void* exported = dlsym(handle, "JNI_OnLoad");
  Dl_info info{};
  if (exported == nullptr || dladdr(exported, &info) == 0 ||
      info.dli_fbase == nullptr) {
    dlclose(handle);
    throw_state(env, "cannot resolve libg base for stepping");
    return nullptr;
  }
  const auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
  if (reinterpret_cast<uintptr_t>(exported) - base !=
      kExpectedJniOnLoadRva) {
    dlclose(handle);
    throw_state(env, "libg version guard rejected stepping");
    return nullptr;
  }
  SafeMemoryReader memory;
  uint64_t manager = 0;
  uint64_t state = 0;
  uint64_t state_vtable = 0;
  uint64_t update_address = 0;
  uint64_t battle = 0;
  int32_t current_type = -1;
  int32_t tick_before = -1;
  if (!memory.read(base + kManagerGlobalRva, &manager) || manager == 0 ||
      !memory.read(manager + 0x20, &state) || state == 0 ||
      !memory.read(manager + 0x30, &current_type) || current_type != 4 ||
      !memory.read(state, &state_vtable) || state_vtable == 0 ||
      !memory.read(state_vtable + 13 * sizeof(uint64_t), &update_address) ||
      update_address != base + kBattleStateUpdateRva ||
      !memory.read(state + 0x90, &battle) || battle == 0) {
    dlclose(handle);
    throw_state(env, "native battle state is not ready for controlled stepping");
    return nullptr;
  }
  memory.read(battle + 0x60, &tick_before);
  capture_episode_state(memory, battle);
  // CE26D0 is the full BattleGameState frame. CE2CC0 is its authoritative
  // simulation call. Run the core exactly once, then run the outer state with
  // libg's own 0x1A85930 gate enabled: that gate skips both the duplicate core
  // call and the HUD/result-page callback while preserving the native state
  // machine required for replay resets and battle teardown.
  auto core_update = reinterpret_cast<BattleStateUpdate>(
      base + kBattleCoreUpdateRva);
  auto state_update = reinterpret_cast<BattleStateUpdate>(update_address);
  auto* skip_core_and_presentation = reinterpret_cast<unsigned char*>(
      base + kSkipCoreAndPresentationFlagRva);
  jint completed = 0;
  bool battle_active = !g_episode.terminated;
  while (completed < steps) {
    if (g_episode.terminated) {
      battle_active = false;
      break;
    }
    const int32_t tick_before_update = g_episode.tick;
    core_update(reinterpret_cast<void*>(state), 0.05F);
    capture_episode_state(memory, battle);
    if (g_episode.tick >= 100 && g_episode.tick == tick_before_update) {
      ++g_episode.stalled_updates;
    } else {
      g_episode.stalled_updates = 0;
    }
    // libg intentionally pauses its clock while entering tiebreak. In the
    // verified standard-1v1 fixture, tick 6000 is the native terminal region;
    // detach presentation at that boundary or the first observed clock pause.
    // The inner core then owns every remaining HP drain and terminal tick.
    if (g_episode.tick >= 6000 || g_episode.stalled_updates > 0) {
      g_episode.core_only_terminal_phase = true;
    }
    if (!g_episode.core_only_terminal_phase) {
      const unsigned char previous_skip = __atomic_exchange_n(
          skip_core_and_presentation, 1, __ATOMIC_ACQ_REL);
      state_update(reinterpret_cast<void*>(state), 0.05F);
      __atomic_store_n(
          skip_core_and_presentation, previous_skip, __ATOMIC_RELEASE);
    }
    ++completed;
    const uint64_t current_state = __atomic_load_n(
        reinterpret_cast<uint64_t*>(manager + 0x20), __ATOMIC_ACQUIRE);
    const int32_t current_state_type = __atomic_load_n(
        reinterpret_cast<int32_t*>(manager + 0x30), __ATOMIC_ACQUIRE);
    const uint64_t current_battle = current_state == state
        ? __atomic_load_n(
              reinterpret_cast<uint64_t*>(state + 0x90), __ATOMIC_ACQUIRE)
        : 0;
    if (current_state != state || current_state_type != 4 ||
        current_battle != battle) {
      battle_active = false;
      g_episode.terminated = true;
      g_episode.termination_reason = "native_battle_state_transition";
      break;
    }
    capture_episode_state(memory, battle);
    if (g_episode.logic_state == 1 &&
        (g_episode.logic_substate == 6 || g_episode.logic_substate == 15)) {
      battle_active = false;
      g_episode.terminated = true;
      g_episode.termination_reason = "native_logic_terminal";
      break;
    }
    // The original inner battle core stops its 20 Hz clock after it has
    // applied the final crown/tiebreak result. Two consecutive no-progress
    // updates distinguish that authoritative terminal state from a single
    // floating-point accumulator frame.
    if (g_episode.stalled_updates >= 2 &&
        episode_crowns(0) + episode_crowns(1) > 0) {
      battle_active = false;
      g_episode.terminated = true;
      g_episode.termination_reason = "native_logic_clock_stopped";
      break;
    }
  }
  int32_t tick_after = -1;
  if (battle_active) {
    memory.read(battle + 0x60, &tick_after);
  } else if (g_episode.tick >= 0) {
    tick_after = g_episode.tick;
  }
  char payload[512];
  std::snprintf(
      payload, sizeof(payload),
      "{\"requested_steps\":%d,\"stepped\":%d,\"battle_active\":%s,"
      "\"fixed_dt\":0.05,\"tick_before\":%d,"
      "\"tick_after\":%d,\"state\":\"0x%llx\","
      "\"battle\":\"0x%llx\",\"entry_rva\":\"0x%llx\","
      "\"state_update_guard_rva\":\"0x%llx\",\"episode\":",
      steps, completed, battle_active ? "true" : "false", tick_before,
      tick_after,
      static_cast<unsigned long long>(state),
      static_cast<unsigned long long>(battle),
      static_cast<unsigned long long>(kBattleCoreUpdateRva),
      static_cast<unsigned long long>(kBattleStateUpdateRva));
  std::string result(payload);
  result.append(episode_json());
  result.push_back('}');
  dlclose(handle);
  return env->NewStringUTF(result.c_str());
}

bool append_trace_part(JNIEnv* env, std::string* output,
                       const std::string& value, size_t limit) {
  if (output == nullptr || output->size() > limit ||
      value.size() > limit - output->size()) {
    throw_state(env, "native tick trace exceeds max_response_bytes");
    return false;
  }
  output->append(value);
  return true;
}

bool append_trace_literal(JNIEnv* env, std::string* output,
                          const char* value, size_t limit) {
  return append_trace_part(env, output, std::string(value), limit);
}

bool take_jstring(JNIEnv* env, jstring value, std::string* output) {
  if (value == nullptr || output == nullptr || env->ExceptionCheck()) {
    return false;
  }
  const char* chars = env->GetStringUTFChars(value, nullptr);
  if (chars == nullptr) {
    return false;
  }
  output->assign(chars);
  env->ReleaseStringUTFChars(value, chars);
  env->DeleteLocalRef(value);
  return !env->ExceptionCheck();
}

std::string terminal_trace_observation() {
  std::string result =
      "{\"schema_version\":1,\"kind\":\"libg_native_terminal_state\",";
  result += "\"tick\":" + std::to_string(g_episode.tick);
  result += ",\"applied_replay_tick\":-1,\"entities\":[],\"effects\":[]";
  result += ",\"projectiles\":[],\"players\":[]";
  result += ",\"entity_count\":0,\"coherent\":false";
  result += ",\"tick_after\":" + std::to_string(g_episode.tick);
  result += ",\"rng_algorithm\":\"libg_xorshift32_v150535029\"";
  result += ",\"rng_state\":null,\"state_hash\":\"unavailable\"";
  result += ",\"state_hash_scope\":\"public-observe-v5\"";
  result += ",\"state_hash_certificate\":false,\"episode\":";
  result += episode_json();
  result.push_back('}');
  return result;
}

// One JNI boundary advances up to 64 authoritative 50 ms ticks and samples
// the public nativeObserve contract after every completed tick.  full-v1 is
// intentionally lossless; a future delta encoding can be added under a new
// trace schema/encoding without changing nativeStep or nativeObserve.
extern "C" JNIEXPORT jstring JNICALL
Java_royale_nativehost_JniHost_nativeStepTrace(
    JNIEnv* env, jclass clazz, jstring libg_path, jint steps,
    jint trace_schema_version, jint max_response_bytes) {
  if (trace_schema_version != kTraceSchemaVersion) {
    throw_state(env, "unsupported native tick trace schema_version");
    return nullptr;
  }
  if (steps < 1 || steps > kMaxTraceSteps) {
    throw_state(env, "native tick trace steps must be in 1..64");
    return nullptr;
  }
  if (max_response_bytes < kMinTraceResponseBytes ||
      max_response_bytes > kMaxTraceResponseBytes) {
    throw_state(
        env, "native tick trace max_response_bytes must be in 65536..33554432");
    return nullptr;
  }
  const size_t response_limit = static_cast<size_t>(max_response_bytes);
  jstring initial_value = Java_royale_nativehost_JniHost_nativeObserve(
      env, clazz, libg_path);
  std::string initial_state;
  if (!take_jstring(env, initial_value, &initial_state)) {
    return nullptr;
  }
  const bool initial_observation_complete =
      initial_state.find("\"coherent\":true") != std::string::npos;

  std::string result;
  result.reserve(std::min(
      response_limit, initial_state.size() * static_cast<size_t>(steps + 1) +
                          static_cast<size_t>(4096)));
  char header[256];
  std::snprintf(
      header, sizeof(header),
      "{\"schema_version\":1,\"kind\":\"libg_native_tick_trace\","
      "\"trace_schema_version\":1,\"encoding\":\"full-v1\","
      "\"requested_steps\":%d,\"max_response_bytes\":%d,"
      "\"initial_frame\":{\"frame_index\":0,\"advanced_steps\":0,"
      "\"observation_complete\":%s,\"state\":",
      steps, max_response_bytes,
      initial_observation_complete ? "true" : "false");
  if (!append_trace_literal(env, &result, header, response_limit) ||
      !append_trace_part(env, &result, initial_state, response_limit) ||
      !append_trace_literal(env, &result, "},\"frames\":[", response_limit)) {
    return nullptr;
  }

  jint completed = 0;
  bool terminal = g_episode.terminated;
  for (jint frame_index = 1; frame_index <= steps && !terminal;
       ++frame_index) {
    jstring step_value = Java_royale_nativehost_JniHost_nativeStep(
        env, clazz, libg_path, 1);
    std::string step_result;
    if (!take_jstring(env, step_value, &step_result)) {
      return nullptr;
    }
    ++completed;
    terminal = g_episode.terminated;

    jstring observation_value = Java_royale_nativehost_JniHost_nativeObserve(
        env, clazz, libg_path);
    std::string observation;
    bool observation_available = take_jstring(
        env, observation_value, &observation);
    if (!observation_available) {
      if (!terminal) {
        return nullptr;
      }
      // A terminal GameState transition can make the live registry disappear
      // between the authoritative final core update and observation. Preserve
      // an explicit final frame instead of silently dropping the terminal.
      if (env->ExceptionCheck()) {
        env->ExceptionClear();
      }
      observation = terminal_trace_observation();
    }
    const bool observation_complete = observation_available &&
        observation.find("\"coherent\":true") != std::string::npos;
    char frame_header[192];
    std::snprintf(
        frame_header, sizeof(frame_header),
        "%s{\"frame_index\":%d,\"advanced_steps\":%d,"
        "\"observation_complete\":%s,\"step\":",
        completed == 1 ? "" : ",", frame_index, completed,
        observation_complete ? "true" : "false");
    if (!append_trace_literal(
            env, &result, frame_header, response_limit) ||
        !append_trace_part(env, &result, step_result, response_limit) ||
        !append_trace_literal(env, &result, ",\"state\":", response_limit) ||
        !append_trace_part(env, &result, observation, response_limit) ||
        !append_trace_literal(env, &result, "}", response_limit)) {
      return nullptr;
    }
  }
  char footer[160];
  std::snprintf(
      footer, sizeof(footer),
      "],\"stepped\":%d,\"terminal\":%s,\"final_frame_index\":%d}",
      completed, terminal ? "true" : "false", completed);
  if (!append_trace_literal(env, &result, footer, response_limit)) {
    return nullptr;
  }
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_royale_nativehost_JniHost_nativeRestartReplay(
    JNIEnv* env, jclass, jstring libg_path, jstring replay_json) {
  const char* path_chars = env->GetStringUTFChars(libg_path, nullptr);
  if (path_chars == nullptr) {
    return nullptr;
  }
  void* handle = dlopen(path_chars, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  env->ReleaseStringUTFChars(libg_path, path_chars);
  if (handle == nullptr) {
    throw_state(env, "libg is not loaded for replay restart");
    return nullptr;
  }
  void* exported = dlsym(handle, "JNI_OnLoad");
  Dl_info info{};
  if (exported == nullptr || dladdr(exported, &info) == 0 ||
      info.dli_fbase == nullptr) {
    dlclose(handle);
    throw_state(env, "cannot resolve libg base for replay restart");
    return nullptr;
  }
  const auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
  if (reinterpret_cast<uintptr_t>(exported) - base !=
      kExpectedJniOnLoadRva) {
    dlclose(handle);
    throw_state(env, "libg version guard rejected replay restart");
    return nullptr;
  }
  SafeMemoryReader memory;
  uint64_t manager = 0;
  int32_t current_type = -1;
  uint64_t current_state = 0;
  uint64_t current_battle = 0;
  if (!memory.read(base + kManagerGlobalRva, &manager) || manager == 0 ||
      !memory.read(manager + 0x30, &current_type) || current_type != 4 ||
      !memory.read(manager + 0x20, &current_state) || current_state == 0 ||
      !memory.read(current_state + 0x90, &current_battle) ||
      current_battle <= 0x1000) {
    dlclose(handle);
    throw_state(env, "native battle state is not ready for replay restart");
    return nullptr;
  }
  const char* json_chars = env->GetStringUTFChars(replay_json, nullptr);
  if (json_chars == nullptr) {
    dlclose(handle);
    return nullptr;
  }
  auto native_alloc = reinterpret_cast<NativeAlloc>(base + kNativeAllocRva);
  auto native_string_from_utf8 = reinterpret_cast<NativeStringFromUtf8>(
      base + kNativeStringFromUtf8Rva);
  void* native_json = native_alloc(16);
  if (native_json == nullptr) {
    env->ReleaseStringUTFChars(replay_json, json_chars);
    dlclose(handle);
    throw_state(env, "libg could not allocate restart replay string");
    return nullptr;
  }
  native_string_from_utf8(native_json, json_chars);
  env->ReleaseStringUTFChars(replay_json, json_chars);

  reset_episode_state();
  __atomic_store_n(
      reinterpret_cast<unsigned char*>(
          base + kSkipCoreAndPresentationFlagRva),
      1, __ATOMIC_RELEASE);
  auto set_replay = reinterpret_cast<SetReplayData>(
      base + kSetReplayDataRva);
  auto manager_update = reinterpret_cast<GameStateManagerUpdate>(
      base + kGameStateManagerUpdateRva);
  // A terminal BattleGameState leaves result/tiebreak singletons owned by the
  // normal route back through HomeState.  Perform that same native transition
  // before queueing the next replay; direct 4 -> 4 replacement is sufficient
  // mid-battle but leaves terminal-only owners stale.
  __atomic_store_n(
      reinterpret_cast<int32_t*>(manager + 0x54), -1, __ATOMIC_RELEASE);
  __atomic_store_n(
      reinterpret_cast<int32_t*>(manager + 0x34), 1, __ATOMIC_RELEASE);
  __atomic_store_n(
      reinterpret_cast<int32_t*>(manager + 0x10), 3, __ATOMIC_RELEASE);
  manager_update(reinterpret_cast<void*>(manager), 0.0f);
  for (int frame = 0; frame < 5; ++frame) {
    manager_update(reinterpret_cast<void*>(manager), 0.05f);
  }
  // CE7C40 forwards replacement replay data to an already-running battle
  // controller when manager+0x20 is non-null.  That path is useful for the
  // live client, but it consumes the replay before CE7810 can construct the
  // replacement BattleGameState.  The service is paused here, so temporarily
  // detach HomeState from the manager while queueing the replay, then put it
  // back for the native state manager to release through its own vtable.
  auto* state_slot = reinterpret_cast<void**>(manager + 0x20);
  void* detached_state = __atomic_exchange_n(
      state_slot, nullptr, __ATOMIC_ACQ_REL);
  set_replay(reinterpret_cast<void*>(manager), native_json);
  __atomic_store_n(state_slot, detached_state, __ATOMIC_RELEASE);
  __atomic_store_n(
      reinterpret_cast<unsigned char*>(
          base + kSkipCoreAndPresentationFlagRva),
      1, __ATOMIC_RELEASE);
  manager_update(reinterpret_cast<void*>(manager), 0.0f);
  // BattleGameState::onEnter creates the battle graph in stages.  Let the
  // verified battle state frame finish native ownership before nativeStep
  // calls the authoritative core. Do not run the generic manager frame here:
  // it enters the UI-backed loading state, which is absent in this host.
  uint64_t initialization_state = 0;
  if (!memory.read(manager + 0x20, &initialization_state) ||
      initialization_state == 0) {
    dlclose(handle);
    throw_state(env, "replacement BattleGameState was not created");
    return nullptr;
  }
  auto state_update = reinterpret_cast<BattleStateUpdate>(
      base + kBattleStateUpdateRva);
  for (int frame = 0; frame < 10; ++frame) {
    __atomic_store_n(
        reinterpret_cast<unsigned char*>(
            base + kSkipCoreAndPresentationFlagRva),
        1, __ATOMIC_RELEASE);
    state_update(reinterpret_cast<void*>(initialization_state), 0.05f);
  }

  int32_t next_type = -1;
  int32_t pending_type = -1;
  uint64_t next_state = 0;
  memory.read(manager + 0x30, &next_type);
  memory.read(manager + 0x34, &pending_type);
  memory.read(manager + 0x20, &next_state);
  char payload[384];
  std::snprintf(
      payload, sizeof(payload),
      "{\"called\":true,\"manager_root\":\"0x%llx\","
      "\"previous_state\":\"0x%llx\",\"previous_battle\":\"0x%llx\","
      "\"next_state\":\"0x%llx\",\"next_state_type\":%d,"
      "\"pending_state_type\":%d,\"set_replay_rva\":\"0x%llx\","
      "\"manager_update_rva\":\"0x%llx\"}",
      static_cast<unsigned long long>(manager),
      static_cast<unsigned long long>(current_state),
      static_cast<unsigned long long>(current_battle),
      static_cast<unsigned long long>(next_state), next_type, pending_type,
      static_cast<unsigned long long>(kSetReplayDataRva),
      static_cast<unsigned long long>(kGameStateManagerUpdateRva));
  dlclose(handle);
  return env->NewStringUTF(payload);
}

extern "C" JNIEXPORT jstring JNICALL
Java_royale_nativehost_JniHost_nativeInitManager(
    JNIEnv* env, jclass, jstring libg_path) {
  const char* path_chars = env->GetStringUTFChars(libg_path, nullptr);
  if (path_chars == nullptr) {
    return nullptr;
  }
  void* handle = dlopen(path_chars, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  env->ReleaseStringUTFChars(libg_path, path_chars);
  if (handle == nullptr) {
    throw_state(env, "libg is not loaded for direct manager init");
    return nullptr;
  }
  void* exported = dlsym(handle, "JNI_OnLoad");
  Dl_info info{};
  if (exported == nullptr || dladdr(exported, &info) == 0 ||
      info.dli_fbase == nullptr) {
    dlclose(handle);
    throw_state(env, "cannot resolve libg base for direct manager init");
    return nullptr;
  }
  const auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
  if (reinterpret_cast<uintptr_t>(exported) - base != kExpectedJniOnLoadRva) {
    dlclose(handle);
    throw_state(env, "libg version guard rejected direct manager init");
    return nullptr;
  }
  SafeMemoryReader memory;
  uint64_t manager_before = 0;
  memory.read(base + kManagerGlobalRva, &manager_before);
  if (manager_before == 0) {
    auto initialize = reinterpret_cast<InitManager>(base + kInitManagerRva);
    initialize();
  }
  uint64_t manager_after = 0;
  memory.read(base + kManagerGlobalRva, &manager_after);
  char payload[256];
  std::snprintf(
      payload, sizeof(payload),
      "{\"called\":true,\"manager_before\":\"0x%llx\"," 
      "\"manager_after\":\"0x%llx\",\"entry_rva\":\"0x%llx\"}",
      static_cast<unsigned long long>(manager_before),
      static_cast<unsigned long long>(manager_after),
      static_cast<unsigned long long>(kInitManagerRva));
  dlclose(handle);
  return env->NewStringUTF(payload);
}

extern "C" JNIEXPORT jstring JNICALL
Java_royale_nativehost_JniHost_nativePumpManager(
    JNIEnv* env, jclass, jstring libg_path) {
  const char* path_chars = env->GetStringUTFChars(libg_path, nullptr);
  if (path_chars == nullptr) {
    return nullptr;
  }
  void* handle = dlopen(path_chars, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  env->ReleaseStringUTFChars(libg_path, path_chars);
  if (handle == nullptr) {
    throw_state(env, "libg is not loaded for direct manager pump");
    return nullptr;
  }
  void* exported = dlsym(handle, "JNI_OnLoad");
  Dl_info info{};
  if (exported == nullptr || dladdr(exported, &info) == 0 ||
      info.dli_fbase == nullptr) {
    dlclose(handle);
    throw_state(env, "cannot resolve libg base for direct manager pump");
    return nullptr;
  }
  const auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
  if (reinterpret_cast<uintptr_t>(exported) - base != kExpectedJniOnLoadRva) {
    dlclose(handle);
    throw_state(env, "libg version guard rejected direct manager pump");
    return nullptr;
  }
  SafeMemoryReader memory;
  uint64_t manager = 0;
  if (!memory.read(base + kManagerGlobalRva, &manager) || manager == 0) {
    dlclose(handle);
    throw_state(env, "direct manager is not initialized");
    return nullptr;
  }
  auto manager_update = reinterpret_cast<GameStateManagerUpdate>(
      base + kGameStateManagerUpdateRva);
  __atomic_store_n(
      reinterpret_cast<unsigned char*>(
          base + kSkipCoreAndPresentationFlagRva),
      1, __ATOMIC_RELEASE);
  manager_update(reinterpret_cast<void*>(manager), 0.0f);

  int32_t current_type = -1;
  int32_t pending_type = -1;
  uint64_t state = 0;
  uint64_t battle = 0;
  memory.read(manager + 0x30, &current_type);
  memory.read(manager + 0x34, &pending_type);
  memory.read(manager + 0x20, &state);
  if (current_type == 4 && state != 0) {
    auto state_update = reinterpret_cast<BattleStateUpdate>(
        base + kBattleStateUpdateRva);
    for (int frame = 0; frame < 10; ++frame) {
      __atomic_store_n(
          reinterpret_cast<unsigned char*>(
              base + kSkipCoreAndPresentationFlagRva),
          1, __ATOMIC_RELEASE);
      state_update(reinterpret_cast<void*>(state), 0.05f);
      memory.read(state + 0x90, &battle);
      if (battle > 0x1000) {
        break;
      }
    }
  }
  char payload[320];
  std::snprintf(
      payload, sizeof(payload),
      "{\"called\":true,\"manager\":\"0x%llx\"," 
      "\"state\":\"0x%llx\",\"battle\":\"0x%llx\"," 
      "\"current_state_type\":%d,\"pending_state_type\":%d}",
      static_cast<unsigned long long>(manager),
      static_cast<unsigned long long>(state),
      static_cast<unsigned long long>(battle), current_type, pending_type);
  dlclose(handle);
  return env->NewStringUTF(payload);
}

extern "C" JNIEXPORT jstring JNICALL
Java_royale_nativehost_JniHost_nativeLoadReplay(
    JNIEnv* env, jclass, jstring libg_path, jstring replay_json) {
  const char* path_chars = env->GetStringUTFChars(libg_path, nullptr);
  if (path_chars == nullptr) {
    return nullptr;
  }
  void* handle = dlopen(path_chars, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  env->ReleaseStringUTFChars(libg_path, path_chars);
  if (handle == nullptr) {
    throw_state(env, "libg is not loaded for replay input");
    return nullptr;
  }
  void* exported = dlsym(handle, "JNI_OnLoad");
  Dl_info info{};
  if (exported == nullptr || dladdr(exported, &info) == 0 ||
      info.dli_fbase == nullptr) {
    dlclose(handle);
    throw_state(env, "cannot resolve libg base for replay input");
    return nullptr;
  }
  const auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
  if (reinterpret_cast<uintptr_t>(exported) - base !=
      kExpectedJniOnLoadRva) {
    dlclose(handle);
    throw_state(env, "libg version guard rejected replay input");
    return nullptr;
  }
  uint64_t manager = 0;
  SafeMemoryReader memory;
  if (!memory.read(base + kManagerGlobalRva, &manager) || manager == 0) {
    dlclose(handle);
    throw_state(env, "game state manager is not ready for replay input");
    return nullptr;
  }
  const char* json_chars = env->GetStringUTFChars(replay_json, nullptr);
  if (json_chars == nullptr) {
    dlclose(handle);
    return nullptr;
  }
  auto native_alloc = reinterpret_cast<NativeAlloc>(base + kNativeAllocRva);
  auto native_free = reinterpret_cast<NativeFree>(base + kNativeFreeRva);
  auto native_string_from_utf8 = reinterpret_cast<NativeStringFromUtf8>(
      base + kNativeStringFromUtf8Rva);
  auto native_string_destroy = reinterpret_cast<NativeStringDestroy>(
      base + kNativeStringDestroyRva);
  auto parse_json_object = reinterpret_cast<ParseJsonObject>(
      base + kParseJsonObjectRva);
  void* native_json = native_alloc(16);
  if (native_json == nullptr) {
    env->ReleaseStringUTFChars(replay_json, json_chars);
    dlclose(handle);
    throw_state(env, "libg could not allocate replay string");
    return nullptr;
  }
  native_string_from_utf8(native_json, json_chars);
  env->ReleaseStringUTFChars(replay_json, json_chars);
  void* parsed_json = parse_json_object(native_json, 0, 100);
  native_string_destroy(native_json);
  native_free(native_json);
  const bool previous_episode_terminated = g_episode.terminated;
  reset_episode_state();

  // CE7C40 performs these exact stores after parsing. Its optional branch
  // forwards a replacement replay to an existing battle controller. The home
  // state uses +0x90 for a small enum, so call that branch only under the
  // guarded BattleGameState type where +0x90 is the verified battle pointer.
  auto* replay_slot = reinterpret_cast<void**>(manager + 0x78);
  void* old_replay = __atomic_exchange_n(
      replay_slot, parsed_json, __ATOMIC_ACQ_REL);
  if (old_replay != nullptr) {
    auto** vtable = *reinterpret_cast<void***>(old_replay);
    auto release = reinterpret_cast<void (*)(void*)>(vtable[1]);
    release(old_replay);
  }
  bool battle_controller_notified = false;
  int32_t current_type = -1;
  int32_t current_tick = -1;
  uint64_t current_state = 0, current_battle = 0;
  if (!previous_episode_terminated &&
      memory.read(manager + 0x30, &current_type) && current_type == 4 &&
      memory.read(manager + 0x20, &current_state) && current_state != 0 &&
      memory.read(current_state + 0x90, &current_battle) &&
      current_battle > 0x1000 &&
      memory.read(current_battle + 0x60, &current_tick) && current_tick >= 100) {
    auto controller_for_battle = reinterpret_cast<BattleReplayController>(
        base + kBattleReplayControllerRva);
    auto submit_replay = reinterpret_cast<SubmitReplayToController>(
        base + kSubmitReplayToControllerRva);
    void* controller = controller_for_battle(
        reinterpret_cast<void*>(current_battle));
    uint64_t receiver = 0;
    if (controller != nullptr &&
        memory.read(reinterpret_cast<uintptr_t>(controller) + 0x2F8,
                    &receiver) &&
        receiver != 0) {
      submit_replay(reinterpret_cast<void*>(receiver), parsed_json);
      battle_controller_notified = true;
    }
  }
  __atomic_store_n(
      reinterpret_cast<int32_t*>(manager + 0x54), -1, __ATOMIC_RELEASE);
  __atomic_store_n(
      reinterpret_cast<int32_t*>(manager + 0x34), 4, __ATOMIC_RELEASE);
  __atomic_store_n(
      reinterpret_cast<int32_t*>(manager + 0x10), 3, __ATOMIC_RELEASE);
  // Keep the isolated process on libg's native headless gate. The manager can
  // still replace BattleGameState while resumed, but no background frame can
  // advance the core or invoke a missing HUD. nativeStep advances CE2CC0 and
  // the gated outer state explicitly.
  __atomic_store_n(
      reinterpret_cast<unsigned char*>(
          base + kSkipCoreAndPresentationFlagRva),
      1, __ATOMIC_RELEASE);
  char payload[320];
  std::snprintf(
      payload, sizeof(payload),
      "{\"called\":true,\"parsed\":%s,"
      "\"manager_root\":\"0x%llx\",\"entry_rva\":\"0x%llx\","
      "\"battle_controller_notified\":%s,"
      "\"unsafe_home_callback_skipped\":%s}",
      parsed_json != nullptr ? "true" : "false",
      static_cast<unsigned long long>(manager),
      static_cast<unsigned long long>(kSetReplayDataRva),
      battle_controller_notified ? "true" : "false",
      current_type == 4 ? "false" : "true");
  dlclose(handle);
  return env->NewStringUTF(payload);
}

extern "C" JNIEXPORT jstring JNICALL
Java_royale_nativehost_JniHost_nativeProbeRuntime(
    JNIEnv* env, jclass, jstring libg_path) {
  const char* path_chars = env->GetStringUTFChars(libg_path, nullptr);
  if (path_chars == nullptr) {
    return nullptr;
  }
  void* handle = dlopen(path_chars, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  env->ReleaseStringUTFChars(libg_path, path_chars);
  if (handle == nullptr) {
    throw_state(env, "libg is not loaded for runtime probe");
    return nullptr;
  }
  void* exported = dlsym(handle, "JNI_OnLoad");
  Dl_info info{};
  if (exported == nullptr || dladdr(exported, &info) == 0 ||
      info.dli_fbase == nullptr) {
    dlclose(handle);
    throw_state(env, "cannot resolve libg base for runtime probe");
    return nullptr;
  }
  const auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
  const auto object = base + kThreadOptionsMapRva;
  SafeMemoryReader memory;
  uint64_t buckets = 0;
  uint64_t bucket_count = 0;
  uint64_t first = 0;
  uint64_t size = 0;
  uint32_t max_load_factor_bits = 0;
  uint64_t root = 0;
  uint64_t context = 0;
  uint64_t state_vtable = 0;
  uint64_t state_methods[24] = {};
  uint64_t battle = 0;
  uint64_t battle_vtable = 0;
  uint64_t battle_methods[12] = {};
  uint64_t replay_data = 0;
  int32_t manager_phase = -1;
  int32_t current_state_type = -1;
  int32_t pending_state_type = -1;
  int32_t tick = -1;
  char replay_key_battle[48] = {};
  char replay_key_cmd[48] = {};
  char replay_key_first_int[48] = {};
  char replay_key_second_int[48] = {};
  const bool map_ok =
      memory.read(object, &buckets) &&
      memory.read(object + 8, &bucket_count) &&
      memory.read(object + 16, &first) && memory.read(object + 24, &size) &&
      memory.read(object + 32, &max_load_factor_bits);
  const bool root_ok = memory.read(base + kManagerGlobalRva, &root);
  const bool context_ok =
      root_ok && root != 0 && memory.read(root + 0x20, &context);
  const bool manager_fields_ok =
      root_ok && root != 0 && memory.read(root + 0x10, &manager_phase) &&
      memory.read(root + 0x30, &current_state_type) &&
      memory.read(root + 0x34, &pending_state_type) &&
      memory.read(root + 0x78, &replay_data);
  const bool state_vtable_ok =
      context_ok && context > 0x1000 && memory.read(context, &state_vtable) &&
      state_vtable > base;
  if (state_vtable_ok) {
    for (size_t index = 0; index < 24; ++index) {
      memory.read(state_vtable + index * sizeof(uint64_t),
                  &state_methods[index]);
    }
  }
  const bool battle_ok =
      context_ok && context != 0 && memory.read(context + 0x90, &battle);
  const bool tick_ok =
      battle_ok && battle != 0 && memory.read(battle + 0x60, &tick);
  const bool battle_vtable_ok =
      battle_ok && battle > 0x1000 && memory.read(battle, &battle_vtable) &&
      battle_vtable > base;
  if (battle_vtable_ok) {
    for (size_t index = 0; index < 12; ++index) {
      memory.read(battle_vtable + index * sizeof(uint64_t),
                  &battle_methods[index]);
    }
  }
  read_native_string(
      memory, base + 0x1AB8860, replay_key_battle, sizeof(replay_key_battle));
  read_native_string(
      memory, base + 0x1AB8890, replay_key_cmd, sizeof(replay_key_cmd));
  read_native_string(memory, base + 0x1AB88A0, replay_key_first_int,
                     sizeof(replay_key_first_int));
  read_native_string(memory, base + 0x1AB88C0, replay_key_second_int,
                     sizeof(replay_key_second_int));
  char method_payload[512];
  size_t method_offset = 0;
  method_payload[method_offset++] = '[';
  for (size_t index = 0; index < 12; ++index) {
    const unsigned long long value = battle_methods[index] >= base
        ? static_cast<unsigned long long>(battle_methods[index] - base)
        : 0ULL;
    const int written = std::snprintf(
        method_payload + method_offset, sizeof(method_payload) - method_offset,
        "%s\"0x%llx\"", index == 0 ? "" : ",", value);
    if (written < 0 || static_cast<size_t>(written) >=
        sizeof(method_payload) - method_offset) {
      break;
    }
    method_offset += static_cast<size_t>(written);
  }
  if (method_offset < sizeof(method_payload) - 1) {
    method_payload[method_offset++] = ']';
  }
  method_payload[method_offset] = '\0';
  char state_method_payload[1024];
  size_t state_method_offset = 0;
  state_method_payload[state_method_offset++] = '[';
  for (size_t index = 0; index < 24; ++index) {
    const unsigned long long value = state_methods[index] >= base
        ? static_cast<unsigned long long>(state_methods[index] - base)
        : 0ULL;
    const int written = std::snprintf(
        state_method_payload + state_method_offset,
        sizeof(state_method_payload) - state_method_offset,
        "%s\"0x%llx\"", index == 0 ? "" : ",", value);
    if (written < 0 || static_cast<size_t>(written) >=
        sizeof(state_method_payload) - state_method_offset) {
      break;
    }
    state_method_offset += static_cast<size_t>(written);
  }
  if (state_method_offset < sizeof(state_method_payload) - 1) {
    state_method_payload[state_method_offset++] = ']';
  }
  state_method_payload[state_method_offset] = '\0';
  char payload[4096];
  std::snprintf(
      payload, sizeof(payload),
      "{\"rva\":\"0x%llx\",\"buckets\":\"0x%llx\","
      "\"bucket_count\":%llu,\"first\":\"0x%llx\",\"size\":%llu,"
      "\"max_load_factor_bits\":\"0x%08x\","
      "\"manager_root\":\"0x%llx\",\"context\":\"0x%llx\","
      "\"state_vtable\":\"0x%llx\",\"state_methods_rva\":%s,"
      "\"manager_phase\":%d,\"current_state_type\":%d,"
      "\"pending_state_type\":%d,\"replay_data\":\"0x%llx\","
      "\"replay_keys\":{\"battle\":\"%s\",\"cmd\":\"%s\","
      "\"first_int\":\"%s\",\"second_int\":\"%s\"},"
      "\"battle\":\"0x%llx\",\"battle_vtable\":\"0x%llx\","
      "\"battle_methods_rva\":%s,\"tick\":%d,"
      "\"read_ok\":{\"map\":%s,\"root\":%s,\"context\":%s,"
      "\"manager_fields\":%s,\"battle\":%s,\"tick\":%s}}",
      static_cast<unsigned long long>(kThreadOptionsMapRva),
      static_cast<unsigned long long>(buckets),
      static_cast<unsigned long long>(bucket_count),
      static_cast<unsigned long long>(first),
      static_cast<unsigned long long>(size), max_load_factor_bits,
      static_cast<unsigned long long>(root),
      static_cast<unsigned long long>(context),
      static_cast<unsigned long long>(state_vtable), state_method_payload,
      manager_phase, current_state_type, pending_state_type,
      static_cast<unsigned long long>(replay_data),
      replay_key_battle, replay_key_cmd, replay_key_first_int,
      replay_key_second_int,
      static_cast<unsigned long long>(battle),
      static_cast<unsigned long long>(battle_vtable), method_payload, tick,
      map_ok ? "true" : "false", root_ok ? "true" : "false",
      context_ok ? "true" : "false", manager_fields_ok ? "true" : "false",
      battle_ok ? "true" : "false", tick_ok ? "true" : "false");
  dlclose(handle);
  return env->NewStringUTF(payload);
}

extern "C" JNIEXPORT jstring JNICALL
Java_royale_nativehost_JniHost_nativeCreateGameMain(
    JNIEnv* env, jclass caller, jstring libg_path, jobject assets,
    jobject activity, jstring data_dir, jstring cache_dir,
    jstring external_cache_dir, jlong available_bytes, jint width, jint height,
    jint density_dpi, jfloat xdpi, jfloat ydpi, jint graphics_api,
    jstring external_files_dir) {
  const char* path_chars = env->GetStringUTFChars(libg_path, nullptr);
  if (path_chars == nullptr) {
    return nullptr;
  }
  void* handle = dlopen(path_chars, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  env->ReleaseStringUTFChars(libg_path, path_chars);
  if (handle == nullptr) {
    const char* error = dlerror();
    throw_state(env, std::string("libg is not loaded: ") +
                         (error != nullptr ? error : "unknown dlopen error"));
    return nullptr;
  }

  void* exported = dlsym(handle, "JNI_OnLoad");
  Dl_info info{};
  if (exported == nullptr || dladdr(exported, &info) == 0 || info.dli_fbase == nullptr) {
    dlclose(handle);
    throw_state(env, "cannot resolve libg load base");
    return nullptr;
  }
  const auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
  const auto actual_jni_rva = reinterpret_cast<uintptr_t>(exported) - base;
  if (actual_jni_rva != kExpectedJniOnLoadRva) {
    dlclose(handle);
    throw_state(env, "libg version guard rejected JNI_OnLoad RVA");
    return nullptr;
  }

  auto create_game_main = reinterpret_cast<CreateGameMain>(
      base + kCreateGameMainRva);
  jstring result = create_game_main(
      env, caller, assets, data_dir, cache_dir, external_cache_dir,
      available_bytes, width, height, density_dpi, xdpi, ydpi, graphics_api,
      external_files_dir, activity);
  dlclose(handle);
  return result;
}
