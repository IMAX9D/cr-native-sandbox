"""Run one real-libg Stage-1 expert self-play collection/update batch.

The entry point deliberately does not launch or restart native Workers.  Every
port must already expose the direct JSON service.  A batch is committed only
after all requested games end normally, its immutable rollout shard verifies,
the Critic update/checkpoint succeeds, and the BASE Actor remains byte-for-byte
unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from expert_selfplay_v1.actor_adapter import actor_state_digest
from expert_selfplay_v1.batched_policy import BatchedPolicyService
from expert_selfplay_v1.contracts import BatchManifest, canonical_schema_hash
from expert_selfplay_v1.critic import (
    ExpertActorCritic,
    PrivilegedCritic,
    PrivilegedCriticConfig,
)
from expert_selfplay_v1.decks import DeckFixture, DeckScheduler, file_sha256
from expert_selfplay_v1.ledger import RolloutLedger
from expert_selfplay_v1.native_observation import NativeObservationEncoder
from expert_selfplay_v1.rollout import EpisodeHeader
from expert_selfplay_v1.rollout_storage import (
    ImmutableRolloutShardWriter,
    LearnerEpisodeChunker,
    verify_rollout_shard,
)
from expert_v1.training_v1.model import (
    ExpertPolicyConfig,
    RecurrentExpertPolicy,
    configure_position_precision,
)
from native_core.env import NativeRoyaleEnv
from training.schema import DefensiveTowerReward


EXPERT_INFERENCE_KIND = "cr_native_expert_inference_weights_v1"
RUN_KIND = "cr_native_expert_selfplay_stage1_run_v1"
PROGRESS_KIND = "cr_native_expert_selfplay_stage1_progress_v1"
DEFAULT_RUNTIME_MANIFEST = PROJECT_ROOT / "bindings" / "runtime-150535029-x86_64.json"
DEFAULT_LEARNER_DECK = PROJECT_ROOT / "examples" / "user-selected-heavy-control.json"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


ACTION_SCHEMA: dict[str, Any] = {
    "kind": "expert_marked_hazard_action_v1",
    "clock": "native_tick_variable_delta_v1",
    "timing": "poisson_hazard_wait_or_event_v1",
    "event_branches": ("normal_card", "active_ability"),
    "card_slots": 4,
    "position_cells": 32 * 18,
    "ability_targeting": "native_pre_action_mask_v1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace one JSON file while rejecting NaN/Inf."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(
        dict(value), ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RunJournal:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.progress_path = root / "progress.json"
        self.events_path = root / "events.jsonl"
        self.started = time.monotonic()

    def event(self, event: str, **fields: Any) -> None:
        row = {"at_utc": utc_now(), "event": event, **fields}
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        with self.events_path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())

    def progress(self, status: str, **fields: Any) -> None:
        atomic_json(self.progress_path, {
            "kind": PROGRESS_KIND,
            "status": status,
            "updated_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - self.started,
            **fields,
        })


def parse_ports(value: str) -> list[int]:
    """Parse ``39031,39033-39035`` preserving order and rejecting aliases."""

    ports: list[int] = []
    seen: set[int] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            raise ValueError("--ports contains an empty item")
        if "-" in token:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"descending port range: {token}")
            rows = range(start, stop + 1)
        else:
            rows = (int(token),)
        for port in rows:
            if not 1 <= port <= 65535:
                raise ValueError(f"port outside 1..65535: {port}")
            if port in seen:
                raise ValueError(f"duplicate port: {port}")
            seen.add(port)
            ports.append(port)
    if not ports:
        raise ValueError("--ports requires at least one Worker")
    return ports


def _reward_schema() -> dict[str, Any]:
    return {
        "kind": DefensiveTowerReward.schema_version,
        "damage_dealt_scale": DefensiveTowerReward.damage_dealt_scale,
        "damage_received_scale": DefensiveTowerReward.damage_received_scale,
        "tower_destroyed_reward": DefensiveTowerReward.tower_destroyed_reward,
        "terminal_win_reward": DefensiveTowerReward.terminal_win_reward,
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object: {path}")
    return value


@dataclass(frozen=True)
class LoadedBase:
    actor: RecurrentExpertPolicy
    config: ExpertPolicyConfig
    encoder: NativeObservationEncoder
    checkpoint_sha256: str
    expert_manifest_sha256: str
    actor_sha256: str
    checkpoint_step: int
    checkpoint_run_id: str


def load_base(
    checkpoint_path: Path,
    expert_manifest_path: Path,
    *,
    device: torch.device,
) -> LoadedBase:
    """Load and bind the inference Actor to its exact frozen encoder manifest."""

    checkpoint_path = checkpoint_path.resolve(strict=True)
    expert_manifest_path = expert_manifest_path.resolve(strict=True)
    expert_manifest_sha256 = sha256_file(expert_manifest_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("kind") != EXPERT_INFERENCE_KIND:
        raise RuntimeError("--checkpoint is not an expert inference checkpoint")
    bound_manifest = str(checkpoint.get("dataset_manifest_sha256", ""))
    if bound_manifest != expert_manifest_sha256:
        raise RuntimeError(
            "expert checkpoint/dataset manifest SHA-256 mismatch: "
            f"{bound_manifest or '<missing>'} != {expert_manifest_sha256}"
        )
    if not isinstance(checkpoint.get("model_config"), Mapping):
        raise RuntimeError("expert checkpoint lacks model_config")
    if not isinstance(checkpoint.get("model_state"), Mapping):
        raise RuntimeError("expert checkpoint lacks model_state")
    config = ExpertPolicyConfig(**dict(checkpoint["model_config"]))
    configure_position_precision(config)
    manifest = _read_json(expert_manifest_path, label="expert manifest")
    encoder = NativeObservationEncoder.from_manifest(manifest)
    encoder.assert_compatible(config)
    actor = RecurrentExpertPolicy(config)
    actor.load_state_dict(checkpoint["model_state"], strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    actor_dtype = torch.float16 if device.type == "cuda" else torch.float32
    actor.to(device=device, dtype=actor_dtype)
    if any(parameter.requires_grad for parameter in actor.parameters()):
        raise RuntimeError("Stage-1 BASE Actor was not frozen")
    if not all(
        bool(torch.isfinite(value).all())
        for value in actor.state_dict().values()
        if value.is_floating_point()
    ):
        raise FloatingPointError("BASE Actor contains NaN/Inf")
    actor_sha256 = actor_state_digest(actor)
    return LoadedBase(
        actor=actor,
        config=config,
        encoder=encoder,
        checkpoint_sha256=sha256_file(checkpoint_path),
        expert_manifest_sha256=expert_manifest_sha256,
        actor_sha256=actor_sha256,
        checkpoint_step=int(checkpoint.get("global_step", -1)),
        checkpoint_run_id=str(checkpoint.get("run_id", "")),
    )


def _initial_hidden_sha256(actor: RecurrentExpertPolicy) -> str:
    device = next(actor.parameters()).device
    hidden = actor.initial_hidden(1, device=device)
    digest = hashlib.sha256()
    for tensor in hidden:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _runtime_identity(path: Path) -> dict[str, Any]:
    value = _read_json(path.resolve(strict=True), label="native runtime manifest")
    digest = str(value.get("libg_sha256", ""))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError("native runtime manifest has no valid libg_sha256")
    if str(value.get("abi")) != "x86_64":
        raise RuntimeError("Stage-1 requires the frozen x86_64 native runtime")
    return {
        "path": str(path.resolve()),
        "manifest_sha256": sha256_file(path.resolve()),
        "runtime_version": str(value.get("runtime_version", "")),
        "abi": "x86_64",
        "libg_sha256": digest,
    }


def _preflight_env(env: NativeRoyaleEnv) -> dict[str, Any]:
    response = env.client.request({"op": "status"})
    if not isinstance(response, Mapping) or not response.get("ok"):
        raise RuntimeError(
            f"native Worker {env.host}:{env.port} failed status preflight: {response!r}"
        )
    state = response.get("state")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"native Worker {env.host}:{env.port} returned no status state")
    read_ok = state.get("read_ok")
    required_reads = ("root", "context", "manager_fields", "battle", "tick")
    if (
        int(state.get("current_state_type", -1)) != 4
        or int(state.get("tick", -1)) < 0
        or state.get("battle") in (None, "0x0")
        or state.get("replay_data") in (None, "0x0")
        or not isinstance(read_ok, Mapping)
        or any(not bool(read_ok.get(name, False)) for name in required_reads)
    ):
        raise RuntimeError(
            f"native Worker {env.host}:{env.port} is not battle-ready: "
            f"state_type={state.get('current_state_type')} "
            f"tick={state.get('tick')} read_ok={read_ok}"
        )
    return {"host": env.host, "port": env.port, "status": dict(state)}


def _complete_episode(value: Any) -> tuple[dict[str, Any], Sequence[Any]]:
    episode = value.episode
    frozen = episode.freeze()
    decisions = frozen.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise RuntimeError("collector returned an empty learner episode")
    final = decisions[-1]
    if not isinstance(final, Mapping) or not bool(final.get("terminated")):
        raise RuntimeError("collector returned an incomplete learner episode")
    if bool(final.get("truncated")):
        raise RuntimeError("time-truncated episode is ineligible for Stage-1")
    step_payloads = value.step_payloads
    if len(step_payloads) != len(decisions):
        raise RuntimeError("collector decision/step-payload counts differ")
    return frozen, step_payloads


def _episode_spec(
    spec_type: type,
    *,
    env: NativeRoyaleEnv,
    fixture: DeckFixture,
    header: EpisodeHeader,
    learner_actor_sha256: str,
    opponent_actor_sha256: str,
) -> Any:
    """Construct the collector's explicit immutable episode request."""

    return spec_type(
        worker_id=f"worker-{env.port}",
        env=env,
        fixture=fixture,
        header=header,
        actor_hashes={
            fixture.learner_side: learner_actor_sha256,
            1 - fixture.learner_side: opponent_actor_sha256,
        },
    )


@dataclass(frozen=True)
class RuntimeDependencies:
    collector_type: type
    episode_spec_type: type
    trainer_type: type
    trainer_config_type: type


def runtime_dependencies() -> RuntimeDependencies:
    # These imports stay here so ``--help`` and static admission checks remain
    # usable while a cloud image is being staged.
    from expert_selfplay_v1.critic_training import (
        CriticTrainingConfig,
        Stage1CriticTrainer,
    )
    from expert_selfplay_v1.online_collector import (
        OnlineEpisodeSpec,
        OnlineSelfPlayCollector,
    )

    return RuntimeDependencies(
        collector_type=OnlineSelfPlayCollector,
        episode_spec_type=OnlineEpisodeSpec,
        trainer_type=Stage1CriticTrainer,
        trainer_config_type=CriticTrainingConfig,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        help="separate frozen opponent Actor checkpoint for collect-only PPO batches",
    )
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--ports", required=True, help="existing Worker ports, e.g. 39031-39062")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--learner-deck", type=Path, default=DEFAULT_LEARNER_DECK)
    parser.add_argument("--opponent-deck-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, default=DEFAULT_RUNTIME_MANIFEST)
    parser.add_argument("--episodes", type=int)
    parser.add_argument(
        "--smoke-workers", type=int,
        help="bounded smoke: use exactly the first N explicit ports (normally 4)",
    )
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--step-ticks", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=12_000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--policy-version", type=int, default=0)
    parser.add_argument("--curriculum-stage", default="stage1_critic")
    parser.add_argument("--opponent-policy-id", default="BASE")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--retain-checkpoints", type=int, default=3)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help=(
            "collect and publish one immutable CLOSED rollout shard without "
            "constructing or updating the Critic"
        ),
    )
    parser.add_argument(
        "--resume-checkpoint", type=Path,
        help="previous Stage-1 checkpoint whose Critic/optimizer/RNG continue this batch",
    )
    return parser


def run(
    args: argparse.Namespace,
    *,
    dependencies: RuntimeDependencies | None = None,
    env_type: type[NativeRoyaleEnv] = NativeRoyaleEnv,
) -> dict[str, Any]:
    collect_only = bool(getattr(args, "collect_only", False))
    opponent_checkpoint = getattr(args, "opponent_checkpoint", None)
    policy_version = int(getattr(args, "policy_version", 0))
    curriculum_stage = str(getattr(args, "curriculum_stage", "stage1_critic"))
    opponent_policy_id = str(getattr(args, "opponent_policy_id", "BASE"))
    ports = parse_ports(args.ports)
    if args.smoke_workers is not None:
        if args.smoke_workers < 1 or args.smoke_workers > len(ports):
            raise ValueError("--smoke-workers must be in 1..len(--ports)")
        ports = ports[: args.smoke_workers]
    episodes = len(ports) if args.episodes is None else int(args.episodes)
    if episodes < 1 or episodes > len(ports):
        raise ValueError("--episodes must be in 1..number of selected Worker ports")
    if args.updates != 1:
        raise ValueError("Stage-1 requires exactly one fresh rollout batch per update")
    if not 1 <= args.step_ticks <= 16:
        raise ValueError("--step-ticks must be in 1..16")
    if args.max_decisions < 1 or args.timeout <= 0 or args.cpu_threads < 1:
        raise ValueError("invalid runtime limit")
    if args.retain_checkpoints < 1:
        raise ValueError("--retain-checkpoints must be positive")
    if collect_only and args.resume_checkpoint is not None:
        raise ValueError("--collect-only does not accept --resume-checkpoint")
    if opponent_checkpoint is not None and not collect_only:
        raise ValueError("--opponent-checkpoint is supported only with --collect-only")
    if policy_version < 0 or not curriculum_stage or not opponent_policy_id:
        raise ValueError("invalid PPO policy/curriculum identity")

    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    journal = RunJournal(run_dir)
    journal.progress("initializing", episodes=episodes, workers=len(ports), updates=args.updates)
    journal.event("run_started", run_dir=str(run_dir), ports=ports)

    envs: list[NativeRoyaleEnv] = []
    ledger: RolloutLedger | None = None
    try:
        torch.set_num_threads(args.cpu_threads)
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        native_runtime = _runtime_identity(args.runtime_manifest)
        loaded = load_base(args.checkpoint, args.expert_manifest, device=device)
        initial_actor_sha256 = loaded.actor_sha256
        initial_hidden_sha256 = _initial_hidden_sha256(loaded.actor)
        opponent_loaded = (
            loaded
            if opponent_checkpoint is None
            else load_base(opponent_checkpoint, args.expert_manifest, device=device)
        )
        opponent_actor_sha256 = opponent_loaded.actor_sha256
        opponent_paths = sorted(args.opponent_deck_root.resolve(strict=True).glob("deck-*.json"))
        scheduler = DeckScheduler(
            learner_preset=args.learner_deck.resolve(strict=True),
            opponent_presets=opponent_paths,
        )
        fixtures = scheduler.build_batch(episode_count=episodes, seed=args.seed)

        envs = [
            env_type(host=args.host, port=port, timeout=args.timeout, profile_native=True)
            for port in ports[:episodes]
        ]
        with ThreadPoolExecutor(max_workers=len(envs)) as executor:
            worker_status = list(executor.map(_preflight_env, envs))
        journal.event("workers_preflight_complete", workers=len(worker_status))

        policy = BatchedPolicyService(device=device, seed=args.seed)
        registered = policy.register_actor(
            loaded.actor, actor_sha256=initial_actor_sha256, verify_content=True
        )
        if registered != initial_actor_sha256 or len(policy.registered_actor_hashes) != 1:
            raise RuntimeError("BASE Actor registry identity diverged")
        if opponent_actor_sha256 != initial_actor_sha256:
            opponent_registered = policy.register_actor(
                opponent_loaded.actor,
                actor_sha256=opponent_actor_sha256,
                verify_content=True,
            )
            if opponent_registered != opponent_actor_sha256:
                raise RuntimeError("opponent Actor registry identity diverged")

        deps = dependencies or runtime_dependencies()
        collector = deps.collector_type(
            encoder=loaded.encoder,
            policy_service=policy,
            reward=DefensiveTowerReward(),
            max_decisions=args.max_decisions,
            rpc_workers=len(envs),
            step_ticks=args.step_ticks,
        )
        trainer = None
        if not collect_only:
            critic_config = PrivilegedCriticConfig(
                actor_latent_size=loaded.config.hidden_size,
                card_vocab_size=loaded.config.card_vocab_size,
                public_grid_channels=loaded.config.grid_channels,
                entity_numeric_size=loaded.config.entity_numeric_size,
                scalar_size=32,
            )
            model = ExpertActorCritic(loaded.actor, PrivilegedCritic(critic_config))
            trainer_config = deps.trainer_config_type(
                retain_checkpoints=args.retain_checkpoints
            )
            trainer = deps.trainer_type(
                model,
                config=trainer_config,
                device=device,
                actor_source_reference={
                    "checkpoint_path": str(args.checkpoint.resolve()),
                    "checkpoint_sha256": loaded.checkpoint_sha256,
                    "expert_manifest_sha256": loaded.expert_manifest_sha256,
                    "actor_state_sha256": initial_actor_sha256,
                },
            )
            # Validate and restore continuation state before spending minutes on a
            # fresh native rollout batch.  A bad checkpoint must fail cheaply.
            if args.resume_checkpoint is not None:
                resume_path = args.resume_checkpoint.resolve(strict=True)
                restored_metrics = trainer.restore_checkpoint(resume_path)
                journal.event(
                    "critic_checkpoint_restored",
                    checkpoint=str(resume_path),
                    checkpoint_sha256=sha256_file(resume_path),
                    restored_metrics=restored_metrics,
                )

        run_id = run_dir.name
        if not _SAFE_ID.fullmatch(run_id):
            raise ValueError("run directory basename is not a safe run_id")
        batch_id = "batch-000001"
        batch_manifest = BatchManifest(
            run_id=run_id,
            batch_id=batch_id,
            policy_version=policy_version,
            behavior_actor_sha256=initial_actor_sha256,
            encoder_schema_sha256=loaded.encoder.schema_sha256(),
            action_schema_sha256=canonical_schema_hash(ACTION_SCHEMA),
            reward_schema_sha256=canonical_schema_hash(_reward_schema()),
            native_lib_sha256=native_runtime["libg_sha256"],
            episode_count=episodes,
        )
        batch_manifest.validate()
        headers = [
            EpisodeHeader(
                episode_id=f"episode-{index:08d}",
                batch_id=batch_id,
                seed=int(fixture.replay["rndSeed"]),
                learner_side=fixture.learner_side,
                behavior_policy_version=policy_version,
                behavior_actor_sha256=initial_actor_sha256,
                opponent_policy_id=opponent_policy_id,
                opponent_actor_sha256=opponent_actor_sha256,
                learner_deck_sha256=fixture.learner_deck_sha256,
                opponent_deck_sha256=fixture.opponent_deck_sha256,
                curriculum_stage=curriculum_stage,
                initial_hidden_sha256=initial_hidden_sha256,
            )
            for index, fixture in enumerate(fixtures)
        ]
        specs = [
            _episode_spec(
                deps.episode_spec_type,
                env=env,
                fixture=fixture,
                header=header,
                learner_actor_sha256=initial_actor_sha256,
                opponent_actor_sha256=opponent_actor_sha256,
            )
            for env, fixture, header in zip(envs, fixtures, headers, strict=True)
        ]

        run_manifest = {
            "kind": RUN_KIND,
            "created_utc": utc_now(),
            "run_id": run_id,
            "stage": curriculum_stage,
            "real_native_rollouts_only": True,
            "collection_only": collect_only,
            "worker_process_management": False,
            "host": args.host,
            "ports": ports[:episodes],
            "episodes": episodes,
            "step_ticks": args.step_ticks,
            "updates": args.updates,
            "resume_checkpoint": (
                None
                if args.resume_checkpoint is None
                else {
                    "path": str(args.resume_checkpoint.resolve(strict=True)),
                    "sha256": sha256_file(args.resume_checkpoint.resolve(strict=True)),
                }
            ),
            "seed": args.seed,
            "policy_version": policy_version,
            "device": str(device),
            "actor_dtype": "float16" if device.type == "cuda" else "float32",
            "critic_autocast_dtype": "bfloat16" if device.type == "cuda" else "float32",
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "file_sha256": loaded.checkpoint_sha256,
                "global_step": loaded.checkpoint_step,
                "run_id": loaded.checkpoint_run_id,
                "actor_state_sha256": initial_actor_sha256,
            },
            "opponent_checkpoint": {
                "path": str(
                    (args.checkpoint if opponent_checkpoint is None else opponent_checkpoint).resolve()
                ),
                "file_sha256": opponent_loaded.checkpoint_sha256,
                "actor_state_sha256": opponent_actor_sha256,
                "policy_id": opponent_policy_id,
            },
            "expert_manifest": {
                "path": str(args.expert_manifest.resolve()),
                "sha256": loaded.expert_manifest_sha256,
            },
            "native_runtime": native_runtime,
            "batch_manifest": asdict(batch_manifest),
            "learner_deck": {
                "path": str(args.learner_deck.resolve()),
                "sha256": file_sha256(args.learner_deck.resolve()),
            },
            "opponent_deck_pool": {
                "root": str(args.opponent_deck_root.resolve()),
                "count": len(opponent_paths),
                "sha256": canonical_schema_hash({
                    str(path.name): file_sha256(path) for path in opponent_paths
                }),
            },
            "action_schema": ACTION_SCHEMA,
            "reward_schema": _reward_schema(),
            "worker_preflight": worker_status,
        }
        atomic_json(run_dir / "manifest.json", run_manifest)

        ledger = RolloutLedger(run_dir / "rollout-ledger.sqlite")
        ledger.open_batch(
            batch_id, policy_version=policy_version, actor_sha256=initial_actor_sha256
        )
        ledger.transition(batch_id, "COLLECTING")
        journal.progress("collecting", episodes_completed=0, episodes=episodes)
        journal.event("collection_started", batch_id=batch_id, episodes=episodes)
        collected = collector.collect_batch(specs)
        if len(collected) != episodes:
            raise RuntimeError(
                f"collector returned {len(collected)}/{episodes} requested episodes"
            )

        frozen_episodes: list[dict[str, Any]] = []
        payloads_by_episode: dict[str, Sequence[Any]] = {}
        chunks: list[dict[str, Any]] = []
        chunker = LearnerEpisodeChunker()
        for row in collected:
            frozen, step_payloads = _complete_episode(row)
            episode_id = str(frozen["header"]["episode_id"])
            if episode_id in payloads_by_episode:
                raise RuntimeError(f"collector returned duplicate episode: {episode_id}")
            frozen_episodes.append(frozen)
            payloads_by_episode[episode_id] = step_payloads
            chunks.extend(chunker.chunk(frozen, step_payloads=step_payloads))
        if {str(row["header"]["episode_id"]) for row in frozen_episodes} != {
            header.episode_id for header in headers
        }:
            raise RuntimeError("collector episode identities differ from scheduled batch")

        writer = ImmutableRolloutShardWriter(
            run_dir / "rollouts", batch_manifest, ledger=ledger, chunker=chunker
        )
        shard = writer.write(
            "shard-000001",
            frozen_episodes,
            step_payloads_by_episode=payloads_by_episode,
        )
        verified = verify_rollout_shard(
            shard.directory, expected_batch_manifest=batch_manifest
        )
        ledger.close_collection(batch_id, minimum_shards=1)
        journal.event(
            "collection_complete",
            episodes=episodes,
            chunks=len(chunks),
            decisions=sum(len(row["decisions"]) for row in frozen_episodes),
            shard_content_sha256=verified["content_sha256"],
        )

        if actor_state_digest(loaded.actor) != initial_actor_sha256:
            raise RuntimeError("BASE Actor changed during rollout collection")
        if actor_state_digest(opponent_loaded.actor) != opponent_actor_sha256:
            raise RuntimeError("opponent Actor changed during rollout collection")
        if collect_only:
            collection_result = {
                "kind": "cr_native_expert_selfplay_stage1_collection_v1",
                "status": "collected",
                "run_id": run_id,
                "episodes": episodes,
                "decisions": sum(len(row["decisions"]) for row in frozen_episodes),
                "chunks": len(chunks),
                "actor_sha256": initial_actor_sha256,
                "shard": str(shard.directory),
                "shard_content_sha256": shard.content_sha256,
                "ledger_state": ledger.state(batch_id),
            }
            if collection_result["ledger_state"] != "CLOSED":
                raise RuntimeError("collect-only batch did not stop in CLOSED")
            atomic_json(run_dir / "collection-result.json", collection_result)
            journal.progress("collected", **{
                key: value for key, value in collection_result.items()
                if key not in ("kind", "status")
            })
            journal.event(
                "collection_run_completed",
                result_path=str(run_dir / "collection-result.json"),
            )
            return collection_result

        if trainer is None:
            raise RuntimeError("Stage-1 trainer was not constructed")
        ledger.transition(batch_id, "UPDATING")
        journal.progress("training", updates_completed=0, updates=args.updates, chunks=len(chunks))
        metrics: list[dict[str, Any]] = []
        checkpoint_path: Path | None = None
        for update in range(args.updates):
            result = dict(trainer.train_update(chunks))
            if actor_state_digest(loaded.actor) != initial_actor_sha256:
                raise RuntimeError("Stage-1 Critic update mutated the frozen BASE Actor")
            checkpoint_path = Path(
                trainer.save_checkpoint(run_dir / "checkpoints", metrics=result)
            )
            if not checkpoint_path.is_file():
                raise RuntimeError("Critic trainer did not publish its checkpoint")
            metrics.append(result)
            journal.event(
                "critic_update_complete",
                update=update + 1,
                metrics=result,
                checkpoint=str(checkpoint_path),
            )
            journal.progress(
                "training",
                updates_completed=update + 1,
                updates=args.updates,
                chunks=len(chunks),
            )

        ledger.transition(batch_id, "VALIDATING")
        final_actor_sha256 = actor_state_digest(loaded.actor)
        if final_actor_sha256 != initial_actor_sha256:
            raise RuntimeError("Actor hash changed before Stage-1 commit")
        if checkpoint_path is None:
            raise RuntimeError("Stage-1 produced no Critic checkpoint")
        ledger.commit(batch_id)
        result = {
            "kind": "cr_native_expert_selfplay_stage1_result_v1",
            "status": "completed",
            "run_id": run_id,
            "episodes": episodes,
            "decisions": sum(len(row["decisions"]) for row in frozen_episodes),
            "chunks": len(chunks),
            "updates": args.updates,
            "actor_sha256_before": initial_actor_sha256,
            "actor_sha256_after": final_actor_sha256,
            "actor_unchanged": True,
            "shard": str(shard.directory),
            "shard_content_sha256": shard.content_sha256,
            "checkpoint": str(checkpoint_path),
            "metrics": metrics,
            "ledger_state": ledger.state(batch_id),
        }
        atomic_json(run_dir / "result.json", result)
        journal.progress("completed", **{
            key: value for key, value in result.items()
            if key not in ("kind", "status", "metrics")
        })
        journal.event("run_completed", result_path=str(run_dir / "result.json"))
        return result
    except BaseException as error:
        journal.progress(
            "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        journal.event(
            "run_failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback="".join(traceback.format_exception(error)),
        )
        raise
    finally:
        if ledger is not None:
            ledger.close()
        for env in envs:
            env.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
