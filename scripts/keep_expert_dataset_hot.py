"""Keep a bounded, read-only subset of training files resident on Linux.

Shares the SAME file-backed pages as the live mmap DataLoader. No RAM copy,
dataset path replacement, model change, or trainer restart is performed.
The helper releases all locks when its trainer exits or it receives SIGTERM.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

GIB = 1024 ** 3
HOT_FIELDS = (
    'public_scalars.npy', 'grid_indices.npy', 'grid_values.npy',
    'entity_numeric.npy', 'entity_offsets.npy', 'grid_offsets.npy',
    'entity_tokens.npy', 'entity_positions.npy', 'entity_relations.npy',
    'own_deck_tokens.npy', 'hand_tokens.npy', 'next_card_token.npy',
    'card_mask.npy', 'revealed_enemy_tokens.npy', 'ability_tokens.npy',
)


def process_identity(pid: int) -> str | None:
    try:
        # starttime prevents a later reused PID from keeping the cache alive.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def number(path: Path, fallback: int) -> int:
    text = path.read_text().strip()
    return fallback if text == 'max' else int(text)


def memory_status() -> dict[str, int]:
    host = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        host[name] = int(value.split()[0]) * 1024
    group = Path('/sys/fs/cgroup')
    return {
        'used': number(group/'memory.current', 0),
        'limit': number(group/'memory.max', host['MemTotal']),
        'host_available': host['MemAvailable'],
    }


def cache_budget(requested: int, reserve: int, memory: dict[str, int]) -> int:
    return max(0, min(requested, memory['limit'] - reserve - memory['used'],
                      memory['host_available'] - reserve))


def candidate_files(dataset: Path, manifest: dict, split: str) -> list[Path]:
    shards = set(manifest['splits'][split])
    priorities = {name: index for index, name in enumerate(HOT_FIELDS)}
    files = []
    for relative in manifest['shard_file_sha256']:
        path = Path(relative)
        if path.suffix != '.npy' or path.parent.as_posix() not in shards:
            continue
        resolved = (dataset/path).resolve()
        if dataset not in resolved.parents:
            raise ValueError(f'cache file escapes dataset: {relative}')
        files.append(resolved)
    return sorted(files, key=lambda path: (priorities.get(path.name, len(priorities)), str(path)))


class ResidentFiles:
    def __init__(self):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
        self.libc.mmap.restype = ctypes.c_void_p
        for name in ('mlock', 'munlock', 'munmap'):
            function = getattr(self.libc, name)
            function.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            function.restype = ctypes.c_int
        self.pages: list[tuple[int, int]] = []
        self.bytes = 0

    def add(self, path: Path, remaining: int) -> None:
        length = min(path.stat().st_size, remaining)
        if length <= 0:
            return
        fd = os.open(path, os.O_RDONLY)
        try:
            # PROT_READ, MAP_SHARED: never create anonymous/COW duplicate data.
            address = self.libc.mmap(None, length, 1, 1, fd, 0)
        finally:
            os.close(fd)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), f'mmap failed: {path}')
        if self.libc.mlock(address, length):
            error = ctypes.get_errno()
            self.libc.munmap(address, length)
            raise OSError(error, f'mlock failed: {path}')
        self.pages.append((address, length))
        self.bytes += length

    def release(self, amount: int) -> None:
        released = 0
        while self.pages and released < amount:
            address, length = self.pages.pop()
            self.libc.munlock(address, length)
            self.libc.munmap(address, length)
            self.bytes -= length
            released += length


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--trainer-pid', type=int, required=True)
    parser.add_argument('--cache-gib', type=float, default=68)
    parser.add_argument('--reserve-gib', type=float, default=12)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    if args.cache_gib <= 0 or args.reserve_gib < 8:
        raise ValueError('positive cache budget and at least 8 GiB reserve required')
    run = json.loads((args.run_root/'manifest.json').read_text())
    dataset = Path(run['dataset_root']).resolve()
    raw_manifest = (dataset/'manifest.json').read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != run['dataset_manifest_sha256']:
        raise ValueError('cache dataset does not match the active run')
    manifest = json.loads(raw_manifest)
    identity = process_identity(args.trainer_pid)
    if identity is None:
        raise RuntimeError('trainer is not running')
    command = Path(f'/proc/{args.trainer_pid}/cmdline').read_bytes()
    if b'expert_v1.training_v1' not in command or run['run_id'].encode() not in command:
        raise RuntimeError('trainer PID does not match requested run')
    args.status.parent.mkdir(parents=True, exist_ok=True)
    lock = args.status.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(10)
    # If the host nevertheless hits OOM, sacrifice this cache before the trainer.
    Path('/proc/self/oom_score_adj').write_text('1000')
    reserve = int(args.reserve_gib * GIB)
    budget = cache_budget(int(args.cache_gib * GIB), reserve, memory_status())
    resident = ResidentFiles()
    started = time.monotonic()
    stopping = False
    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def report(status: str, error: str | None = None):
        value = dict(status=status, pid=os.getpid(), trainer_pid=args.trainer_pid,
                     mode='shared_readonly_file_pages_mlock', cache_bytes=resident.bytes,
                     cache_gib=resident.bytes/GIB, budget_gib=budget/GIB,
                     reserve_gib=args.reserve_gib, mapped_files=len(resident.pages),
                     elapsed_seconds=round(time.monotonic()-started, 2),
                     memory=memory_status(), error=error,
                     dataset_manifest_sha256=run['dataset_manifest_sha256'])
        temporary = args.status.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(args.status)
        print(json.dumps(value), flush=True)
    report('warming')
    try:
        for index, path in enumerate(candidate_files(dataset, manifest, run['training']['train_split'])):
            if stopping or process_identity(args.trainer_pid) != identity or resident.bytes >= budget:
                break
            memory = memory_status()
            if memory['used'] > memory['limit']-reserve or memory['host_available'] < reserve:
                break
            try:
                resident.add(path, budget-resident.bytes)
            except OSError as error:
                report('capacity_limited', str(error))
                if error.errno in (errno.ENOMEM, errno.EAGAIN):
                    break
                raise
            if index % 500 == 0:
                report('warming')
        report('resident')
        while not stopping and process_identity(args.trainer_pid) == identity:
            memory = memory_status()
            if memory['used'] > memory['limit']-reserve or memory['host_available'] < reserve:
                resident.release(2*GIB)
                report('pressure_release')
            else:
                report('resident')
            time.sleep(10)
    except Exception as error:
        report('failed', str(error))
        raise
    finally:
        resident.release(resident.bytes)
        report('stopped')
        lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
