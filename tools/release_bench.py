#!/usr/bin/env python3
"""Measures the release benchmarks on this machine: the CKKS bootstrapping presets,
poulpy-libm functions and poulpy-blas operations.

Run it from the repository root on every machine of the release, then merge the files:

    python3 tools/release_bench.py run --label ryzen-9950x
    python3 tools/release_bench.py run --label ryzen-9950x --suites libm blas
    python3 tools/release_bench.py merge release.json ryzen-9950x.json m5-pro.json

`run` detects the instruction sets and GPU of the host, builds `poulpy-release-bench`
once per set of target flags (each build in its own target directory under
`target/release-bench`), then measures every backend and every preset or case of the
selected suites in its own process, one at a time. Rayon backends run on 16 threads on
x86 and on every thread on ARM, capped at what the host has. The record is rewritten
after every measurement, so an interrupted run keeps what it measured and `--resume`
completes it.

Timing covers the measured call alone (a bootstrap, a libm function or a blas
operation), synchronized with the device on the GPU backends: setup, key generation,
encryption and decryption are excluded and reported apart. Latency is the median of the
timed iterations and spread is the median absolute deviation over the median.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

SCHEMA = "poulpy-release-bench/2"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRATE = "poulpy-release-bench"
# The crate is outside the workspace and has its own lock.
MANIFEST = "tools/poulpy-release-bench/Cargo.toml"
LOCK = os.path.join(ROOT, "tools", "poulpy-release-bench", "Cargo.lock")

# Builds by architecture: name, crate features, target flags, instruction sets the host needs.
# A backend is measured from the first build that contains it, so the portable
# backend always comes from a build without extra target flags.
X86_BUILDS = [
    ("base", ["portable-rayon"], "", []),
    ("avx", ["avx"], "-C target-feature=+avx2,+fma", ["avx2", "fma"]),
    ("avx512", ["avx512"], "-C target-feature=+avx2,+fma,+avx512f", ["avx2", "fma", "avx512f"]),
    (
        "ifma",
        ["ifma"],
        "-C target-feature=+avx2,+fma,+avx512f,+avx512ifma,+avx512vl",
        ["avx2", "fma", "avx512f", "avx512ifma", "avx512vl"],
    ),
]
ARM_BUILDS = [("base", ["portable-rayon", "neon"], "", ["neon"])]
GPU_FEATURES = ["cuda", "hip", "metal", "vulkan"]
SUITES = ["bootstrapping", "libm", "blas"]
X86_RAYON_THREADS = 16


def sh(*cmd, cwd=ROOT):
    """Output of a command, or None when it is missing or fails."""
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- host description


def arch():
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("arm64", "aarch64"):
        return "aarch64"
    return machine


def isa():
    """The instruction sets of the host that select a backend."""
    found = set()
    if arch() == "aarch64":
        found.add("neon")
    elif sys.platform == "darwin":
        flags = " ".join(filter(None, [sh("sysctl", "-n", "machdep.cpu.features"), sh("sysctl", "-n", "machdep.cpu.leaf7_features")]))
        tokens = set(flags.lower().split())
        found |= {f for f in ("avx2", "fma", "avx512f", "avx512vl", "bmi2", "adx") if f in tokens}
        if "avx512ifma" in tokens or "avx512ifma52" in tokens:
            found.add("avx512ifma")
    else:
        cpuinfo = read("/proc/cpuinfo") or ""
        line = next((l for l in cpuinfo.splitlines() if l.startswith("flags")), "")
        tokens = set(line.split(":", 1)[-1].split())
        found |= {f for f in ("avx2", "fma", "avx512f", "avx512ifma", "avx512vl", "bmi2", "adx") if f in tokens}
    return sorted(found)


def cpu():
    info = {"arch": arch(), "logical_cores": os.cpu_count()}
    if sys.platform == "darwin":
        info["model"] = sh("sysctl", "-n", "machdep.cpu.brand_string")
        info["physical_cores"] = int(sh("sysctl", "-n", "hw.physicalcpu") or 0) or None
        for level, name in ((0, "performance_cores"), (1, "efficiency_cores")):
            count = sh("sysctl", "-n", f"hw.perflevel{level}.physicalcpu")
            if count:
                info[name] = int(count)
    else:
        cpuinfo = read("/proc/cpuinfo") or ""
        model = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
        info["model"] = model.group(1).strip() if model else None
        if info["model"] is None:
            lscpu = sh("lscpu") or ""
            model = re.search(r"^Model name:\s*(.+)$", lscpu, re.M)
            info["model"] = model.group(1).strip() if model else None
        cores = sh("lscpu", "-p=core,socket")
        if cores:
            info["physical_cores"] = len({l for l in cores.splitlines() if not l.startswith("#")})
        sockets = sh("lscpu", "-p=socket")
        if sockets:
            info["sockets"] = len({l for l in sockets.splitlines() if not l.startswith("#")})
    info["isa"] = isa()
    return info


def memory_bytes():
    if sys.platform == "darwin":
        return int(sh("sysctl", "-n", "hw.memsize") or 0) or None
    meminfo = read("/proc/meminfo") or ""
    total = re.search(r"^MemTotal:\s*(\d+) kB", meminfo, re.M)
    return int(total.group(1)) * 1024 if total else None


def os_info():
    info = {"system": platform.system(), "kernel": platform.release()}
    if sys.platform == "darwin":
        info["name"] = f"macOS {sh('sw_vers', '-productVersion')} ({sh('sw_vers', '-buildVersion')})"
    else:
        release = read("/etc/os-release") or ""
        name = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', release, re.M)
        info["name"] = name.group(1) if name else platform.platform()
    return info


def power():
    """The power policy: timings on laptops and with frequency scaling depend on it."""
    info = {}
    if sys.platform == "darwin":
        settings = sh("pmset", "-g") or ""
        for key in ("powermode", "lowpowermode", "highpowermode"):
            value = re.search(rf"^\s*{key}\s+(\d+)", settings, re.M)
            if value:
                info[key] = int(value.group(1))
        battery = sh("pmset", "-g", "batt") or ""
        source = re.search(r"drawing from '([^']+)'", battery)
        info["source"] = source.group(1) if source else None
    else:
        base = "/sys/devices/system/cpu"
        info["governor"] = read(f"{base}/cpu0/cpufreq/scaling_governor")
        info["energy_performance_preference"] = read(f"{base}/cpu0/cpufreq/energy_performance_preference")
        info["boost"] = read(f"{base}/cpufreq/boost")
        info["intel_no_turbo"] = read(f"{base}/intel_pstate/no_turbo")
        info["amd_pstate"] = read(f"{base}/amd_pstate/status")
    return {k: v for k, v in info.items() if v is not None}


def power_warnings(info):
    """Power settings known to make timings slower or bimodal."""
    warnings = []
    if sys.platform == "darwin":
        if info.get("powermode", info.get("highpowermode")) not in (None, 2) and info.get("highpowermode") != 1:
            warnings.append("macOS is not in high power mode, timings may be bimodal (sudo pmset -a powermode 2)")
        if info.get("source") == "Battery Power":
            warnings.append("running on battery")
    elif info.get("governor") not in (None, "performance"):
        warnings.append(f"CPU governor is {info['governor']}, not performance")
    return warnings


def gpus():
    """GPUs of the host, best effort per vendor tool."""
    found = []
    smi = sh("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
    for line in (smi or "").splitlines():
        name, driver, mem = (part.strip() for part in line.split(","))
        found.append({"vendor": "NVIDIA", "name": name, "driver": driver, "memory": mem, "cuda": cuda_version()})
    rocm = sh("rocminfo")
    if rocm:
        for agent in rocm.split("*******")[1:]:
            if re.search(r"Device Type:\s*GPU", agent):
                name = re.search(r"Marketing Name:\s*(.+)", agent)
                arch_name = re.search(r"^\s*Name:\s*(gfx\w+)", agent, re.M)
                found.append(
                    {
                        "vendor": "AMD",
                        "name": name.group(1).strip() if name else None,
                        "arch": arch_name.group(1) if arch_name else None,
                        "rocm": read("/opt/rocm/.info/version"),
                    }
                )
    if sys.platform == "darwin":
        try:
            displays = json.loads(sh("system_profiler", "-json", "SPDisplaysDataType") or "{}")
        except json.JSONDecodeError:
            displays = {}
        for gpu in displays.get("SPDisplaysDataType", []):
            found.append(
                {
                    "vendor": gpu.get("spdisplays_vendor", "").replace("sppci_vendor_", "") or None,
                    "name": gpu.get("sppci_model"),
                    "cores": int(gpu["sppci_cores"]) if str(gpu.get("sppci_cores", "")).isdigit() else None,
                    "metal": gpu.get("spdisplays_mtlgpufamilysupport"),
                }
            )
    return found


def cuda_version():
    nvcc = sh("nvcc", "--version") or ""
    version = re.search(r"release ([\d.]+)", nvcc)
    return version.group(1) if version else None


def default_gpu_feature():
    if sys.platform == "darwin" and arch() == "aarch64":
        return "metal"
    if shutil.which("nvidia-smi") and shutil.which("nvcc"):
        return "cuda"
    if shutil.which("rocminfo") and (shutil.which("hipcc") or os.path.exists("/opt/rocm")):
        return "hip"
    return None


def git_info():
    status = sh("git", "status", "--porcelain", "--untracked-files=no")
    with open(LOCK, "rb") as f:
        lock_sha = hashlib.sha256(f.read()).hexdigest()
    manifest = read(os.path.join(ROOT, "Cargo.toml")) or ""
    version = re.search(r'\[workspace\.package\][^\[]*?version\s*=\s*"([^"]+)"', manifest, re.S)
    return {
        "version": version.group(1) if version else None,
        "commit": sh("git", "rev-parse", "HEAD"),
        "describe": sh("git", "describe", "--tags", "--always", "--dirty"),
        "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "cargo_lock_sha256": lock_sha,
    }


def toolchain():
    rustc = sh("rustc", "-vV") or ""
    fields = dict(line.split(": ", 1) for line in rustc.splitlines()[1:] if ": " in line)
    lock = read(LOCK) or ""
    cubecl = re.search(r'name = "cubecl"\nversion = "([^"]+)"', lock)
    return {
        "rustc": rustc.splitlines()[0] if rustc else None,
        "rustc_commit": fields.get("commit-hash"),
        "llvm": fields.get("LLVM version"),
        "host_triple": fields.get("host"),
        "cargo": sh("cargo", "-V"),
        "rust_toolchain_toml": read(os.path.join(ROOT, "rust-toolchain.toml")),
        "cubecl": cubecl.group(1) if cubecl else None,
        "profile": "release",
    }


def default_threads():
    logical = os.cpu_count() or 1
    return min(X86_RAYON_THREADS, logical) if arch() == "x86_64" else logical


# ---------------------------------------------------------------- builds and runs


def plan_builds(host_isa, gpu_feature):
    builds = []
    for name, features, rustflags, needs in X86_BUILDS if arch() == "x86_64" else ARM_BUILDS:
        if not all(flag in host_isa for flag in needs):
            continue
        features = list(features)
        if name == "base" and gpu_feature:
            features.append(gpu_feature)
        builds.append({"name": name, "features": features, "rustflags": rustflags})
    return builds


def build(spec):
    target_dir = os.path.join(ROOT, "target", "release-bench", spec["name"])
    cmd = ["cargo", "build", "--locked", "--release", "--manifest-path", MANIFEST, "--features", ",".join(spec["features"])]
    env = dict(os.environ, CARGO_TARGET_DIR=target_dir, RUSTFLAGS=spec["rustflags"])
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    print(f"== build {spec['name']}: {' '.join(cmd)}  RUSTFLAGS={spec['rustflags']!r}", flush=True)
    start = time.monotonic()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    binary = os.path.join(target_dir, "release", CRATE)
    listed = json.loads(subprocess.run([binary, "list"], capture_output=True, text=True, check=True).stdout)
    return dict(
        spec,
        command=" ".join(cmd),
        binary=os.path.relpath(binary, ROOT),
        build_seconds=round(time.monotonic() - start, 1),
        backends=listed["backends"],
        presets=listed["presets"],
        libm=listed["libm"],
        blas=listed["blas"],
    )


def run_one(binary, args, env, timeout):
    """Runs one measurement and returns (record or None, error, peak RSS, wall seconds)."""
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        start = time.monotonic()
        proc = subprocess.Popen([binary, "run", *args], cwd=ROOT, env=env, stdout=out, stderr=err)
        deadline = start + timeout if timeout else None
        while True:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if deadline and time.monotonic() > deadline:
                proc.kill()
                pid, status, usage = os.wait4(proc.pid, 0)
                break
            time.sleep(0.2)
        proc.returncode = os.waitstatus_to_exitcode(status)
        wall = round(time.monotonic() - start, 1)
        # ru_maxrss is in bytes on macOS and in KiB on Linux.
        rss = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    if deadline and time.monotonic() > deadline and proc.returncode != 0:
        return None, f"timed out after {timeout}s", rss, wall
    if proc.returncode != 0:
        tail = "\n".join((stderr or stdout).strip().splitlines()[-20:])
        return None, f"exit status {proc.returncode}: {tail}", rss, wall
    try:
        return json.loads(stdout.strip().splitlines()[-1]), None, rss, wall
    except (json.JSONDecodeError, IndexError):
        return None, f"unreadable output: {stdout[-500:]}", rss, wall


def timing_stats(samples):
    median = statistics.median(samples)
    mad = statistics.median(abs(s - median) for s in samples)
    return {
        "median_ns": median,
        "mad_ns": mad,
        "spread": mad / median if median else None,
        "mean_ns": statistics.fmean(samples),
        "stdev_ns": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ns": min(samples),
        "max_ns": max(samples),
    }


def result_id(backend, suite, case, pow2, mode):
    if suite == "bootstrapping":
        return f"{backend}/{suite}/{case}/{'pow2' if pow2 else 'default'}/{mode}"
    return f"{backend}/{suite}/{case}"


def case_key(r):
    """What a result measures, whatever the backend: its suite, preset or case, and key variant."""
    if r["suite"] == "bootstrapping":
        return ("bootstrapping", r["preset"]["name"], r["preset"]["pow2_keys"])
    return (r["suite"], r[r["suite"]]["case"], None)


def meets(r):
    """Whether a result reached its precision: the advertised one of a preset, the target of a case."""
    precision = r["precision"]
    return precision["meets_advertised"] if r["suite"] == "bootstrapping" else precision["meets_target"]


def derive(results):
    """Speedups against the serial portable backend and, for Rayon, against the
    serial counterpart, on this host for the same preset or case, key variant and
    mode, and the time per matrix of a blas case."""
    medians = {
        (r["backend"]["backend"], *case_key(r)): r["timing"]["median_ns"]
        for r in results
        if r["status"] == "ok" and r["mode"] == "direct"
    }
    for r in results:
        if r["status"] != "ok":
            continue
        key = case_key(r)
        median = r["timing"]["median_ns"]
        portable = medians.get(("NTT4x30Portable", *key))
        serial_name = r["backend"]["backend"].removesuffix("Rayon")
        serial = medians.get((serial_name, *key)) if serial_name != r["backend"]["backend"] else None
        r["derived"] = {
            "speedup_vs_portable": portable / median if portable else None,
            "speedup_vs_serial": serial / median if serial else None,
        }
        if r["suite"] == "blas":
            r["derived"]["median_ns_per_matrix"] = median / r["blas"]["matrices"]


def write(path, record):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def cmd_run(opts):
    os.chdir(ROOT)
    poulpy = git_info()
    if poulpy["dirty"] and not opts.allow_dirty:
        sys.exit("tracked files are modified: commit them or pass --allow-dirty")
    host_isa = isa()
    suites = opts.suites or SUITES
    gpu_feature = None if opts.gpu == "none" else (default_gpu_feature() if opts.gpu == "auto" else opts.gpu)
    # The GPU backends run the bootstrapping and libm suites only.
    if "bootstrapping" not in suites and "libm" not in suites:
        gpu_feature = None
    threads = min(opts.threads or default_threads(), os.cpu_count() or 1)
    builds = plan_builds(host_isa, gpu_feature)
    label = opts.label or platform.node().split(".")[0]
    out = opts.out or os.path.join("target", "release-bench", f"{label}-{(poulpy['commit'] or 'unknown')[:10]}.json")
    pow2_variants = {"both": [False, True], "on": [True], "off": [False]}[opts.pow2]

    print(f"host {label}: {cpu().get('model')}, isa {','.join(host_isa) or 'none'}, gpu feature {gpu_feature}")
    print(f"builds: {', '.join(b['name'] + '[' + ','.join(b['features']) + ']' for b in builds)}")
    print(f"suites {', '.join(suites)}, rayon threads {threads}, warmup {opts.warmup}, iterations {opts.iters}, output {out}")
    for warning in power_warnings(power()):
        print(f"warning: {warning}")
    if opts.dry_run:
        return

    previous = {}
    if opts.resume and os.path.exists(out):
        with open(out) as f:
            old = json.load(f)
        if old.get("poulpy", {}).get("commit") != poulpy["commit"]:
            sys.exit(f"{out} was measured at another commit")
        previous = {r["id"]: r for r in old["results"] if r["status"] == "ok"}

    built = [build(spec) for spec in builds]

    # Every backend from the first build that contains it.
    jobs, seen = [], set()
    for b in built:
        for backend in b["backends"]:
            name = backend["backend"]
            if name in seen or (opts.backends and name not in opts.backends):
                continue
            seen.add(name)
            modes = ["direct", "replay"] if backend["parallelism"] == "device" and not opts.no_replay else ["direct"]
            for suite in suites:
                if suite not in backend["suites"]:
                    continue
                if suite == "bootstrapping":
                    for preset in opts.presets or b["presets"]:
                        for pow2 in pow2_variants:
                            for mode in modes:
                                jobs.append((b, backend, suite, preset, pow2, mode))
                else:
                    for case in getattr(opts, suite) or b[suite]:
                        jobs.append((b, backend, suite, case, None, "direct"))

    record = {
        "schema": SCHEMA,
        "started_at": now(),
        "finished_at": None,
        "label": label,
        "command": " ".join(sys.argv),
        "poulpy": poulpy,
        "host": {
            "cpu": cpu(),
            "memory_bytes": memory_bytes(),
            "os": os_info(),
            "power": power(),
            "power_warnings": power_warnings(power()),
            "gpus": gpus(),
        },
        "toolchain": toolchain(),
        "builds": [{k: v for k, v in b.items() if k not in ("backends", "presets", "libm", "blas")} for b in built],
        "method": {
            "suites": suites,
            "warmup": opts.warmup,
            "iterations": opts.iters,
            "rayon_threads": threads,
            "timed_region": "one bootstrap, libm function or blas operation call, followed by a device synchronization",
            "excluded": "setup (context compilation, plans, key generation and preparation, encryption, one checking call), reported as setup_ns, and decryption",
            "statistics": "latency is the median of the timed iterations, spread is the median absolute deviation over the median",
            "process": "one process per measurement, run sequentially",
            "load_average_start": os.getloadavg(),
        },
        "results": [],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write(out, record)

    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    for index, (b, backend, suite, case, pow2, mode) in enumerate(jobs, 1):
        rid = result_id(backend["backend"], suite, case, pow2, mode)
        if rid in previous:
            record["results"].append(previous[rid])
            print(f"[{index}/{len(jobs)}] {rid}: kept from {out}")
            continue
        flag = {"bootstrapping": "--preset", "libm": "--libm", "blas": "--blas"}[suite]
        args = ["--backend", backend["backend"], flag, case, "--warmup", str(opts.warmup), "--iters", str(opts.iters)]
        args += ["--pow2"] if pow2 else []
        args += ["--replay"] if mode == "replay" else []
        args += ["--device", str(opts.device)] if opts.device is not None and backend["parallelism"] == "device" else []
        print(f"[{index}/{len(jobs)}] {rid} ...", end=" ", flush=True)
        measured, error, rss, wall = run_one(os.path.join(ROOT, b["binary"]), args, env, opts.timeout)
        entry = {"id": rid, "build": b["name"], "status": "ok" if measured else "error", "peak_rss_bytes": rss, "wall_seconds": wall}
        if measured and "unavailable" in measured:
            entry.update(measured)
            entry["status"] = "unavailable"
            print(f"no graph replay: {measured['unavailable']}, the direct measurement stands")
        elif measured:
            entry.update(measured)
            entry["timing"] = timing_stats(measured["samples_ns"])
            median = entry["timing"]["median_ns"] / 1e6
            flag = "" if meets(entry) else "  BELOW TARGET PRECISION"
            print(f"median {median:.1f} ms, spread {entry['timing']['spread']:.2%}{flag}")
        else:
            entry.update({"suite": suite, "backend": backend, "mode": mode, "error": error})
            if suite == "bootstrapping":
                entry["preset"] = {"name": case, "pow2_keys": pow2}
            else:
                entry[suite] = {"case": case}
            print(f"FAILED: {error.splitlines()[0] if error else ''}")
        record["results"].append(entry)
        derive(record["results"])
        write(out, record)

    record["finished_at"] = now()
    record["method"]["load_average_end"] = os.getloadavg()
    write(out, record)
    failed = [r["id"] for r in record["results"] if r["status"] == "error"]
    unavailable = [r["id"] for r in record["results"] if r["status"] == "unavailable"]
    low = [r["id"] for r in record["results"] if r["status"] == "ok" and not meets(r)]
    measured = len(record["results"]) - len(failed) - len(unavailable)
    print(f"wrote {out}: {measured} measured, {len(failed)} failed, {len(unavailable)} without graph replay")
    for rid in failed:
        print(f"  failed: {rid}")
    for rid in unavailable:
        print(f"  no graph replay: {rid}")
    for rid in low:
        print(f"  below target precision: {rid}")


def cmd_merge(opts):
    runs = []
    for path in opts.inputs:
        with open(path) as f:
            runs.append(json.load(f))
    commits = {run["poulpy"]["commit"] for run in runs}
    if len(commits) > 1 and not opts.allow_mixed:
        sys.exit(f"the inputs were measured at different commits: {', '.join(sorted(commits))}")
    labels = [run["label"] for run in runs]
    if len(set(labels)) != len(labels):
        sys.exit(f"two inputs share a label: {labels}")
    merged = {
        "schema": SCHEMA,
        "merged_at": now(),
        "poulpy": runs[0]["poulpy"] if len(commits) == 1 else None,
        "runs": runs,
    }
    write(opts.output, merged)
    print(f"wrote {opts.output}: {len(runs)} hosts, {sum(len(r['results']) for r in runs)} results")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="measure the release benchmarks on this host")
    run.add_argument("--label", help="name of this host in the record (default: hostname)")
    run.add_argument("--out", help="output file (default: target/release-bench/<label>-<commit>.json)")
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--iters", type=int, default=20)
    run.add_argument("--threads", type=int, help="Rayon threads (default: 16 on x86 and all on ARM, capped at the host)")
    run.add_argument("--gpu", choices=["auto", "none", *GPU_FEATURES], default="auto", help="GPU runtime to build")
    run.add_argument("--device", type=int, help="index of the GPU among those of its runtime")
    run.add_argument("--no-replay", action="store_true", help="skip the replayed GPU measurements")
    run.add_argument("--backends", nargs="+", help="only these backends")
    run.add_argument("--suites", nargs="+", choices=SUITES, help="only these suites (default: all)")
    run.add_argument("--presets", nargs="+", help="only these bootstrapping presets")
    run.add_argument("--libm", nargs="+", help="only these libm cases")
    run.add_argument("--blas", nargs="+", help="only these blas cases")
    run.add_argument("--pow2", choices=["both", "on", "off"], default="both", help="power-of-two rotation keys")
    run.add_argument("--timeout", type=int, help="seconds before a measurement is killed")
    run.add_argument("--resume", action="store_true", help="keep the measurements already in --out")
    run.add_argument("--allow-dirty", action="store_true", help="measure with modified tracked files")
    run.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    run.set_defaults(func=cmd_run)

    merge = sub.add_parser("merge", help="combine the records of several hosts")
    merge.add_argument("output")
    merge.add_argument("inputs", nargs="+")
    merge.add_argument("--allow-mixed", action="store_true", help="accept inputs from different commits")
    merge.set_defaults(func=cmd_merge)

    opts = parser.parse_args()
    opts.func(opts)


if __name__ == "__main__":
    main()
