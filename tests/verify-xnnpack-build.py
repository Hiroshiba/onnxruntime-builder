#!/usr/bin/env python3
"""Reject an incorrectly configured/fallback-only XNNPACK experiment."""
import argparse
import json
import pathlib
import shlex


def verify_compile_commands(path):
    commands = json.loads(path.read_text())
    required = {"-pthread", "-msimd128", "-fno-fast-math", "-ffp-contract=off", "-fwasm-exceptions"}
    forbidden = {"-ffast-math", "-Ofast", "-mrelaxed-simd", "-mfp16", "-ffinite-math-only", "-funsafe-math-optimizations", "-fassociative-math"}
    counts = {"total": 0, "xnnpack": 0, "pthreadpool": 0, "wasmsimd": 0}
    pool_sources = set()
    for entry in commands:
        source = entry["file"]
        if "pthreadpool-src" in source:
            pool_sources.add(pathlib.Path(source).name)
            if source.endswith("/shim.c"):
                raise ValueError("Serial pthreadpool Emscripten shim was configured")
        args = entry.get("arguments") or shlex.split(entry["command"])
        flags = set(args)
        if flags & forbidden:
            raise ValueError(f"Forbidden math/ISA flags in {source}: {flags & forbidden}")
        if source.endswith((".c", ".cc", ".cpp", ".cxx")):
            if not required <= flags:
                raise ValueError(f"Missing required flags in {source}: {required - flags}")
            counts["total"] += 1
            counts["xnnpack"] += "googlexnnpack-src" in source
            counts["pthreadpool"] += "pthreadpool-src" in source
            counts["wasmsimd"] += "-wasmsimd" in pathlib.Path(source).name
        if "wasmrelaxed" in source:
            raise ValueError("Relaxed SIMD source was configured")
    if not all(counts.values()):
        raise ValueError(f"Missing expected compilation units: {counts}")
    if not {"pthreads.c", "portable-api.c", "memory.c"} <= pool_sources:
        raise ValueError(f"Real pthreadpool backend missing: {pool_sources}")
    return {"passed": True, "required_flags": sorted(required), "source_counts": counts,
            "pthreadpool_sources": sorted(pool_sources), "pthreadpool_backend": "pthreads"}


def verify_smoke(path):
    text = path.read_text()
    profile_text = text.split("PROFILE_JSON_BEGIN\n", 1)[1].split("\nPROFILE_JSON_END", 1)[0]
    profile = json.loads(profile_text)
    assignments = {}
    for event in profile:
        args = event.get("args", {})
        if event.get("cat") == "Node" and args.get("provider"):
            assignments.setdefault(args["op_name"], set()).add(args["provider"])
    for op in ("Conv", "ConvTranspose"):
        if assignments.get(op) != {"XnnpackExecutionProvider"}:
            raise ValueError(f"{op} was not exclusively executed by XNNPACK: {assignments}")
    counts_line = next(line.removeprefix("THREAD_COUNTS ") for line in text.splitlines() if line.startswith("THREAD_COUNTS "))
    counts = json.loads(counts_line)
    expected = {"ort_fallback_workers": 0, "xnnpack_workers": 1, "xnnpack_intra_op_threads": 2,
                "ort_global_intra_op_threads": 1, "ort_global_inter_op_threads": 1}
    if counts != expected or "PASS: pthread create/join" not in text:
        raise ValueError(f"Thread budget/reference check did not pass: {counts}")
    return {"passed": True, "execution_provider": "XnnpackExecutionProvider",
            "operators": {op: sorted(eps) for op, eps in assignments.items()},
            "thread_counts": counts, "profile": profile}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "smoke"))
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    result = (verify_compile_commands if args.mode == "compile" else verify_smoke)(args.input)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: XNNPACK {args.mode} verification")
