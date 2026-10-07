#!/usr/bin/env python3
"""Read-only validation of the exact successful CI archive before publication."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, description: str) -> None:
    if not condition:
        raise ValueError(description)


def verify(folder: Path) -> None:
    name = os.environ["RELEASE_NAME"]
    archive = folder / f"{name}.tgz"
    require(sha256(archive.read_bytes()) == os.environ["EXPECTED_ARCHIVE_SHA256"],
            "Compressed archive digest differs from reviewed artifact")
    checksum = (folder / f"{name}.tgz.sha256").read_text().split()
    require(checksum == [os.environ["EXPECTED_ARCHIVE_SHA256"], f"{name}.tgz"],
            "Companion checksum does not identify the exact reviewed archive")

    # Inspect in place: do not extract or execute anything from the archive.
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and ".." not in path.parts,
                    "Unsafe archive path")
            require(path.parts and path.parts[0] == name, "Unexpected archive root")
            require(member.isfile() or member.isdir(), "Links/devices are not allowed")
            require(member.name not in names, "Duplicate archive member")
            names.add(member.name)

        def read(relative: str) -> bytes:
            member = tar.getmember(f"{name}/{relative}")
            require(member.isfile(), f"Expected regular file: {relative}")
            stream = tar.extractfile(member)
            require(stream is not None, f"Unreadable member: {relative}")
            return stream.read()

        manifest_bytes = read("BUILD_INFO.json")
        require(sha256(manifest_bytes) == os.environ["EXPECTED_MANIFEST_SHA256"],
                "Manifest digest differs from independently reviewed build metadata")
        info = json.loads(manifest_bytes)
        expected = {
            "schema_version": 1,
            "library": "onnxruntime",
            "version": "1.23.2",
            "source_commit": "a83fc4d58cb48eb68890dd689f94f28288cf2278",
            "builder_commit": os.environ["SOURCE_COMMIT"],
            "thread_pool_scope": "global_ort_plus_per_session_xnnpack",
            "xnnpack": True,
            "xnnpack_kernel_smoke_passed": True,
            "cpuinfo_backend": "emscripten",
            "cpuinfo_source_commit": "8a1772a0c5c447df2d18edf33ec4603a8c9c04a6",
            "relaxed_simd": False,
            "fast_math": False,
            "fp_contract": "off",
            "fp32_experiment": True,
            "model_precision_conversion": False,
            "pthreadpool_backend": "pthreads",
            "pthreadpool_execution_smoke_passed": True,
            "xnnpack_source_commit": "fe98e0b93565382648129271381c14d6205255e3",
            "pthreadpool_source_commit": "4e80ca24521aa0fb3a746f9ea9c3eaa20e9afbb0",
            "emscripten_version": "4.0.8",
            "target": "wasm32-unknown-emscripten",
            "simd": True,
            "pthreads": True,
            "signed": False,
            "exception_abi": "wasm",
        }
        for key, value in expected.items():
            require(type(info.get(key)) is type(value) and info[key] == value,
                    f"Unexpected manifest {key}")
        require(info.get("build_flags") == [
            "--build_wasm_static_lib", "--enable_wasm_simd", "--enable_wasm_threads",
            "--use_xnnpack", "--disable_wasm_exception_catching", "--disable_rtti"], "Unexpected build flags")
        require(info.get("required_link_flags") == [
            "-pthread", "-msimd128", "-fwasm-exceptions", "-fno-fast-math", "-ffp-contract=off"], "Unexpected link flags")
        expected_smoke = {
            "passed": True, "runtime": "node", "execution_provider": "XnnpackExecutionProvider",
            "operators": ["Conv", "ConvTranspose"], "dimensions": "1D",
            "profile_verified": True, "numerical_reference_verified": True,
        }
        for key, value in expected_smoke.items():
            require(info.get("smoke_test", {}).get(key) == value, f"Unexpected smoke {key}")
        require(info.get("source_patches") == ["xnnpack-static-dependencies.patch"], "Unexpected source patches")
        require(info.get("smoke_thread_budget") == {
            "ort_global_intra_op": 1, "ort_global_inter_op": 1, "xnnpack_intra_op": 2,
            "ort_worker_pthreads": 0, "xnnpack_worker_pthreads": 1}, "Unexpected thread budget")
        smoke_text = read("SMOKE_TEST.txt").decode()
        require("PASS: pthread create/join; CPU and XNNPACK FP32 Conv1d/ConvTranspose1d values match scalar reference; profile verification required" in smoke_text,
                "Missing numerical/thread smoke evidence")
        profile = json.loads(smoke_text.split("PROFILE_JSON_BEGIN\n", 1)[1].split("\nPROFILE_JSON_END", 1)[0])
        assignments = {}
        for event in profile:
            args = event.get("args", {})
            if event.get("cat") == "Node" and args.get("provider"):
                assignments.setdefault(args["op_name"], set()).add(args["provider"])
        for op in ("Conv", "ConvTranspose"):
            require(assignments.get(op) == {"XnnpackExecutionProvider"}, f"{op} fallback instead of XNNPACK")
        evidence = json.loads(read("XNNPACK_SMOKE.json"))
        require(evidence.get("passed") is True and evidence.get("execution_provider") == "XnnpackExecutionProvider"
                and evidence.get("profile") == profile, "Inconsistent XNNPACK profile evidence")
        counts = {"ort_fallback_workers": 0, "xnnpack_workers": 1, "xnnpack_intra_op_threads": 2,
                  "ort_global_intra_op_threads": 1, "ort_global_inter_op_threads": 1}
        require(evidence.get("thread_counts") == counts, "Unexpected actual worker counts")
        counts_line = next(line[len("THREAD_COUNTS "):] for line in smoke_text.splitlines() if line.startswith("THREAD_COUNTS "))
        require(json.loads(counts_line) == counts, "Inconsistent worker-count evidence")
        require(read("PTHREADPOOL_SMOKE.txt").decode().strip() ==
                "PASS: pinned pthreadpool threads=2, one worker pthread, caller+worker executed tasks, all results verified",
                "Missing real pthreadpool dispatch proof")
        require(read("XNNPACK_KERNEL_SMOKE.txt").decode().strip() ==
                "PASS: pinned XNNPACK ordinary-SIMD strict-FP32 1D-shaped convolution with two-thread pool",
                "Missing actual XNNPACK kernel execution proof")
        audit = json.loads(read("COMPILE_AUDIT.json"))
        require(audit.get("passed") is True and audit.get("pthreadpool_backend") == "pthreads", "Compile audit failed")
        require(set(audit.get("required_flags", [])) == {"-pthread", "-msimd128", "-fno-fast-math", "-ffp-contract=off", "-fwasm-exceptions"}, "Wrong audited flags")
        require({"pthreads.c", "portable-api.c", "memory.c"} <= set(audit.get("pthreadpool_sources", []))
                and "shim.c" not in audit.get("pthreadpool_sources", []), "Serial pthreadpool backend")
        require(all(type(audit.get("source_counts", {}).get(key)) is int and audit["source_counts"][key] > 0
                    for key in ("total", "xnnpack", "pthreadpool", "wasmsimd", "cpuinfo_emscripten")), "Missing audited source groups")
        require(read("GIT_COMMIT_ID").decode().strip() == expected["source_commit"],
                "Source commit file disagrees with manifest")
        require(read("VERSION_NUMBER").decode().strip() == "1.23.2", "Wrong source version")
        library_hash = sha256(read("lib/libonnxruntime_webassembly.a"))
        require(library_hash == os.environ["EXPECTED_LIBRARY_SHA256"],
                "Static library differs from independently verified library")
        checksums = {}
        for line in read("SHA256SUMS").decode().splitlines():
            digest, file = line.split()
            require(file not in checksums, "Duplicate checksum entry")
            checksums[file] = digest
        require(set(checksums) == {"lib/libonnxruntime_webassembly.a", "xnnpack-static-dependencies.patch",
                                  "COMPILE_AUDIT.json", "XNNPACK_SMOKE.json", "PTHREADPOOL_SMOKE.txt", "XNNPACK_KERNEL_SMOKE.txt"}, "Unexpected checksum files")
        for file, digest in checksums.items():
            require(sha256(read(file)) == digest, f"Invalid checksum: {file}")
        require(read("xnnpack-static-dependencies.patch") == Path(__file__).with_name("xnnpack-static-dependencies.patch").read_bytes(),
                "Archive source patch differs from reviewed builder tree")
        require(f"{name}/include/onnxruntime/core/session/onnxruntime_c_api.h" in names,
                "Missing C API header")
        for required in ["LICENSE", "ThirdPartyNotices.txt", "WASM_THREADED.md"]:
            require(bool(read(required)), f"Missing or empty {required}")
    print("PASS: reviewed archive, manifest, library and smoke evidence match all pins")


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
