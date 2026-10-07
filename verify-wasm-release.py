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
            "builder_commit": os.environ["SOURCE_BUILD_COMMIT"],
            "validation_commit": os.environ["SOURCE_COMMIT"],
            "build_run_id": int(os.environ["SOURCE_BUILD_RUN_ID"]),
            "validation_run_id": int(os.environ["SOURCE_RUN_ID"]),
            "thread_pool_scope": "global",
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
            "--disable_wasm_exception_catching", "--disable_rtti"], "Unexpected build flags")
        require(info.get("required_link_flags") == [
            "-pthread", "-msimd128", "-fwasm-exceptions"], "Unexpected link flags")
        smoke = info.get("smoke_test", {})
        require(smoke.get("passed") is True and smoke.get("runtime") == "node"
                and type(smoke.get("intra_op_threads")) is int
                and smoke["intra_op_threads"] == 2
                and type(smoke.get("inter_op_threads")) is int
                and smoke["inter_op_threads"] == 1, "Expected successful global two-thread Node smoke")
        require(read("SMOKE_TEST.txt").decode().strip() ==
                "PASS: pthread create/join; ORT session intra_op_threads=2; inference values verified",
                "Unexpected smoke-test evidence")
        require(read("GIT_COMMIT_ID").decode().strip() == expected["source_commit"],
                "Source commit file disagrees with manifest")
        require(read("VERSION_NUMBER").decode().strip() == "1.23.2", "Wrong source version")
        library_hash = sha256(read("lib/libonnxruntime_webassembly.a"))
        require(library_hash == os.environ["EXPECTED_LIBRARY_SHA256"],
                "Static library differs from independently verified library")
        require(read("SHA256SUMS").decode().split() ==
                [library_hash, "lib/libonnxruntime_webassembly.a"], "Invalid library checksum")
        require(f"{name}/include/onnxruntime/core/session/onnxruntime_c_api.h" in names,
                "Missing C API header")
        for required in ["LICENSE", "ThirdPartyNotices.txt", "WASM_THREADED.md"]:
            require(bool(read(required)), f"Missing or empty {required}")
    print("PASS: reviewed archive, manifest, library and smoke evidence match all pins")


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
