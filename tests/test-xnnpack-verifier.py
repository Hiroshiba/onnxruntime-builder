#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("verifier", pathlib.Path(__file__).with_name("verify-xnnpack-build.py"))
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = pathlib.Path(self.folder.name) / "input"

    def compile_input(self, extra=""):
        flags = "-pthread -msimd128 -fno-fast-math -ffp-contract=off -fwasm-exceptions " + extra
        sources = ["/googlexnnpack-src/src/f32-gemm/gen/f32-gemm-4x8-minmax-wasmsimd-x86-splat.c",
                   "/pytorch_cpuinfo-src/src/emscripten/init.c"]
        sources += ["/pthreadpool-src/src/" + p for p in ("pthreads.c", "portable-api.c", "memory.c")]
        self.path.write_text(json.dumps([{"file": p, "command": f"emcc {flags} -c {p}"} for p in sources]))

    def smoke_input(self, provider="XnnpackExecutionProvider", workers=1):
        profile = [{"cat": "Node", "args": {"op_name": op, "provider": provider}} for op in ("Conv", "ConvTranspose")]
        counts = {"ort_fallback_workers": 0, "xnnpack_workers": workers, "xnnpack_intra_op_threads": 2,
                  "ort_global_intra_op_threads": 1, "ort_global_inter_op_threads": 1}
        self.path.write_text(f"PROFILE_JSON_BEGIN\n{json.dumps(profile)}\nPROFILE_JSON_END\nTHREAD_COUNTS {json.dumps(counts)}\nPASS: pthread create/join\n")

    def test_accept_strict_compile(self):
        self.compile_input()
        self.assertTrue(verifier.verify_compile_commands(self.path)["passed"])

    def test_reject_relaxed_and_fast_math(self):
        for flag in ("-mrelaxed-simd", "-ffast-math", "-Ofast", "-ffinite-math-only"):
            self.compile_input(flag)
            with self.assertRaises(ValueError):
                verifier.verify_compile_commands(self.path)

    def test_accept_actual_provider_schema(self):
        self.smoke_input()
        self.assertTrue(verifier.verify_smoke(self.path)["passed"])

    def test_reject_cpu_fallback(self):
        self.smoke_input(provider="CPUExecutionProvider")
        with self.assertRaises(ValueError):
            verifier.verify_smoke(self.path)

    def test_reject_extra_worker(self):
        self.smoke_input(workers=2)
        with self.assertRaises(ValueError):
            verifier.verify_smoke(self.path)

    def test_reject_serial_shim(self):
        self.compile_input()
        self.path.write_text(self.path.read_text().replace("pthreads.c", "shim.c"))
        with self.assertRaises(ValueError):
            verifier.verify_compile_commands(self.path)


if __name__ == "__main__":
    unittest.main()
