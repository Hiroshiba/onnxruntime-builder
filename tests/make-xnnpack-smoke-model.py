#!/usr/bin/env python3
"""Generate a tiny fixed-shape FP32 ONNX model using only the Python stdlib.

The wire fields follow onnx/onnx.proto. The final ORT session load checks the
serialized graph; both CPU and XNNPACK results are checked against scalar math.
No downloaded model, quantization or half-precision conversion is involved.
"""
import pathlib
import struct
import sys


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def integer(field, value):
    return varint(field << 3) + varint(value)


def data(field, value):
    if isinstance(value, str):
        value = value.encode()
    return varint((field << 3) | 2) + varint(len(value)) + value


def tensor(name, dims, values):
    return (b"".join(integer(1, d) for d in dims) + integer(2, 1)
            + data(8, name) + data(9, struct.pack("<" + "f" * len(values), *values)))


def value_info(name, dims):
    shape = b"".join(data(1, integer(1, d)) for d in dims)
    return data(1, name) + data(2, data(1, integer(1, 1) + data(2, shape)))


def ints_attribute(name, values):
    return data(1, name) + b"".join(integer(8, v) for v in values) + integer(20, 7)


def node(name, op, inputs, output):
    return (b"".join(data(1, i) for i in inputs) + data(2, output)
            + data(3, name) + data(4, op)
            + data(5, ints_attribute("pads", [1, 1]))
            + data(5, ints_attribute("kernel_shape", [3])))


def make_model():
    graph = data(1, node("smoke_conv1d", "Conv", ["X", "CW", "CB"], "H"))
    graph += data(1, node("smoke_convtranspose1d", "ConvTranspose", ["H", "TW", "TB"], "Y"))
    graph += data(2, "fixed_fp32_conv1d_convtranspose1d")
    graph += data(5, tensor("CW", [8, 4, 3], [(i % 7 - 3) / 16 for i in range(96)]))
    graph += data(5, tensor("CB", [8], [(i - 3) / 32 for i in range(8)]))
    graph += data(5, tensor("TW", [8, 4, 3], [(i % 5 - 2) / 16 for i in range(96)]))
    graph += data(5, tensor("TB", [4], [(i - 1) / 32 for i in range(4)]))
    graph += data(11, value_info("X", [1, 4, 16]))
    graph += data(12, value_info("Y", [1, 4, 16]))
    graph += data(13, value_info("H", [1, 8, 16]))
    return integer(1, 8) + data(2, "xnnpack-wasm-smoke") + data(7, graph) + data(8, integer(2, 13))


if __name__ == "__main__":
    pathlib.Path(sys.argv[1]).write_bytes(make_model())
