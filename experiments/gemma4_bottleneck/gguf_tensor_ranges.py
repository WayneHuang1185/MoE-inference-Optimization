#!/usr/bin/env python3
"""Dump GGUF tensor file offset ranges without external dependencies."""

from __future__ import annotations

import argparse
import csv
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path


GGUF_MAGIC = 0x46554747
GGUF_DEFAULT_ALIGNMENT = 32

VALUE_SIZES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}

QUANT_SIZES = {
    0: (1, 4),       # F32
    1: (1, 2),       # F16
    2: (32, 18),     # Q4_0
    3: (32, 20),     # Q4_1
    6: (32, 22),     # Q5_0
    7: (32, 24),     # Q5_1
    8: (32, 34),     # Q8_0
    9: (32, 40),     # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    19: (256, 50),   # IQ1_S
    20: (32, 18),    # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),   # IQ2_S
    23: (256, 148),  # IQ4_XS
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    29: (256, 56),   # IQ1_M
    30: (1, 2),      # BF16
    34: (256, 54),   # TQ1_0
    35: (256, 66),   # TQ2_0
    39: (32, 17),    # MXFP4
    40: (64, 36),    # NVFP4
}

QUANT_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
}


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    tensor_type: int
    relative_offset: int


class Reader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.f = path.open("rb")
        self.alignment = GGUF_DEFAULT_ALIGNMENT

    def read(self, fmt: str):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise EOFError("unexpected EOF")
        values = struct.unpack("<" + fmt, data)
        return values[0] if len(values) == 1 else values

    def read_string(self) -> str:
        n = self.read("Q")
        data = self.f.read(n)
        if len(data) != n:
            raise EOFError("unexpected EOF in string")
        return data.decode("utf-8", errors="replace")

    def read_value(self, value_type: int):
        if value_type in VALUE_SIZES:
            fmts = {
                0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                7: "?", 10: "Q", 11: "q", 12: "d",
            }
            return self.read(fmts[value_type])
        if value_type == 8:  # STRING
            return self.read_string()
        if value_type == 9:  # ARRAY
            elem_type = self.read("I")
            count = self.read("Q")
            if elem_type in VALUE_SIZES:
                self.f.seek(VALUE_SIZES[elem_type] * count, 1)
                return None
            if elem_type == 8:
                for _ in range(count):
                    self.read_string()
                return None
            raise ValueError(f"unsupported array value type: {elem_type}")
        raise ValueError(f"unsupported value type: {value_type}")

    def read_tensor_info(self) -> TensorInfo:
        name = self.read_string()
        n_dims = self.read("I")
        dims = [self.read("Q") for _ in range(n_dims)]
        tensor_type = self.read("I")
        relative_offset = self.read("Q")
        return TensorInfo(name, dims, tensor_type, relative_offset)

    def parse(self) -> tuple[int, list[TensorInfo]]:
        magic = self.read("I")
        if magic != GGUF_MAGIC:
            raise ValueError(f"bad GGUF magic: {magic:#x}")
        version = self.read("I")
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version: {version}")
        tensor_count = self.read("Q")
        kv_count = self.read("Q")

        for _ in range(kv_count):
            key = self.read_string()
            value_type = self.read("I")
            value = self.read_value(value_type)
            if key == "general.alignment" and value_type == 4:
                self.alignment = int(value)

        tensors = [self.read_tensor_info() for _ in range(tensor_count)]
        data_offset = align(self.f.tell(), self.alignment)
        return data_offset, tensors


def align(value: int, alignment: int) -> int:
    rem = value % alignment
    return value if rem == 0 else value + alignment - rem


def tensor_nbytes(info: TensorInfo) -> int:
    n_elements = math.prod(info.dims)
    block_size, type_size = QUANT_SIZES[info.tensor_type]
    return math.ceil(n_elements / block_size) * type_size


def classify_tensor(name: str) -> tuple[str, str]:
    layer_match = re.match(r"blk\.(\d+)\.", name)
    layer = layer_match.group(1) if layer_match else ""
    if "ffn" in name or "expert" in name or "_exps" in name:
        family = "moe_ffn"
    elif "attn" in name:
        family = "attention"
    elif "token_embd" in name:
        family = "embedding"
    elif "output" in name:
        family = "output"
    else:
        family = "other"
    return layer, family


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reader = Reader(Path(args.model))
    data_offset, tensors = reader.parse()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name", "layer", "family", "type", "n_dims", "dims",
                "n_elements", "n_bytes", "relative_offset",
                "file_start", "file_end",
            ],
        )
        writer.writeheader()
        for info in tensors:
            n_bytes = tensor_nbytes(info)
            file_start = data_offset + info.relative_offset
            layer, family = classify_tensor(info.name)
            writer.writerow({
                "name": info.name,
                "layer": layer,
                "family": family,
                "type": QUANT_NAMES.get(info.tensor_type, str(info.tensor_type)),
                "n_dims": len(info.dims),
                "dims": "x".join(str(d) for d in info.dims),
                "n_elements": math.prod(info.dims),
                "n_bytes": n_bytes,
                "relative_offset": info.relative_offset,
                "file_start": file_start,
                "file_end": file_start + n_bytes,
            })
    print(f"wrote {args.output} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
