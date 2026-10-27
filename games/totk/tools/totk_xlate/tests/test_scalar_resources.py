"""Validate the scalar subset of the existing resource correctness fixtures."""
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


def run(arguments):
    return subprocess.run([str(arg) for arg in arguments], capture_output=True,
                          text=True, timeout=60)


def instructions(path):
    data = path.read_bytes()
    words = struct.unpack(f"<{len(data) // 4}I", data)
    assert words[0] == 0x07230203, path
    cursor = 5
    while cursor < len(words):
        count, opcode = words[cursor] >> 16, words[cursor] & 0xffff
        assert count > 0 and cursor + count <= len(words), path
        yield opcode, words[cursor + 1:cursor + count]
        cursor += count


def main():
    fixture, cross = map(Path, sys.argv[1:3])
    validator = shutil.which("spirv-val")
    assert validator, "spirv-val is required"
    with tempfile.TemporaryDirectory(prefix="scalar-resources-", dir=fixture.parent) as directory:
        root = Path(directory)
        generated = run([fixture, root])
        assert generated.returncode == 0, generated.stderr
        modules = sorted(root.glob("*.spv"))
        assert len(modules) == 138, len(modules)
        required_capabilities = {
            "sampled_1d": 43,
            "sampled_array1d": 43,
            "sampled_arraycube": 45,
            "sampled_msarray": 48,
        }
        for path in modules:
            validated = run([validator, "--target-env", "vulkan1.1", path])
            assert validated.returncode == 0, f"{path.name}: {validated.stderr}"
            module = list(instructions(path))
            if path.stem in required_capabilities:
                assert any(op == 17 and args == (required_capabilities[path.stem],)
                           for op, args in module), path
            if path.stem == "typeless_fallback":
                assert not any(op == 98 for op, _ in module), path
                assert any(op == 46 for op, _ in module), path
            if path.stem.startswith(("image_s8", "image_s16", "image_s32")):
                signed = {args[0] for op, args in module if op == 21 and args[1:] == (32, 1)}
                vectors = {args[0] for op, args in module
                           if op == 23 and args[1] in signed and args[2] == 4}
                assert any(op == 25 and args[1] in signed for op, args in module), path
                assert any(op == 98 and args[0] in vectors for op, args in module), path
            if path.stem.startswith("query_samples_"):
                multisample = not path.stem.startswith("query_samples_1")
                assert any(op == 107 for op, _ in module) == multisample, path
            if path.stem.startswith(("surface_atomic17_", "surface_atomic18_")):
                assert any(op == 230 for op, _ in module), path
            translated = run([cross, path, "--msl", "--msl-version", "30100",
                              "--msl-texture-buffer-native", "--output",
                              path.with_suffix(".metal")])
            assert translated.returncode == 0, f"{path.name}: {translated.stderr}"
        print(f"Validated {len(modules)} scalar resource modules and explicit rejection cases")


if __name__ == "__main__":
    main()
