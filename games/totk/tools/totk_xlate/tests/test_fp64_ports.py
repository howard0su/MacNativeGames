"""FP32-fallback conversion regressions using original Maxwell shaders, not game data.

Usage: test_fp64_ports.py totk_yuzu_probe with_msl [totk_msl_compile_probe]
Arithmetic deliberately remains FP32; the packed conversions use nearest-even.
"""
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

from synthetic import EXIT, graphics, mov32

PROBE = Path(sys.argv.pop(1)).resolve()
WITH_MSL = sys.argv.pop(1) == '1'
METAL = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else None


def narrow(bits):
    """Use the host IEEE conversion, independently of the shader's integer IR."""
    sign = (bits >> 32) & 0x80000000
    if bits & 0x7ff0000000000000 == 0x7ff0000000000000:
        fraction = bits & 0xfffffffffffff
        return sign | 0x7f800000 | ((fraction >> 29) | 0x400000 if fraction else 0)
    try:
        value = struct.unpack('<d', struct.pack('<Q', bits))[0]
        return struct.unpack('<I', struct.pack('<f', value))[0]
    except OverflowError:
        return sign | 0x7f800000


def widen(bits):
    if bits & 0x7f800000 == 0x7f800000 and bits & 0x7fffff:
        return ((bits & 0x80000000) << 32) | 0x7ff8000000000000 | ((bits & 0x7fffff) << 29)
    value = struct.unpack('<f', struct.pack('<I', bits))[0]
    return struct.unpack('<Q', struct.pack('<d', value))[0]


def f2f(destination, source, *, wide):
    return (0x5ca8000000000000 | (source << 20) | (7 << 16)
            | ((2 if wide else 3) << 10) | ((3 if wide else 2) << 8) | destination)


def local(register, offset, *, store, wide=False):
    return ((0xef50000000000000 if store else 0xef40000000000000)
            | ((5 if wide else 4) << 48) | (offset << 20)
            | (7 << 16) | (255 << 8) | register)


def shader(instructions):
    code = bytearray(graphics([*instructions, EXIT], color=True))
    struct.pack_into('<I', code, 4, 64)  # Graphics local-memory allocation.
    struct.pack_into('<I', code, 0x48, 0xf)  # Export all four raw float words.
    return code


class Fp64Ports(unittest.TestCase):
    def setUp(self):
        # Keep scratch files inside the worktree, including when invoked by CTest.
        directory = tempfile.TemporaryDirectory(prefix='.fp64-test-', dir=Path.cwd())
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def check_shader(self, instructions, expected):
        code, spv, msl = (self.root / name for name in ('input.code', 'out.spv', 'out.metal'))
        code.write_bytes(shader(instructions))
        command = [str(PROBE), 'ps', str(code), str(spv)]
        if WITH_MSL:
            command.append(str(msl))
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        words = struct.unpack('<' + 'I' * (spv.stat().st_size // 4), spv.read_bytes())
        self.assertEqual(words[0], 0x07230203)
        offset = 5
        while offset < len(words):
            size, opcode = words[offset] >> 16, words[offset] & 0xffff
            self.assertGreater(size, 0)
            # No Float64 capability or 64-bit floating type may escape lowering.
            self.assertFalse(opcode == 17 and words[offset + 1] == 10)
            self.assertFalse(opcode == 22 and words[offset + 2] == 64)
            offset += size
        self.assertEqual(offset, len(words))
        if WITH_MSL and METAL:
            result = subprocess.run(
                [str(METAL), str(msl), 'float', ','.join(f'{word:08x}' for word in expected)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)

    def check_narrow(self, cases):
        for start in range(0, len(cases), 4):
            batch = cases[start:start + 4]
            instructions = [mov32(0) | register for register in range(4)]
            for channel, bits in enumerate(batch):
                # Local loads prevent folding the conversion into a host-side constant.
                instructions += [mov32(bits & 0xffffffff) | 8, mov32(bits >> 32) | 9,
                                 local(8, channel * 8, store=True, wide=True),
                                 local(8, channel * 8, store=False, wide=True),
                                 f2f(channel, 8, wide=False)]
            with self.subTest(bits=[f'{value:016x}' for value in batch]):
                self.check_shader(instructions, [narrow(bits) for bits in batch]
                                  + [0] * (4 - len(batch)))

    def check_widen(self, cases):
        for start in range(0, len(cases), 2):
            batch = cases[start:start + 2]
            instructions = [mov32(0) | register for register in range(4)]
            expected = []
            for channel, bits in enumerate(batch):
                instructions += [mov32(bits) | 8, local(8, channel * 4, store=True),
                                 local(8, channel * 4, store=False),
                                 f2f(channel * 2, 8, wide=True)]
                result = widen(bits)
                expected += [result & 0xffffffff, result >> 32]
            with self.subTest(bits=[f'{value:08x}' for value in batch]):
                self.check_shader(instructions, expected + [0] * (4 - len(expected)))

    def test_double_add_high_word_and_memory_round_trip(self):
        # DADD.RN R0,R0,R2; MOV R0,R1 used to export d0080000, not 40080000.
        initialize = [mov32(0), mov32(0x3ff00000) | 1,
                      mov32(0) | 2, mov32(0x40000000) | 3]
        self.check_shader([*initialize, 0x5c70000000270000, 0x5c98078000170000,
                           mov32(0) | 1, mov32(0) | 2, mov32(0) | 3],
                          [0x40080000, 0, 0, 0])
        self.check_shader([*initialize, local(0, 0, store=True, wide=True),
                           local(4, 0, store=False, wide=True),
                           0x5c70000000270000 | (4 << 8),
                           local(0, 8, store=True, wide=True),
                           mov32(0), mov32(0) | 1,
                           local(0, 8, store=False, wide=True),
                           mov32(0) | 2, mov32(0) | 3],
                          [0, 0x40080000, 0, 0])

    def test_narrow_rounding_boundaries_and_special_values(self):
        cases = [
            0, 0x3ff0000000000000, 0x4000000000000000, 0x4008000000000000,
            0x3ff000000fffffff, 0x3ff0000010000000, 0x3ff0000010000001,
            0x3ff0000030000000, 0x3ffffffff0000000,
            0x380fffffdfffffff, 0x380fffffe0000000, 0x380fffffe0000001,
            0x3810000000000000, 0x3800000000000000,
            0x36a0000000000000, 0x3690000000000000, 0x3690000000000001,
            0x36a7ffffffffffff, 0x36a8000000000000, 0x36a8000000000001,
            0x36b3ffffffffffff, 0x36b4000000000000, 0x36b4000000000001,
            0x368fffffffffffff, 1, 0x000fffffffffffff, 0x0010000000000000,
            0x47efffffe0000000, 0x47efffffefffffff, 0x47effffff0000000,
            0x47f0000000000000, 0x7fefffffffffffff, 0x7ff0000000000000,
            0x7ff0000000000001, 0x7ff123456789abcd, 0x7ff8000000000000,
        ]
        self.check_narrow(cases + [bits | (1 << 63) for bits in cases])

    def test_double_condition_flags_lower_immediate_zero_operands(self):
        for a, b, flags in ((0x3ff00000, 0xbff00000, 1),
                            (0xc0000000, 0x3ff00000, 2),
                            (0x3ff00000, 0x40000000, 0)):
            with self.subTest(a=f'{a:08x}', b=f'{b:08x}'):
                self.check_shader([
                    mov32(0), mov32(a) | 1, mov32(0) | 2, mov32(b) | 3,
                    local(0, 0, store=True, wide=True),
                    local(0, 0, store=False, wide=True),
                    0x5c70000000270004 | (1 << 47),  # DADD.RN.CC R4,R0,R2.
                    mov32(0x3f800000) | 8,
                    # P2R.CC R0,R8,3 keeps both zero/sign comparisons live.
                    0x38e8000000000000 | (1 << 40) | (3 << 20) | (7 << 16) | (8 << 8),
                    mov32(0) | 1, mov32(0) | 2, mov32(0) | 3,
                ], [0x3f800000 | flags, 0, 0, 0])

    def test_widen_normal_subnormal_signed_zero_and_nan(self):
        cases = [0, 1, 2, 3, 0x00400000, 0x007fffff, 0x00800000,
                 0x3f800000, 0x3f800001, 0x40400000, 0x7f7fffff,
                 0x7f800000, 0x7f800001, 0x7fa12345, 0x7fc00000]
        self.check_widen(cases + [bits | (1 << 31) for bits in cases])

    def test_double_minmax_selection_and_live_condition_flags(self):
        for maximum in (False, True):
            for a, b, minimum, maximum_value in (
                (0, 0, 0, 0),
                (0xbff00000, 0x3ff00000, 0xbff00000, 0x3ff00000),
                (0x3ff00000, 0x40000000, 0x3ff00000, 0x40000000),
                (0xc0000000, 0xbff00000, 0xc0000000, 0xbff00000),
            ):
                selected = maximum_value if maximum else minimum
                flags = 1 if selected == 0 else 2 if selected & 0x80000000 else 0
                with self.subTest(maximum=maximum, a=f'{a:08x}', b=f'{b:08x}'):
                    self.check_shader([
                        mov32(0), mov32(a) | 1, mov32(0) | 2, mov32(b) | 3,
                        local(0, 0, store=True, wide=True),
                        local(0, 0, store=False, wide=True),
                        # DMNMX.CC R4,R0,R2,PT (negated PT selects maximum).
                        0x5c50000000270004 | (7 << 39) | (1 << 47) | (maximum << 42),
                        mov32(0x3f800000) | 8,
                        0x38e8000000000000 | (1 << 40) | (3 << 20) | (7 << 16) | (8 << 8),
                        0x5c98078000570001,  # MOV R1,R5 exports the selected high word.
                        mov32(0) | 2, mov32(0) | 3,
                    ], [0x3f800000 | flags, selected, 0, 0])

    def test_deterministic_conversion_samples(self):
        randoms = random.Random(0x64_32)
        # Concentrate on the representable and rounding-sensitive FP32 range.
        self.check_narrow([(randoms.randrange(868, 1155) << 52)
                           | randoms.getrandbits(52) | (randoms.getrandbits(1) << 63)
                           for _ in range(16)])
        self.check_widen([randoms.getrandbits(32) for _ in range(8)])


if __name__ == '__main__':
    unittest.main()
