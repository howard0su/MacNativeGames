"""Original Maxwell instruction fixtures with independent integer reference results."""
import json
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from synthetic import EXIT, graphics, mov32

SERVICE = Path(sys.argv.pop(1)).resolve()
CROSS = Path(sys.argv.pop(1)).resolve()
METAL = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else None
MASK = 0xffffffff
PT = 7 << 16


def signed(value, width=32):
    value &= (1 << width) - 1
    return value - (1 << width) if value >> (width - 1) else value


def cset(condition, destination):
    return 0x5098000000000000 | (7 << 39) | PT | (condition << 8) | destination


class InstructionPorts(unittest.TestCase):
    def setUp(self):
        # Keep generated artifacts within the fresh worktree build, never in a system temp dir.
        root = tempfile.TemporaryDirectory(prefix='instruction-ports-', dir=SERVICE.parent)
        self.addCleanup(root.cleanup)
        self.root = Path(root.name)
        self.code = self.root / 'fixture.code'
        self.output = self.root / 'output'
        self.sequence = 0

    def compile(self, instructions, *, c1=(), error=None, **state):
        self.sequence += 1
        name = f'ports-{self.sequence}'
        code = graphics([*instructions, EXIT], color=True)
        struct.pack_into('<I', code, 0, (5 << 10) | 2)
        struct.pack_into('<I', code, 0x48, 15)
        self.code.write_bytes(code)
        request = dict(variant=name, stage='ps', code=str(self.code),
                       fout={'0': 'uint'}, **state)
        if c1:
            request['c1'] = struct.pack('<' + 'I' * len(c1), *c1).hex()
        result = subprocess.run(
            [str(SERVICE), '--service', '--spirv-cross', str(CROSS),
             '--out', str(self.output), '--chain', 'totk-yuzu-test-v1'],
            input=json.dumps(request) + '\n', capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        reply = json.loads(result.stdout)
        if error:
            self.assertFalse(reply['ok'], reply)
            self.assertIn(error, reply['error'])
            return
        self.assertTrue(reply['ok'], reply)
        validator = shutil.which('spirv-val')
        if validator:
            validation = subprocess.run(
                [validator, '--target-env', 'vulkan1.1', str(self.output / f'{name}.spv')],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(validation.returncode, 0, validation.stderr)
        return name

    def check(self, instructions, expected, **state):
        name = self.compile(instructions, **state)
        if METAL:
            metadata = self.output / f'{name}_b.json'
            arguments = [str(metadata)] if json.loads(metadata.read_text())['bindings'] else []
            result = subprocess.run(
                [str(METAL), str(self.output / f'{name}_b.metal'), 'uint',
                 ','.join(f'{value & MASK:08x}' for value in expected),
                 *arguments],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_iadd_extended_flags_and_cumulative_zero(self):
        # Seed all flags with an ordinary add, then consume the extended result's flags.
        cases = [(MASK, 1, MASK, 0), (0, 0, 0, 0), (1, 0, 0, 0),
                 (MASK, 1, 0x7fffffff, 0), (0, 0, 0x80000000, MASK),
                 (MASK, 1, MASK, MASK)]
        for a, b, c, d in cases:
            with self.subTest(values=(a, b, c, d)):
                first = (a + b) & MASK
                carry = (a + b) >> 32
                total = c + d + carry
                value = total & MASK
                overflow = bool(((value ^ c) & ~(c ^ d)) & 0x80000000)
                flags = [first == 0 and value == 0, total > MASK, overflow, value >> 31]
                instructions = [mov32(a) | 4, mov32(b) | 5, mov32(c) | 6, mov32(d) | 7,
                                0x5c10000000000000 | (1 << 47) | (5 << 20) | PT | (4 << 8) | 8,
                                0x5c10000000000000 | (1 << 47) | (1 << 43) |
                                (7 << 20) | PT | (6 << 8) | 8]
                self.check(instructions + [cset(cc, dest) for dest, cc in enumerate((2, 22, 23, 21))],
                           [MASK if flag else 0 for flag in flags])

    def test_imul_high_signed_unsigned_mixed(self):
        rng = random.Random(0x1a4d)
        for a, b in [(MASK, MASK), (0x80000000, MASK), (0xffff0001, 0x1234ffff),
                     *[(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(5)]]:
            instructions = [mov32(a) | 4, mov32(b) | 5]
            expected = []
            for dest, (sa, sb) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                instructions.append(0x5c38000000000000 | (sa << 40) | (sb << 41) |
                                    (1 << 39) | (5 << 20) | PT | (4 << 8) | dest)
                expected.append(((signed(a) if sa else a) * (signed(b) if sb else b)) >> 32)
            self.check(instructions, expected)

    def test_iadd32i_extended_carry_and_zero(self):
        self.check([mov32(MASK) | 4,
                    0x1c00000000000000 | (1 << 52) | (1 << 20) | PT | (4 << 8) | 8,
                    0x1c00000000000000 | (1 << 52) | (1 << 53) | PT | (4 << 8),
                    cset(22, 1), cset(2, 2), cset(23, 3)],
                   [0, MASK, MASK, 0])

    def test_runtime_operands_reach_integer_backend(self):
        # The unchanged Metal harness supplies these user-cbuf values at run time.
        ldc = 0xef90000000000000 | (4 << 48) | (3 << 36) | PT | (255 << 8)
        a, b = 0x00ff0001, 0xff000100
        instructions = [ldc | (16 << 20) | 4, ldc | (20 << 20) | 5]
        expected = []
        for dest, (sa, sb) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            expected.append(((signed(a) if sa else a) * (signed(b) if sb else b)) >> 32)
            instructions.append(0x5c38000000000000 | (sa << 40) | (sb << 41) |
                                (1 << 39) | (5 << 20) | PT | (4 << 8) | dest)
        self.check(instructions, expected)
        # A runtime carry-in changes the signed-overflow outcome at INT_MAX.
        self.check([ldc | 4, mov32(MASK) | 5, mov32(0x7fffffff) | 6,
                    0x5c10000000000000 | (1 << 47) | (5 << 20) | PT | (4 << 8) | 8,
                    0x5c10000000000000 | (1 << 47) | (1 << 43) |
                    (255 << 20) | PT | (6 << 8) | 8,
                    *[cset(cc, dest) for dest, cc in enumerate((2, 22, 23, 21))]],
                   [0, 0, MASK, MASK])
        expected = []
        instructions = [ldc | (16 << 20) | 4, ldc | (20 << 20) | 5, mov32(MASK) | 6]
        for dest, (scale, sat) in enumerate(((0, 0), (1, 0), (2, 0), (1, 1))):
            value = (a * signed(b) - 1) >> (0, 7, 15)[scale]
            if sat:
                value = min(max(value, -0x80000000), 0x7fffffff)
            expected.append(value)
            instructions.append(0x5f00000000000000 | (3 << 37) | (3 << 29) |
                                (1 << 50) | (1 << 49) | (scale << 51) | (sat << 55) |
                                (6 << 39) | (5 << 20) | PT | (4 << 8) | dest)
        self.check(instructions, expected)

    def test_imul_imad_operand_forms_and_accumulator(self):
        # Register, cbuf, sign-extended immediate and full immediate must agree.
        a, b, c = 0x81234567, 0xfffffff9, 0x11223344
        product = (signed(a) * signed(b)) >> 32
        self.check([mov32(a) | 4, mov32(b) | 5,
                    0x5c38000000000000 | (3 << 40) | (1 << 39) | (5 << 20) | PT | (4 << 8),
                    0x4c38000000000000 | (3 << 40) | (1 << 39) | (1 << 34) | PT | (4 << 8) | 1,
                    0x3838000000000000 | (1 << 56) | (3 << 40) | (1 << 39) |
                    (0x7fff9 << 20) | PT | (4 << 8) | 2,
                    0x1f00000000000000 | (3 << 54) | (1 << 53) |
                    (b << 20) | PT | (4 << 8) | 3],
                   [product] * 4, c1=[b])
        expected = (a * b + c) & MASK
        self.check([mov32(a) | 4, mov32(b) | 5, mov32(c) | 6, mov32(c) | 3,
                    0x5a00000000000000 | (6 << 39) | (5 << 20) | PT | (4 << 8),
                    0x4a00000000000000 | (6 << 39) | (1 << 34) | PT | (4 << 8) | 1,
                    0x5200000000000000 | (5 << 39) | (1 << 34) | (1 << 20) | PT | (4 << 8) | 2,
                    0x8000000000000000 | (b << 20) | PT | (4 << 8) | 3],
                   [expected] * 4, c1=[b, c])

    def test_imad_high_products_and_negation(self):
        a, b, c = 0x80000001, 7, 0x12345678
        instructions = [mov32(a) | 4, mov32(b) | 5, mov32(c) | 6]
        expected = []
        for dest, (sa, sb, mode) in enumerate(((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 2))):
            operand_b = (-b if mode == 2 else b) & MASK
            product = (signed(a) if sa else a) * (signed(operand_b) if sb else operand_b)
            expected.append((product >> 32) + (-c if mode == 1 else c))
            instructions.append(0x5a00000000000000 | (1 << 54) | (sa << 48) |
                                (sb << 53) | (mode << 51) | (6 << 39) |
                                (5 << 20) | PT | (4 << 8) | dest)
        self.check(instructions, expected)
        self.check([mov32(a) | 4, mov32(c) | 6, mov32(c) | 1,
                    0x3400000000000000 | (1 << 54) | (1 << 48) | (1 << 53) |
                    (6 << 39) | (b << 20) | PT | (4 << 8),
                    0x8000000000000000 | (1 << 54) | (1 << 57) | (1 << 53) |
                    (b << 20) | PT | (4 << 8) | 1,
                    0x3838000000000000 | (b << 20) | PT | (4 << 8) | 2,
                    0x1f00000000000000 | (b << 20) | PT | (4 << 8) | 3],
                   [(signed(a) * b >> 32) + c] * 2 + [a * b] * 2)

    def test_xmad_csfu_and_add_flags(self):
        a, b, c = 0xffff, 0xffff, 0x20001
        instructions = [mov32(a) | 4, mov32(b) | 5, mov32(c) | 6]
        expected = []
        for dest, (sa, sb) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            aa, bb = signed(a, 16) if sa else a, signed(b, 16) if sb else b
            expected.append(aa * bb + c - ((aa < 0) + (bb < 0)) * 0x10000)
            instructions.append(0x5b00000000000000 | (3 << 50) | (sa << 48) |
                                (sb << 49) | (6 << 39) | (5 << 20) | PT | (4 << 8) | dest)
        self.check(instructions, expected)
        # The product and C produce an unsigned carry and a zero result.
        self.check([mov32(0xffff) | 4, mov32(0xffff) | 5, mov32(0x1ffff) | 6,
                    0x5b00000000000000 | (1 << 47) | (6 << 39) | (5 << 20) | PT | (4 << 8) | 8,
                    *[cset(cc, dest) for dest, cc in enumerate((2, 22, 23, 21))]],
                   [MASK, MASK, 0, 0])
        self.check([mov32(0x8000) | 4, mov32(0x8000) | 5, mov32(0x40000000) | 6,
                    0x5b00000000000000 | (1 << 47) | (3 << 48) |
                    (6 << 39) | (5 << 20) | PT | (4 << 8) | 8,
                    *[cset(cc, dest) for dest, cc in enumerate((2, 22, 23, 21))]],
                   [0, 0, MASK, MASK])

    def test_vmad_scale_negation_plus_one_and_saturation(self):
        # Reference uses arbitrary-precision mathematics, not donor lowering.
        cases = [(MASK, MASK, 0, 0, 0, 2), (MASK, MASK, 0, 0, 0, 1),
                 (MASK, MASK, 0, 0, 0, 0), (0, 0, 0x80000000, 1, 0, 0),
                 (0xffffffff, 2, MASK, 1, 0, 0), (0x80000000, 2, MASK, 1, 1, 2),
                 (0x10000, 0x10001, MASK, 0, 0, 3), (0x7fffffff, 2, 0, 1, 1, 0)]
        for a, b, c, sa, sb, mode in cases:
            with self.subTest(values=(a, b, c, sa, sb, mode)):
                instructions = [mov32(a) | 4, mov32(b) | 5, mov32(c) | 6]
                expected = []
                for dest, (scale, sat) in enumerate(((0, 0), (1, 0), (2, 0), (1, 1))):
                    aa, bb = signed(a) if sa else a, signed(b) if sb else b
                    result_signed = bool(sa or sb or mode in (1, 2))
                    cc = signed(c) if result_signed else c
                    value = (-aa * bb if mode == 2 else aa * bb) + (-cc if mode == 1 else cc)
                    if mode == 3:
                        value += 1
                    value >>= (0, 7, 15)[scale]
                    if sat:
                        low, high = (-0x80000000, 0x7fffffff) if result_signed else (0, MASK)
                        value = min(max(value, low), high)
                    expected.append(value)
                    instructions.append(0x5f00000000000000 | (3 << 37) | (3 << 29) |
                                        (1 << 50) | (sa << 48) | (sb << 49) |
                                        (mode << 53) | (scale << 51) | (sat << 55) |
                                        (6 << 39) | (5 << 20) | PT | (4 << 8) | dest)
                self.check(instructions, expected)

    def test_video_halfword_selection_and_signed_immediate(self):
        a, b, c = 0x8001ffff, 0xffff0003, MASK
        instructions = [mov32(a) | 4, mov32(b) | 5, mov32(c) | 6]
        # VMAD signed high halfword times an immediate -2 and an unsigned low halfword.
        instructions += [
            0x5f00000000000000 | (5 << 36) | (1 << 48) | (1 << 49) |
            (0xfffe << 20) | (6 << 39) | PT | (4 << 8),
            0x5f00000000000000 | (5 << 36) | (4 << 28) | (1 << 48) |
            (1 << 50) | (5 << 20) | (6 << 39) | PT | (4 << 8) | 1,
            0x5b00000000000000 | (1 << 53) | (1 << 48) | (1 << 35) |
            (6 << 39) | (5 << 20) | PT | (4 << 8) | 2,
            0x3600000000000000 | (1 << 53) | (1 << 48) |
            (3 << 20) | (6 << 39) | PT | (4 << 8) | 3]
        ah = signed(a >> 16, 16)
        self.check(instructions, [ah * -2 - 1, ah * 3 - 1,
                                  ah * (b >> 16) + c, ah * 3 + c])

    def test_vmnmx_saturation_respects_merge_width_and_preserves_other_bits(self):
        c = 0xa5b6c7d8
        for a, b, source_signed, dest_signed in [
                (0x12345, 0x23456, False, False),
                (0x12345, 0x23456, False, True),
                (0x80000000, 7, True, True),
                (0x80000000, 7, True, False)]:
            with self.subTest(values=(a, b, source_signed, dest_signed)):
                instructions = [mov32(a) | 4, mov32(b) | 5, mov32(c) | 6]
                value = min(signed(a), signed(b)) if source_signed else min(a, b)
                expected = []
                for dest, (width, shift) in enumerate(((16, 16), (16, 0), (8, 0), (8, 16))):
                    lo, hi = (-(1 << (width - 1)), (1 << (width - 1)) - 1) if dest_signed \
                        else (0, (1 << width) - 1)
                    part = min(max(value, lo), hi) & ((1 << width) - 1)
                    expected.append((c & ~(((1 << width) - 1) << shift)) | (part << shift))
                    instructions.append(
                        0x3a00000000000000 | (1 << 55) | (dest_signed << 54) |
                        (dest << 51) | (1 << 50) | (source_signed << 48) |
                        (source_signed << 49) | (3 << 37) | (3 << 29) |
                        (6 << 39) | (5 << 20) | PT | (4 << 8) | dest)
                self.check(instructions, expected)

    def test_vmnmx_mixed_sign_comparisons_in_both_operand_orders(self):
        a, b = 0xfffffff0, 0x80000000
        instructions = [mov32(a) | 4, mov32(b) | 5, mov32(0) | 6]
        for dest, (reg_a, reg_b, sa, sb, maximum) in enumerate(
                ((4, 5, 1, 0, 0), (4, 5, 1, 0, 1),
                 (5, 4, 0, 1, 0), (5, 4, 0, 1, 1))):
            instructions.append(
                0x3a00000000000000 | (maximum << 56) | (4 << 51) | (1 << 50) |
                (sa << 48) | (sb << 49) | (3 << 37) | (3 << 29) |
                (6 << 39) | (reg_b << 20) | PT | (reg_a << 8) | dest)
        self.check(instructions, [a, b, a, b])

    def test_vmnmx_saturates_before_secondary_operation(self):
        instructions = [mov32(MASK) | 4, mov32(MASK - 1) | 5, mov32(1) | 6, mov32(MASK) | 7]
        for dest, (op, reg_c) in enumerate(((4, 6), (5, 7), (6, 7), (4, 7))):
            instructions.append(
                0x3a00000000000000 | (1 << 55) | (1 << 54) | (op << 51) | (1 << 50) |
                (3 << 37) | (3 << 29) | (reg_c << 39) | (5 << 20) | PT | (4 << 8) | dest)
        self.check(instructions, [0x80000000, MASK, 0x7fffffff, 0x7ffffffe])

    def test_vmnmx_secondary_compare_preserves_intermediate_signedness(self):
        instructions = [mov32(MASK) | 4, mov32(1) | 6, mov32(0xfffffff0) | 5]
        for dest, (reg_a, source_signed, dest_signed, op) in enumerate(
                ((4, 0, 1, 5), (4, 0, 1, 6), (5, 1, 0, 5), (5, 1, 0, 6))):
            instructions.append(
                0x3a00000000000000 | (dest_signed << 54) | (op << 51) | (1 << 50) |
                (source_signed << 48) | (source_signed << 49) | (3 << 37) | (3 << 29) |
                (6 << 39) | (reg_a << 20) | PT | (reg_a << 8) | dest)
        self.check(instructions, [1, MASK, 0xfffffff0, 1])

    def test_vmnmx_signed_halfword_and_immediate_forms(self):
        instructions = [mov32(0x8001ffff) | 4, mov32(1) | 6]
        for dest, (op, maximum, sat) in enumerate(((4, 0, 0), (4, 1, 0), (0, 0, 1), (2, 0, 1))):
            instructions.append(
                0x3a00000000000000 | (maximum << 56) | (sat << 55) | (1 << 54) |
                (op << 51) | (3 << 48) | (5 << 36) | (0xff80 << 20) |
                (6 << 39) | PT | (4 << 8) | dest)
        self.check(instructions, [0xffff8002, 0xffffff81, 0x80010001, 0x80])

    def test_unproven_modes_fail_explicitly(self):
        for instruction, message in [
                (0x5a00000000000000 | (1 << 49), 'CC, X or SAT'),
                (0x5a00000000000000 | (1 << 50), 'CC, X or SAT'),
                (0x5a00000000000000 | (3 << 51), 'IMAD PO'),
                (0x5c38000000000000 | (1 << 47), 'CC, X or SAT'),
                (0x5b00000000000000 | (1 << 38), 'XMAD X'),
                (0x5b00000000000000 | (1 << 37) | (1 << 47), 'MRG+CC'),
                (0x5f00000000000000 | (3 << 51), 'reserved scale'),
                (0x5f00000000000000 | (1 << 47), 'VMAD CC'),
                (0x3a00000000000000 | (1 << 47), 'VMNMX CC'),
                (0x3a00000000000000 | (7 << 51), 'VMNMX op')]:
            with self.subTest(instruction=hex(instruction)):
                self.compile([instruction | PT], error=message)


if __name__ == '__main__':
    unittest.main()
