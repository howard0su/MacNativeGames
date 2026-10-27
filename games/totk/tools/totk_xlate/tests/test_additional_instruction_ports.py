"""Independent GPU checks for audited Maxwell operand forms and flag writes."""
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest

from synthetic import EXIT, graphics, mov32

SERVICE = Path(sys.argv.pop(1)).resolve()
METAL = Path(sys.argv.pop(1)).resolve()
PROBE = Path(sys.argv.pop(1)).resolve()
PT = 7 << 16
CC = 1 << 47
READ_FLAGS = 0x38e8000000000000 | PT | (255 << 8) | (1 << 40) | (15 << 20)
SET_OVERFLOW = 0x5c10000000000000 | PT | CC | (6 << 8) | (7 << 20) | 5


def mov(register, bits):
    return mov32(bits) | register


def immediate_float(bits):
    return (((bits >> 12) & 0x7ffff) << 20) | (((bits >> 31) & 1) << 56)


class AdditionalInstructionPorts(unittest.TestCase):
    def check_gpu(self, instructions, expected, *, c1=b""):
        with tempfile.TemporaryDirectory(prefix=".instruction-test-", dir=Path.cwd()) as directory:
            root = Path(directory)
            code = root / "shader.code"
            code.write_bytes(graphics([*instructions, EXIT], color=True))
            request = dict(variant="ports", stage="ps", code=str(code),
                           fout={"0": "uint"}, c1=c1.hex())
            result = subprocess.run(
                [str(SERVICE), "--service", "--spirv-cross", str(SERVICE),
                 "--out", str(root), "--chain", "yuzu-ports-test-only"],
                input=json.dumps(request) + "\n", text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            reply = json.loads(result.stdout)
            self.assertTrue(reply["ok"], result.stdout + result.stderr)
            validated = subprocess.run(
                ["spirv-val", "--target-env", "vulkan1.1", str(root / "ports.spv")],
                text=True, capture_output=True, timeout=30)
            self.assertEqual(validated.returncode, 0, validated.stderr)
            reflection = root / "ports_b.json"
            bindings = json.loads(reflection.read_text())["bindings"]
            checked = subprocess.run(
                [str(METAL), str(root / "ports_b.metal"), "uint",
                 f"{expected & 0xffffffff:08x}", *([str(reflection)] if bindings else [])],
                text=True, capture_output=True, timeout=60)
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def check_probe_gpu(self, instructions, expected):
        with tempfile.TemporaryDirectory(prefix=".instruction-test-", dir=Path.cwd()) as directory:
            root = Path(directory)
            code, spirv, metal = (root / name for name in ("input.code", "out.spv", "out.metal"))
            # Export the flags as an ordinary float, not subnormal raw float bits.
            convert = 0x5cb8000000000000 | PT | (2 << 8) | (2 << 10)
            code.write_bytes(graphics([*instructions, convert, EXIT], color=True))
            result = subprocess.run(
                [str(PROBE), "ps", str(code), str(spirv), str(metal)],
                text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            validated = subprocess.run(
                ["spirv-val", "--target-env", "vulkan1.1", str(spirv)],
                text=True, capture_output=True, timeout=30)
            self.assertEqual(validated.returncode, 0, validated.stderr)
            bits = struct.unpack("<I", struct.pack("<f", expected))[0]
            checked = subprocess.run(
                [str(METAL), str(metal), "float", f"{bits:08x}"],
                text=True, capture_output=True, timeout=60)
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_signed_integer_to_float_absolute_preserves_source_width(self):
        for source in ('register', 'cbuf'):
            for width, value, negate in (
                (64, -3, False), (64, 0, False),
                (64, 1 << 40, False), (64, -(1 << 40), False),
                (64, -3, True), (32, -3, False),
            ):
                with self.subTest(source=source, width=width, value=value, negate=negate):
                    raw = value & ((1 << width) - 1)
                    modifiers = (PT | (2 << 8) | ((3 if width == 64 else 2) << 10) |
                                 (1 << 13) | (1 << 49) | (negate << 45))
                    if source == 'register':
                        instructions = [mov(2, raw & 0xffffffff), mov(3, raw >> 32),
                                        0x5cb8000000000000 | modifiers | (2 << 20)]
                        c1 = b''
                    else:
                        instructions = [0x4cb8000000000000 | modifiers | (1 << 34)]
                        c1 = raw.to_bytes(width // 8, 'little')
                    expected = -abs(value) if negate else abs(value)
                    bits = struct.unpack('<I', struct.pack('<f', expected))[0]
                    self.check_gpu(instructions, bits, c1=c1)

    def test_immediate_range_reduction_preserves_modifiers(self):
        for bits, modifiers, expected in (
            (0x3fc00000, 0, 0x3fc00000),
            (0xbfc00000, 1 << 49, 0x3fc00000),
            (0x3fc00000, 1 << 45, 0xbfc00000),
        ):
            with self.subTest(bits=bits, modifiers=modifiers):
                instruction = 0x3890000000000000 | PT | immediate_float(bits) | modifiers
                self.check_gpu([instruction], expected)

    def test_immediate_float_to_integer_uses_rounding_and_source_width(self):
        for bits, rounding, expected in (
            (0x3fc00000, 0, 2),
            (0x40200000, 0, 2),
            (0xc0200000, 0, -2),
            (0xc0200000, 1, -3),
            (0xc0200000, 2, -2),
            (0xc0200000, 3, -2),
        ):
            with self.subTest(bits=bits, rounding=rounding):
                instruction = (0x38b0000000000000 | PT | (2 << 8) | (2 << 10) |
                               (1 << 12) | (rounding << 39) | immediate_float(bits))
                self.check_gpu([instruction], expected)
        for source, operand in ((1, 0x3e00 << 20), (3, 0x3ff80 << 20)):
            with self.subTest(source=source):
                instruction = (0x38b0000000000000 | PT | (2 << 8) |
                               (source << 10) | (1 << 12) | operand)
                self.check_gpu([instruction], 2)

    def test_register_and_constant_predicate_masks_preserve_unselected_bytes(self):
        for byte in range(4):
            shifted = 0x80 << (byte * 8)
            expected = (0x11223344 & ~shifted) | shifted
            for form in ("register", "constant"):
                with self.subTest(byte=byte, form=form):
                    prefix = [mov(4, 0x11223344)]
                    opcode = 0x5ce8000000000000 if form == "register" else 0x4ce8000000000000
                    if form == "register":
                        prefix.append(mov(1, 0x10080))
                        operand = 1 << 20
                    else:
                        operand = 1 << 34
                    self.check_gpu(
                        [*prefix, opcode | PT | (4 << 8) | (byte << 41) | operand],
                        expected, c1=struct.pack("<I", 0x10080))

    def test_immediate_predicate_mask_includes_pt_and_is_only_one_byte(self):
        instruction = 0x38e8000000000000 | PT | (4 << 8) | (0x10080 << 20)
        self.check_gpu([mov(4, 0x11223344), instruction], 0x112233c4)

    def test_float_condition_codes_use_numeric_zero_and_preserve_integer_flags(self):
        for opcode, extra in (
            (0x5c58000000000000, 0),  # FADD
            (0x5c68000000000000, 0),  # FMUL
            (0x5980000000000000, 3 << 39),  # FFMA
            (0x5c60000000000000, 7 << 39),  # FMNMX
        ):
            for lhs, rhs, addend, sign_zero in (
                (0, 0, 0, 1),
                (0xbf800000, 0x3f800000, 0, 2 if opcode != 0x5c58000000000000 else 1),
                (0x7fc00000, 0x3f800000, 0, 0),
            ):
                with self.subTest(opcode=f"{opcode:x}", lhs=lhs, rhs=rhs):
                    prefix = [mov(1, lhs), mov(2, rhs), mov(3, addend),
                              mov(6, 0x7fffffff), mov(7, 1)]
                    operation = opcode | PT | CC | (1 << 8) | (2 << 20) | 4 | extra
                    self.check_gpu([*prefix, SET_OVERFLOW, operation, READ_FLAGS], 8 | sign_zero)

    def test_double_condition_codes_use_the_final_numeric_result(self):
        for opcode, extra in (
            (0x5c70000000000000, 0),
            (0x5c80000000000000, 0),
            (0x5b70000000000000, 12 << 39),
            (0x5c50000000000000, 7 << 39),
        ):
            for high_a, high_b, sign_zero in (
                (0, 0, 1),
                (0xbff00000, 0x3ff00000, 1 if opcode == 0x5c70000000000000 else 2),
                (0x7ff80000, 0x3ff00000, 0),
            ):
                with self.subTest(opcode=f"{opcode:x}", a=high_a, b=high_b):
                    prefix = [mov(8, 0), mov(9, high_a), mov(10, 0), mov(11, high_b),
                              mov(12, 0), mov(13, 0), mov(6, 0x7fffffff), mov(7, 1)]
                    operation = opcode | PT | CC | (8 << 8) | (10 << 20) | 4 | extra
                    self.check_gpu([*prefix, SET_OVERFLOW, operation, READ_FLAGS], 8 | sign_zero)

    def test_swizzled_add_condition_codes_preserve_integer_overflow(self):
        for operand, zero in ((0, 1), (0x7fc00000, 0)):
            with self.subTest(operand=operand):
                instruction = 0x50f8000000000000 | PT | CC | (1 << 8) | (2 << 20) | 4
                self.check_probe_gpu(
                    [mov(1, operand), mov(2, operand), mov(6, 0x7fffffff), mov(7, 1),
                     SET_OVERFLOW, instruction, READ_FLAGS], 8 | zero)

    def test_integer_minmax_funnel_and_ternary_flags_preserve_carry_overflow(self):
        for operation, expected in (
            (0x5c20000000000000 | (7 << 39) | (1 << 48), 10),
            (0x5bf8000000000000 | (10 << 39), 10),
            (0x5be0000000000000 | (255 << 28) | (7 << 48), 10),
            (0x5be0000000000000 | (7 << 48), 9),
        ):
            with self.subTest(operation=f"{operation:x}"):
                prefix = [mov(8, 0x80000000), mov(9, 0), mov(10, 0x80000000),
                          mov(6, 0x7fffffff), mov(7, 1)]
                instruction = operation | PT | CC | (8 << 8) | (9 << 20) | 4
                self.check_gpu([*prefix, SET_OVERFLOW, instruction, READ_FLAGS], expected)

    def test_extended_logical_zero_accumulates_incoming_zero(self):
        for prime in (0, 1):
            for opcode, modifiers in (
                (0x5c40000000000000, (1 << 43) | (7 << 48)),
                (0x5be0000000000000, (1 << 38) | (7 << 48)),
            ):
                with self.subTest(prime=prime, opcode=f"{opcode:x}"):
                    prime_flags = (0x5c10000000000000 | PT | CC |
                                   (6 << 8) | (7 << 20) | 5)
                    instruction = opcode | PT | CC | (8 << 8) | (9 << 20) | modifiers | 4
                    self.check_gpu(
                        [mov(6, prime), mov(7, 0), mov(8, 0), mov(9, 0), prime_flags,
                         instruction, READ_FLAGS], int(prime == 0))

    def test_extended_integer_comparison_flags_follow_integer_or_float_boolean(self):
        for addend in (1, 2):
            for float_boolean in (False, True):
                with self.subTest(addend=addend, float_boolean=float_boolean):
                    prefix = [mov(6, 0xffffffff), mov(7, addend), mov(8, 17), mov(9, 17)]
                    instruction = (0x5b50000000000000 | PT | CC | (8 << 8) | (9 << 20) |
                                   (7 << 39) | (1 << 43) | (float_boolean << 44) |
                                   (2 << 49) | 4)
                    instructions = [*prefix, SET_OVERFLOW, instruction]
                    flags = (0 if addend == 1 else 1) if float_boolean else (2 if addend == 1 else 0)
                    self.check_gpu([*instructions, READ_FLAGS], flags)
                    value = (0x3f800000 if float_boolean else 0xffffffff) if addend == 1 else 0
                    self.check_gpu([*instructions, 0x5c98078000470000], value)

    def test_conditional_indirect_branch_preserves_fallthrough_and_table_targets(self):
        def address(index):
            return 0x80 + 8 * (4 * (index // 3) + index % 3 + 1)

        branch_index = 7
        table = struct.pack("<II", address(9) - address(branch_index) - 8,
                            address(11) - address(branch_index) - 8)
        for predicate in (0, 1):
            for inverted in (False, True):
                for index in (0, 1):
                    with self.subTest(predicate=predicate, inverted=inverted, index=index):
                        instructions = [
                            mov(0, 0x55), mov(6, predicate),
                            0x38f0000000000000 | PT | (6 << 8) | (1 << 20),
                            mov(1, index),
                            0x3820000000000000 | PT | (7 << 39) | (1 << 8) | (1 << 20) | 2,
                            0x3848000000000000 | PT | (2 << 8) | (2 << 20) | 2,
                            0xef90000000000000 | PT | (4 << 48) | (1 << 36) | (2 << 8) | 3,
                            0xe25000000000000f | (inverted << 19) | (3 << 8),
                            EXIT, mov(0, 0x11111111), EXIT, mov(0, 0x22222222),
                        ]
                        expected = (0x11111111, 0x22222222)[index] if predicate != inverted else 0x55
                        self.check_gpu(instructions, expected, c1=table)

    def test_never_taken_indirect_branch_does_not_require_a_table(self):
        for condition in ((15 | PT | (1 << 19)), PT):
            with self.subTest(condition=condition):
                self.check_gpu([mov(0, 0x55),
                                0xe250000000000000 | condition | (255 << 8)], 0x55)

    def test_native_tessellation_patch_levels_have_typed_inputs_and_correct_indices(self):
        for stage, storage in (("tes", "Input"), ("tcs", "Output")):
            for component in range(6):
                with self.subTest(stage=stage, component=component):
                    with tempfile.TemporaryDirectory(prefix=".patch-test-", dir=Path.cwd()) as directory:
                        root = Path(directory)
                        code, spirv = root / "input.code", root / "out.spv"
                        load = (0xefd8000000000000 | PT | (255 << 8) | (255 << 39) |
                                (1 << 31) | ((4 * component) << 20) | 4)
                        store = (0xeff0000000000000 | PT | (255 << 8) | (255 << 39) |
                                 ((0x20 if stage == "tcs" else 0x80) << 20) |
                                 ((stage == "tcs") << 31) | 4)
                        shader = bytearray(graphics([load, store, EXIT]))
                        struct.pack_into("<I", shader, 8, 1 << 24)
                        code.write_bytes(shader)
                        result = subprocess.run(
                            [str(PROBE), stage, str(code), str(spirv)],
                            text=True, capture_output=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        validated = subprocess.run(
                            ["spirv-val", "--target-env", "vulkan1.1", str(spirv)],
                            text=True, capture_output=True, timeout=30)
                        self.assertEqual(validated.returncode, 0, validated.stderr)
                        disassembled = subprocess.run(
                            ["spirv-dis", str(spirv)], text=True, capture_output=True, timeout=30)
                        self.assertEqual(disassembled.returncode, 0, disassembled.stderr)
                        text = disassembled.stdout
                        builtin = "TessLevelOuter" if component < 4 else "TessLevelInner"
                        variable = re.search(r"OpDecorate\s+(%\w+)\s+BuiltIn\s+" + builtin, text)
                        self.assertIsNotNone(variable, text)
                        identity = re.escape(variable.group(1))
                        self.assertRegex(text, r"OpDecorate\s+" + identity + r"\s+Patch")
                        self.assertRegex(text, identity + r"\s+=\s+OpVariable\s+%\w+\s+" + storage)
                        access = re.search(r"OpAccessChain\s+%\w+\s+" + identity + r"\s+(%\w+)", text)
                        self.assertIsNotNone(access, text)
                        index = component if component < 4 else component - 4
                        self.assertRegex(text, re.escape(access.group(1)) +
                                         r"\s+=\s+OpConstant\s+%\w+\s+" + str(index) + r"\s")

    def test_tessellation_patch_load_rejects_non_tessellation_stage(self):
        with tempfile.TemporaryDirectory(prefix=".patch-test-", dir=Path.cwd()) as directory:
            root = Path(directory)
            code = root / "input.code"
            load = 0xefd8000000000000 | PT | (255 << 8) | (255 << 39) | (1 << 31)
            code.write_bytes(graphics([load, EXIT], color=True))
            result = subprocess.run(
                [str(PROBE), "ps", str(code), str(root / "out.spv")],
                text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Patch load outside tessellation stage", result.stderr)


if __name__ == "__main__":
    unittest.main()
