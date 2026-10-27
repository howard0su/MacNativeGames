"""Original synthetic Maxwell programs; no game data."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from synthetic import EXIT, graphics, mov32

PROBE = Path(sys.argv.pop(1)).resolve()
WITH_MSL = sys.argv.pop(1) == '1'
METAL_PROBE = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else None
class CompilerProbe(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def translate(self, code, stage='vs', msl=False):
        path = self.root / 'input.code'
        path.write_bytes(code)
        command = [str(PROBE), stage, str(path), str(self.root / 'out.spv')]
        if msl:
            command.append(str(self.root / 'out.metal'))
        return subprocess.run(command, capture_output=True, text=True, timeout=30)

    def words(self):
        data = (self.root / 'out.spv').read_bytes()
        self.assertEqual(len(data) % 4, 0)
        words = struct.unpack('<' + 'I' * (len(data) // 4), data)
        self.assertEqual(words[0], 0x07230203)
        instructions = []
        offset = 5
        while offset < len(words):
            count, opcode = words[offset] >> 16, words[offset] & 0xffff
            self.assertGreater(count, 0)
            self.assertLessEqual(offset + count, len(words))
            instructions.append((opcode, words[offset+1:offset+count]))
            offset += count
        return instructions

    def test_vertex_exit_emits_vertex_entry(self):
        result = self.translate(graphics([EXIT]))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')
        entries = [args for opcode, args in self.words() if opcode == 15]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][0], 0)  # SPIR-V Vertex execution model.

    def test_fragment_constant_color(self):
        # MOV32I R0, floatBits(0.5); EXIT; SPH enables target 0 component x.
        mov = mov32(0x3f000000)
        result = self.translate(graphics([mov, EXIT], color=True), 'ps', WITH_MSL)
        self.assertEqual(result.returncode, 0, result.stderr)
        instructions = self.words()
        entries = [args for opcode, args in instructions if opcode == 15]
        self.assertEqual(entries[0][0], 4)  # Fragment.
        self.assertTrue(any(opcode == 43 and args[-1] == 0x3f000000
                            for opcode, args in instructions))
        if WITH_MSL:
            source = (self.root / 'out.metal').read_text()
            self.assertIn('[[color(0)]]', source)
            self.assertIn('0.5', source)
            if METAL_PROBE:
                compiled = subprocess.run(
                    [str(METAL_PROBE), str(self.root / 'out.metal')],
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)

    def test_truncated_instruction_is_an_error(self):
        result = self.translate(graphics([EXIT])[:-1])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unaligned', result.stderr)
        self.assertFalse((self.root / 'out.spv').exists())

    def test_no_implicit_zero_filled_code(self):
        result = self.translate(graphics([0]))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'out.spv').exists())

    def test_unexecuted_constant_tail_need_not_be_instruction_aligned(self):
        result = self.translate(graphics([EXIT]) + b'\x11\x22\x33')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stubbed_instruction_is_not_success(self):
        # MOV32I with unsupported lane mask must not silently drop the write.
        mov = mov32(0, mask=2)
        result = self.translate(graphics([mov, EXIT]))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Masked Mov', result.stderr)
        self.assertFalse((self.root / 'out.spv').exists())

    def test_compute_does_not_invent_launch_state(self):
        code = bytes(8) + struct.pack('<Q', EXIT)
        result = self.translate(code, 'cs')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('compute workgroup', result.stderr)

    def test_unsupported_stage(self):
        result = self.translate(graphics([EXIT]), 'gs')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('stage must be', result.stderr)


if __name__ == '__main__':
    unittest.main()
