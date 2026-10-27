"""Original Maxwell storage-image fixtures; no game-derived shader bytes."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from synthetic import EXIT, mov32

SERVICE = Path(sys.argv.pop(1)).resolve()
CROSS = Path(sys.argv.pop(1)).resolve()
GPU = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else None
MODE = os.environ.get('TOTK_TEST_COMPILE_MODE', 'full')


def copy_shader(size, shape):
    instructions = [mov32(0) | register for register in range(4)]
    instructions += [0xf0c8000000000000 | (33 << 20) | (7 << 16) | 4,
                     mov32(0) | 5]
    # Separate input/output descriptors make signed load extension observable
    # through a four-component unsigned output, rather than truncating it again.
    source = (size << 20) | (1 << 52) if size < 7 else 15 << 20
    common = (1 << 51) | (shape << 33) | (7 << 16) | (4 << 8)
    instructions += [0xeb00000000000000 | common | source | (72 << 36),
                     0xeb20000000000000 | common | (1 << 52) | (6 << 20) | (74 << 36),
                     EXIT]
    code = bytearray()
    for instruction in instructions:
        if len(code) % 32 == 0:
            code += bytes(8)
        code += struct.pack('<Q', instruction)
    return code


class StorageImages(unittest.TestCase):
    def test_explicit_formats_override_usage_metadata_and_preserve_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = []
            for shape, name in ((0, '1d'), (3, '2d'), (1, 'buffer')):
                for size in range(8):
                    variant = f'image-{name}-{size}'
                    code = root / (variant + '.code')
                    code.write_bytes(copy_shader(size, shape))
                    requests.append(dict(variant=variant, stage='cs', code=str(code),
                                         local=[4, 1, 1], c1='', depth_minus_w=True))
            output = root / 'results'
            process = subprocess.run(
                [str(SERVICE), '--service', '--spirv-cross', str(CROSS), '--out', str(output),
                 '--chain', 'storage-image-regression', '--compile-mode', MODE],
                input=''.join(json.dumps(row) + '\n' for row in requests),
                capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stderr)
            replies = [json.loads(line) for line in process.stdout.splitlines()]
            self.assertEqual(len(replies), len(requests), process.stderr)
            for request, reply in zip(requests, replies):
                variant = request['variant']
                size = int(variant.rsplit('-', 1)[1])
                with self.subTest(variant=variant, mode=MODE):
                    if size == 7:
                        self.assertTrue(reply['ok'], reply)
                        self.assertIn('Compatibility fallback: typeless image read returns zero',
                                      process.stderr)
                        source = (output / (variant + '_b.metal')).read_text()
                        self.assertIn('.write(uint4(0u, 0u, 0u, 0u)', source)
                        self.assertNotIn('.read(', source)
                        validated = subprocess.run(
                            ['spirv-val', '--target-env', 'spv1.0', str(output / (variant + '.spv'))],
                            capture_output=True, text=True, timeout=30)
                        self.assertEqual(validated.returncode, 0, validated.stderr)
                        continue
                    self.assertTrue(reply['ok'], reply)
                    metadata = json.loads((output / (variant + '_b.json')).read_text())
                    images = [binding for binding in metadata['bindings'] if binding['kind'] == 'img']
                    self.assertEqual(len(images), 2)
                    expected = 'int' if size in (1, 3) else 'uint'
                    self.assertIn('<' + expected, images[0]['msl_type'])
                    self.assertIn('<uint', images[1]['msl_type'])
                    validated = subprocess.run(
                        ['spirv-val', '--target-env', 'spv1.0', str(output / (variant + '.spv'))],
                        capture_output=True, text=True, timeout=30)
                    self.assertEqual(validated.returncode, 0, validated.stderr)
                    if GPU:
                        executed = subprocess.run(
                            [str(GPU), str(output / (variant + '_b.metal')),
                             str(output / (variant + '_b.json')), str(size)],
                            capture_output=True, text=True, timeout=60)
                        self.assertEqual(executed.returncode, 0, executed.stderr)


if __name__ == '__main__':
    unittest.main()
