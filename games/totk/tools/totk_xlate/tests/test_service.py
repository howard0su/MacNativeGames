"""Exercise the native service without treating synthetic success as parity."""
from unittest.mock import patch
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest

from synthetic import EXIT, graphics, mov32

SERVICE = Path(sys.argv.pop(1)).resolve()
CROSS = Path(sys.argv.pop(1)).resolve()
METAL = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else None
COMPILE_MODE = os.environ.get('TOTK_TEST_COMPILE_MODE', 'full')


class ShaderService(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.code = self.root / 'original.code'
        self.code.write_bytes(graphics([EXIT]))
        self.output = self.root / 'results'
        self.request = dict(variant='original-exit', stage='vs', code=str(self.code))

    def run_service(self, text, *, process=False, cross=CROSS, extra=(),
                    compile_mode=COMPILE_MODE, mode_env=None):
        environment = dict(os.environ)
        environment.pop('TOTK_SHADER_DEBUGMODE', None)
        environment.pop('TOTK_XLATE_SPIRV_CROSS', None)
        environment.pop('TOTK_SHADER_COMPILE_MODE', None)
        if mode_env is not None:
            environment['TOTK_SHADER_COMPILE_MODE'] = mode_env
        if process:
            environment['TOTK_XLATE_SPIRV_CROSS'] = 'process'
        mode_args = ['--compile-mode', compile_mode] if compile_mode else []
        return subprocess.run(
            [str(SERVICE), '--service', '--spirv-cross', str(cross),
             '--out', str(self.output), '--chain', 'totk-yuzu-test-v1',
             *mode_args, *extra],
            input=text, text=True, capture_output=True, timeout=60, env=environment)

    def send(self, *requests, **kwargs):
        completed = self.run_service(
            ''.join(json.dumps(request) + '\n' for request in requests), **kwargs)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        replies = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(len(replies), len(requests))
        return replies

    def reflection(self, variant=None):
        return json.loads(
            (self.output / ((variant or self.request['variant']) + '_b.json')).read_text())

    def test_protocol_recovers_and_publishes_contract(self):
        completed = self.run_service('\n \t\r\n{\n[]\n' + json.dumps(self.request) + '\n')
        self.assertEqual(completed.returncode, 0, completed.stderr)
        replies = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual([reply['ok'] for reply in replies], [False, False, True])
        self.assertIn('variant="original-exit"', completed.stderr)
        self.assertIn('[request 3] ok', completed.stderr)
        self.assertIn('[request 1] failed:', completed.stderr)
        reflection = self.reflection()
        self.assertEqual(reflection['key'], 'original-exit_b')
        self.assertEqual(reflection['chain'], 'totk-yuzu-test-v1')
        self.assertEqual(reflection['msl_entry_kind'], 'vertex')
        self.assertEqual(reflection['bindings'], [])
        self.assertEqual(reflection['compile_mode'], COMPILE_MODE)
        self.assertTrue((self.output / 'original-exit_b.metal').is_file())
        self.assertTrue((self.output / 'original-exit.spv').is_file())
        self.assertEqual(list(self.output.glob('.yuzu-request-*')), [])

    def test_compile_modes_preserve_code_and_resources(self):
        ldc = 0xef90000000000000 | (4 << 48) | (3 << 36) | (7 << 16) | (255 << 8)
        self.code.write_bytes(graphics([ldc, EXIT], color=True))
        request = dict(self.request, stage='ps', fout={'0': 'uint'})
        outputs = []
        for mode in ('full', 'online-fast'):
            completed = self.run_service(json.dumps(request)+'\n', compile_mode=mode)
            self.assertTrue(json.loads(completed.stdout)['ok'], completed.stderr)
            self.assertIn(f'IR verification={"on" if mode == "full" else "off"}', completed.stderr)
            self.assertIn('GPU loop watchdog=on', completed.stderr)
            reflection = self.reflection()
            self.assertEqual(reflection.pop('compile_mode'), mode)
            outputs.append((reflection, (self.output/'original-exit.spv').read_bytes(),
                            (self.output/'original-exit_b.metal').read_bytes()))
            if METAL:
                result = subprocess.run(
                    [str(METAL), str(self.output/'original-exit_b.metal'),
                     str(self.output/'original-exit_b.json')],
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs[0], outputs[1])

    def test_default_mode_and_explicit_mode_override_environment(self):
        for mode, environment, expected in ((None, None, 'full'),
                                            (None, 'online-fast', 'online-fast'),
                                            ('full', 'online-fast', 'full')):
            with self.subTest(mode=mode, environment=environment):
                self.assertTrue(self.send(self.request, compile_mode=mode,
                                          mode_env=environment)[0]['ok'])
                self.assertEqual(self.reflection()['compile_mode'], expected)

    def test_failed_retry_removes_stale_commit_marker(self):
        self.assertTrue(self.send(self.request)[0]['ok'])
        bad = dict(self.request, stage='gs')
        self.assertFalse(self.send(bad)[0]['ok'])
        self.assertFalse((self.output / 'original-exit_b.json').exists())

    def test_ir_diagnostics_preserve_protocol_and_translation(self):
        baseline = self.run_service(json.dumps(self.request) + '\n')
        self.assertTrue(json.loads(baseline.stdout)['ok'], baseline.stderr)
        expected = (self.output / 'original-exit.spv').read_bytes()
        with patch.dict(os.environ, {'TOTK_XLATE_DUMP_IR': '1'}):
            completed = self.run_service(json.dumps(self.request) + '\n')
        self.assertTrue(json.loads(completed.stdout)['ok'], completed.stderr)
        self.assertIn('Yuzu IR before resource lowering:', completed.stderr)
        self.assertEqual((self.output / 'original-exit.spv').read_bytes(), expected)

    def test_fp64_lowering_uses_one_frontend_and_preserves_constant_reads(self):
        # Load a c1 double, add it to itself, then convert to fragment color.
        load = (0xef90000000000000 | (5 << 48) | (1 << 36) |
                (7 << 16) | (255 << 8) | 2)
        add = 0x5c70000000000000 | (2 << 20) | (7 << 16) | (2 << 8) | 2
        convert = 0x5ca8000000000000 | (2 << 20) | (7 << 16) | (3 << 10) | (2 << 8)
        self.code.write_bytes(graphics([load, add, convert, EXIT], color=True))
        request = dict(self.request, stage='ps', c1=struct.pack('<d', 2.25).hex())
        for process in (False, True):
            with self.subTest(process=process):
                with patch.dict(os.environ, {'TOTK_XLATE_DUMP_IR': '1'}):
                    completed = self.run_service(json.dumps(request) + '\n', process=process)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertTrue(json.loads(completed.stdout)['ok'], completed.stderr)
                self.assertEqual(completed.stderr.count('Yuzu IR before resource lowering:'), 1)
                self.assertIn('FPAdd32', completed.stderr)
                self.assertNotIn('FPAdd64', completed.stderr)
                reflection = self.reflection()
                self.assertNotIn('softfloat64', reflection)
                self.assertEqual([binding['kind'] for binding in reflection['bindings']], ['c1'])
                if METAL:
                    result = subprocess.run(
                        [str(METAL), str(self.output / 'original-exit_b.metal'),
                         str(self.output / 'original-exit_b.json')],
                        text=True, capture_output=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_requests_do_not_poison_next_request(self):
        for change in (dict(code=str(self.root / 'missing.code')), dict(c1='0'),
                       dict(c1='xx'), dict(attr={'32': 'float'}),
                       dict(fout={'0': 'unsupported'}), dict(tex=[]),
                       dict(depth_minus_w='false'), dict(variant='../escape')):
            with self.subTest(change=change):
                replies = self.send(dict(self.request, **change), self.request)
                self.assertEqual([reply['ok'] for reply in replies], [False, True])

    def test_compute_does_not_invent_memory_state(self):
        self.code.write_bytes(bytes(8) + struct.pack('<Q', EXIT))
        reply = self.send(dict(self.request, stage='cs'))[0]
        self.assertFalse(reply['ok'])
        self.assertIn('thread group', reply['error'])
        self.assertFalse((self.output / 'original-exit_b.json').exists())
        # No memory allocation is needed for a kernel that never accesses it.
        request = dict(self.request, stage='cs', local=[1, 1, 1])
        self.assertTrue(self.send(request)[0]['ok'])

    def test_compute_with_explicit_launch_and_memory_state(self):
        self.code.write_bytes(bytes(8) + struct.pack('<Q', EXIT))
        request = dict(self.request, stage='cs', local=[2, 3, 1], local_memory=0, shared_memory=0)
        reply = self.send(request)[0]
        self.assertTrue(reply['ok'], reply)
        reflection = self.reflection()
        self.assertEqual(reflection['msl_entry_kind'], 'kernel')
        self.assertEqual(reflection['threadgroup_size'], [2, 3, 1])

    def test_compute_memory_access_requires_the_allocation_it_uses(self):
        for opcode, field, message in (
                (0xef50000000000000, 'local_memory', 'local memory allocation'),
                (0xef58000000000000, 'shared_memory', 'shared memory allocation')):
            with self.subTest(field=field):
                store = opcode | (4 << 48) | (7 << 16) | (255 << 8)
                self.code.write_bytes(graphics([mov32(1), store, EXIT])[0x50:])
                request = dict(self.request, stage='cs', local=[1, 1, 1])
                reply = self.send(request)[0]
                self.assertFalse(reply['ok'])
                self.assertIn(message, reply['error'])
                request[field] = 4
                reply = self.send(request)[0]
                self.assertTrue(reply['ok'], reply)
    def test_shader_failure_is_recoverable(self):
        broken = self.root / 'broken.code'
        broken.write_bytes(graphics([mov32(0, mask=2), EXIT]))
        replies = self.send(dict(self.request, code=str(broken)), self.request)
        self.assertEqual([reply['ok'] for reply in replies], [False, True])
        self.assertIn('Masked Mov', replies[0]['error'])

    def test_used_resource_has_actual_argument_id_and_logical_slot(self):
        # LDC.32 R0, c[3][RZ]; keep the result live through fragment color 0.
        ldc = 0xef90000000000000 | (4 << 48) | (3 << 36) | (7 << 16) | (255 << 8)
        self.code.write_bytes(graphics([ldc, EXIT], color=True))
        reply = self.send(dict(self.request, stage='ps'))[0]
        self.assertTrue(reply['ok'], reply)
        reflection = self.reflection()
        self.assertEqual(len(reflection['bindings']), 1)
        binding = reflection['bindings'][0]
        self.assertEqual((binding['kind'], binding['reg']), ('cbuf', 0))
        metal = (self.output / 'original-exit_b.metal').read_text()
        self.assertIn(f"{binding['msl_name']} [[id({binding['arg_id']})]]", metal)
        self.assertIn('_m0', metal)
        self.assertTrue(self.send(dict(self.request, stage='ps'), process=True)[0]['ok'])
        self.assertEqual(self.reflection()['bindings'], reflection['bindings'])
        if METAL:
            result = subprocess.run(
                [str(METAL), str(self.output / 'original-exit_b.metal'),
                 str(self.output / 'original-exit_b.json')],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_vertex_declarations_preserve_used_attribute_types_and_dependencies(self):
        common = (7 << 16) | (255 << 8) | (255 << 39)
        load = 0xefd8000000000000 | common | (0x80 << 20)
        store = 0xeff0000000000000 | common | (0x70 << 20)
        self.code.write_bytes(graphics([load, store, EXIT]))
        for kind, msl_type in (('float', 'float4'), ('sint', 'int4'), ('uint', 'uint4')):
            with self.subTest(kind=kind):
                request = dict(self.request, attr={'0': kind, '1': 'uint'})
                self.assertTrue(self.send(request)[0]['ok'])
                reflection = self.reflection()
                self.assertEqual(reflection['vertex_inputs'], [dict(location=0, msl_type=msl_type)])
                self.assertEqual(reflection['deps']['attr'], {'0': kind})
                self.assertTrue(self.send(request, process=True)[0]['ok'])
                self.assertEqual(self.reflection(), reflection)
                if METAL:
                    result = subprocess.run(
                        [str(METAL), str(self.output/'original-exit_b.metal'),
                         str(self.output/'original-exit_b.json')],
                        capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_truncated_instructions_fail_without_publishing(self):
        self.code.write_bytes(graphics([EXIT])[:-1])
        reply = self.send(self.request)[0]
        self.assertFalse(reply['ok'])
        self.assertIn('instruction', reply['error'])
        self.assertFalse((self.output/'original-exit_b.json').exists())

    def test_bounded_dynamic_c1_read_and_negative_controls(self):
        # Load an unknown user index, clamp to five, scale to bytes, then read c1.
        ldc_index = (0xef90000000000000 | (4 << 48) | (3 << 36) |
                     (7 << 16) | (255 << 8) | 1)
        minimum = 0x3820000000000000 | (7 << 39) | (5 << 20) | (7 << 16) | (1 << 8) | 1
        shift = 0x3848000000000000 | (2 << 20) | (7 << 16) | (1 << 8) | 1
        ldc_c1 = 0xef90000000000000 | (4 << 48) | (1 << 36) | (7 << 16) | (1 << 8)
        instructions = [ldc_index, minimum, shift, ldc_c1, EXIT]
        self.code.write_bytes(graphics(instructions, color=True))
        request = dict(self.request, stage='ps', fout={'0': 'uint'},
                       c1=struct.pack('<6I', 0, 0, 0, 0, 0, 0x3f000000).hex())
        reply = self.send(request)[0]
        self.assertTrue(reply['ok'], reply)
        if METAL:
            completed = subprocess.run(
                [str(METAL), str(self.output / 'original-exit_b.metal'), 'uint', '3f000000',
                 str(self.output / 'original-exit_b.json')],
                text=True, capture_output=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
        for code, c1 in ((instructions, request['c1'][:-8]),
                         ([ldc_index, shift, ldc_c1, EXIT], request['c1'])):
            with self.subTest(c1_bytes=len(c1) // 2, clamped=code == instructions):
                self.code.write_bytes(graphics(code, color=True))
                reply = self.send(dict(request, c1=c1))[0]
                self.assertFalse(reply['ok'], reply)
                self.assertIn('dynamic c1 read cannot be bounded', reply['error'])

    def test_dynamic_c1_reads_require_declared_zero_padded_allocation_and_guard_loads(self):
        ldc_index = (0xef90000000000000 | (4 << 48) | (3 << 36) |
                     (7 << 16) | (255 << 8) | 1)
        shift = 0x3848000000000000 | (7 << 16) | (1 << 8) | 1
        ldc_c1 = 0xef90000000000000 | (1 << 36) | (7 << 16) | (1 << 8)
        for width in (4, 8):
            for index_offset, data, capacity, expected in (
                    (0, [1] * (7 * width // 4) + [0x3f000000, 1], 65536, '3f000000'),
                    (0, [1], 65536, '00000000'),
                    (0, [1, 2, 3, 4], 16, '00000000'),
                    (0, [1] * (7 * width // 4) + [0x3f000000], 7 * width + 4,
                     '3f000000' if width == 4 else '00000000'),
                    (16, [1, 2, 3, 4], 65536, '00000000')):
                instructions = [ldc_index | (index_offset << 20),
                                shift | ((2 if width == 4 else 3) << 20),
                                ldc_c1 | ((4 if width == 4 else 5) << 48), EXIT]
                self.code.write_bytes(graphics(instructions, color=True))
                request = dict(self.request, stage='ps', fout={'0': 'uint'},
                               c1=struct.pack('<' + 'I' * len(data), *data).hex(),
                               c1_buffer_size=capacity)
                for process in (False, True):
                    with self.subTest(width=width, offset=index_offset, capacity=capacity,
                                      words=len(data), process=process):
                        reply = self.send(request, process=process)[0]
                        self.assertTrue(reply['ok'], reply)
                        if METAL:
                            result = subprocess.run(
                                [str(METAL), str(self.output / 'original-exit_b.metal'),
                                 'uint', expected, str(self.output / 'original-exit_b.json')],
                                text=True, capture_output=True, timeout=60)
                            self.assertEqual(result.returncode, 0, result.stderr)
        for capacity in (0, 3, 65537, -1):
            reply = self.send(dict(request, c1_buffer_size=capacity))[0]
            self.assertFalse(reply['ok'], reply)
            self.assertFalse((self.output / 'original-exit_b.json').exists())

    def test_guarded_dynamic_c1_float_load_preserves_values_and_zero_padding(self):
        ldc_index = (0xef90000000000000 | (4 << 48) | (3 << 36) |
                     (7 << 16) | (255 << 8) | 1)
        shift = 0x3848000000000000 | (2 << 20) | (7 << 16) | (1 << 8) | 1
        ldc_c1 = (0xef90000000000000 | (4 << 48) | (1 << 36) |
                  (7 << 16) | (1 << 8))
        add = 0x5c58000000000000 | (2 << 20) | (7 << 16)
        self.code.write_bytes(graphics(
            [ldc_index, shift, ldc_c1, mov32(0x3f800000) | 2, add, EXIT], color=True))
        for data, capacity, expected in (([1] * 7 + [0x3f000000], 65536, '3fc00000'),
                                         ([1], 65536, '3f800000'),
                                         ([1, 2, 3, 4], 16, '3f800000')):
            request = dict(self.request, stage='ps', c1_buffer_size=capacity,
                           c1=struct.pack('<' + 'I' * len(data), *data).hex())
            for process in (False, True):
                with self.subTest(words=len(data), capacity=capacity, process=process):
                    reply = self.send(request, process=process)[0]
                    self.assertTrue(reply['ok'], reply)
                    if METAL:
                        result = subprocess.run(
                            [str(METAL), str(self.output / 'original-exit_b.metal'),
                             'float', expected, str(self.output / 'original-exit_b.json')],
                            text=True, capture_output=True, timeout=60)
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_storage_pointer_from_user_uniform_is_a_supported_binding(self):
        # LDC.64 R2, c[6][RZ]; optional STG.E.32; LDG.E.32 R0, [R2+8].
        pointer = (0xef90000000000000 | (5 << 48) | (6 << 36) |
                   (7 << 16) | (255 << 8) | 2)
        address = (4 << 48) | (1 << 45) | (8 << 20) | (7 << 16) | (2 << 8)
        load = 0xeed0000000000000 | address
        store = 0xeed8000000000000 | address
        request = dict(self.request, stage='ps', fout={'0': 'uint'})
        for written in (False, True):
            with self.subTest(written=written):
                expected = 0x12345678 if written else 0x3f000000
                instructions = [pointer]
                if written:
                    instructions += [mov32(expected), store]
                self.code.write_bytes(graphics(instructions + [load, EXIT], color=True))
                reply = self.send(request)[0]
                self.assertTrue(reply['ok'], reply)
                bindings = self.reflection()['bindings']
                storage = [binding for binding in bindings if binding['kind'] == 'sbuf_ubo']
                self.assertEqual(len(storage), 1)
                self.assertEqual((storage[0]['reg'], storage[0]['off']), (3, 0))
                self.assertEqual(storage[0]['rw'], written)
                self.assertTrue(self.send(request, process=True)[0]['ok'])
                self.assertEqual(self.reflection()['bindings'], bindings)
                if METAL:
                    completed = subprocess.run(
                        [str(METAL), str(self.output / 'original-exit_b.metal'),
                         'uint', f'{expected:08x}', str(self.output / 'original-exit_b.json')],
                        capture_output=True, text=True, timeout=60)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
        unsupported = pointer & ~(0x1f << 36)
        self.code.write_bytes(graphics([unsupported, load, EXIT], color=True))
        reply = self.send(request)[0]
        self.assertFalse(reply['ok'], reply)
        self.assertIn('storage buffer pointer has no supported NVN slot', reply['error'])
        self.code.write_bytes(graphics([mov32(16) | 2, mov32(0) | 3, load, EXIT], color=True))
        reply = self.send(request)[0]
        self.assertFalse(reply['ok'], reply)
        self.assertIn('Storage buffer failed to track', reply['error'])

    def test_compute_storage_pointer_loaded_through_uniform_address_table(self):
        ldc = 0xef90000000000000 | (5 << 48) | (7 << 16) | (255 << 8)
        global_address = (1 << 45) | (7 << 16)
        uniform_pointer = ldc | (0x240 << 20) | 2
        storage_pointer = (0xeed0000000000000 | global_address |
                           (5 << 48) | (2 << 8) | 4)
        load = (0xeed0000000000000 | global_address |
                (4 << 48) | (8 << 20) | (4 << 8))
        output_pointer = ldc | (0x310 << 20) | 6
        store = 0xeed8000000000000 | global_address | (4 << 48) | (6 << 8)
        self.code.write_bytes(graphics(
            [uniform_pointer, storage_pointer, load, output_pointer, store, EXIT])[0x50:])
        request = dict(self.request, stage='cs', local=[1, 1, 1])
        for process in (False, True):
            with self.subTest(process=process):
                reply = self.send(request, process=process)[0]
                self.assertTrue(reply['ok'], reply)
                bindings = self.reflection()['bindings']
                self.assertTrue(any(b['kind'] == 'cbuf' and b['reg'] == 3 for b in bindings))
                storage = [b for b in bindings if b['kind'] == 'sbuf_ubo']
                self.assertEqual([(b['reg'], b['off']) for b in storage], [(3, 0)])
                if METAL:
                    result = subprocess.run(
                        [str(METAL), str(self.output / 'original-exit_b.metal'), 'compute',
                         '3f000000', str(self.output / 'original-exit_b.json')],
                        capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
        invalid_pointer = uniform_pointer & ~(0xffff << 20)
        self.code.write_bytes(graphics(
            [invalid_pointer, storage_pointer, load, output_pointer, store, EXIT])[0x50:])
        reply = self.send(request)[0]
        self.assertFalse(reply['ok'], reply)
        self.assertFalse((self.output / 'original-exit_b.json').exists())

    def test_compute_pointer_spills_preserve_real_storage_writes(self):
        ldc = (0xef90000000000000 | (5 << 48) | (0x310 << 20) |
               (7 << 16) | (255 << 8) | 2)
        global_store = (0xeed8000000000000 | (4 << 48) | (1 << 45) |
                        (7 << 16) | (4 << 8))
        for memory, store_op, load_op in (
                ('shared_memory', 0xef58000000000000, 0xef48000000000000),
                ('local_memory', 0xef50000000000000, 0xef40000000000000)):
            for wide in (False, True):
                size = 5 if wide else 4
                store = store_op | (size << 48) | (7 << 16) | (255 << 8) | 2
                load = load_op | (size << 48) | (7 << 16) | (255 << 8) | 4
                spills = [store] if wide else [store, store | (4 << 20) | 1]
                reloads = [load] if wide else [load, load | (4 << 20) | 1]
                self.code.write_bytes(graphics(
                    [ldc, *spills, *reloads, mov32(0x12345678), global_store, EXIT])[0x50:])
                request = dict(self.request, stage='cs', local=[1, 1, 1], **{memory: 16})
                for process in (False, True):
                    with self.subTest(memory=memory, wide=wide, process=process):
                        reply = self.send(request, process=process)[0]
                        self.assertTrue(reply['ok'], reply)
                        self.assertTrue(any(b['kind'] == 'sbuf' and b['rw']
                                            for b in self.reflection()['bindings']))
                        if METAL:
                            result = subprocess.run(
                                [str(METAL), str(self.output / 'original-exit_b.metal'),
                                 'compute', '12345678',
                                 str(self.output / 'original-exit_b.json')],
                                capture_output=True, text=True, timeout=60)
                            self.assertEqual(result.returncode, 0, result.stderr)

    def test_compute_strided_shared_pointer_and_unrelated_field(self):
        # Each invocation owns a 12-byte record: pointer.low, pointer.high, unrelated data.
        tid = 0xf0c8000000000000 | (33 << 20) | (7 << 16) | 1
        stride = 0x3600000000000000 | (255 << 39) | (12 << 20) | (7 << 16) | (1 << 8) | 1
        pointer = (0xef90000000000000 | (5 << 48) | (0x310 << 20) |
                   (7 << 16) | (255 << 8) | 2)
        store = 0xef58000000000000 | (4 << 48) | (7 << 16) | (1 << 8)
        load = 0xef48000000000000 | (4 << 48) | (7 << 16) | (1 << 8)
        global_store = (0xeed8000000000000 | (4 << 48) | (1 << 45) |
                        (7 << 16) | (4 << 8))
        self.code.write_bytes(graphics(
            [tid, stride, pointer, mov32(0x12345678), store | (8 << 20),
             store | 2, store | (4 << 20) | 3, load | 4, load | (4 << 20) | 5,
             global_store, EXIT])[0x50:])
        request = dict(self.request, stage='cs', local=[1, 1, 1], shared_memory=12)
        for process in (False, True):
            with self.subTest(process=process):
                reply = self.send(request, process=process)[0]
                self.assertTrue(reply['ok'], reply)
                if METAL:
                    result = subprocess.run(
                        [str(METAL), str(self.output / 'original-exit_b.metal'),
                         'compute', '12345678', str(self.output / 'original-exit_b.json')],
                        capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_spilled_user_pointer_keeps_load_offset_and_binding(self):
        pointer = (0xef90000000000000 | (5 << 48) | (6 << 36) |
                   (7 << 16) | (255 << 8) | 2)
        store = 0xef58000000000000 | (5 << 48) | (7 << 16) | (255 << 8) | 2
        load = 0xef48000000000000 | (5 << 48) | (7 << 16) | (255 << 8) | 4
        read = (0xeed0000000000000 | (4 << 48) | (1 << 45) |
                (8 << 20) | (7 << 16) | (4 << 8))
        output = (pointer & ~(0x1f << 36)) | (0x310 << 20) | 4
        write = (0xeed8000000000000 | (4 << 48) | (1 << 45) |
                 (7 << 16) | (6 << 8))
        self.code.write_bytes(graphics([pointer, store, load, read, output, write, EXIT])[0x50:])
        request = dict(self.request, stage='cs', local=[1, 1, 1], shared_memory=8)
        for process in (False, True):
            with self.subTest(process=process):
                reply = self.send(request, process=process)[0]
                self.assertTrue(reply['ok'], reply)
                storage = [b for b in self.reflection()['bindings'] if b['kind'] == 'sbuf_ubo']
                self.assertEqual([(b['reg'], b['off'], b['rw']) for b in storage], [(3, 0, False)])
                if METAL:
                    result = subprocess.run(
                        [str(METAL), str(self.output / 'original-exit_b.metal'),
                         'compute', '3f000000', str(self.output / 'original-exit_b.json')],
                        capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_pointer_spills_reject_missing_clobbered_and_mismatched_origins(self):
        pointer = (0xef90000000000000 | (5 << 48) | (0x310 << 20) |
                   (7 << 16) | (255 << 8) | 2)
        other_pointer = (pointer & ~(0xffff << 20)) | (0x320 << 20) | 4
        tid = 0xf0c8000000000000 | (33 << 20) | (7 << 16) | 1
        compare = 0x3660000000000000 | (5 << 49) | (7 << 39) | (7 << 16) | (1 << 8) | 7
        global_store = (0xeed8000000000000 | (4 << 48) | (1 << 45) |
                        (7 << 16) | (4 << 8))
        for memory, store_op, load_op in (
                ('shared_memory', 0xef58000000000000, 0xef48000000000000),
                ('local_memory', 0xef50000000000000, 0xef40000000000000)):
            store = store_op | (4 << 48) | (7 << 16) | (255 << 8) | 2
            load = load_op | (5 << 48) | (7 << 16) | (255 << 8) | 4
            stores = [store, store | (4 << 20) | 1]
            cases = {
                'missing': [pointer],
                'unknown overwrite': [pointer, *stores, mov32(16) | 2, store],
                'different target': [pointer, *stores, other_pointer, store | 4],
                'mismatched high': [pointer, store, other_pointer, store | (4 << 20) | 5],
                'partial overwrite': [pointer, *stores, store_op | (7 << 16) |
                                      (255 << 8) | (1 << 20) | 2],
                'conditional initializer': [tid, compare, pointer,
                                            *(s & ~(7 << 16) for s in stores)],
            }
            request = dict(self.request, stage='cs', local=[2, 1, 1], **{memory: 16})
            for case, instructions in cases.items():
                self.code.write_bytes(graphics(
                    [*instructions, load, mov32(0x12345678), global_store, EXIT])[0x50:])
                for process in (False, True):
                    with self.subTest(memory=memory, case=case, process=process):
                        reply = self.send(request, process=process)[0]
                        self.assertFalse(reply['ok'], reply)
                        self.assertIn('Storage buffer failed to track', reply['error'])
                        self.assertFalse((self.output / 'original-exit_b.json').exists())

    def test_shared_spill_address_is_not_pointer_provenance(self):
        pointer = (0xef90000000000000 | (5 << 48) | (0x310 << 20) |
                   (7 << 16) | (255 << 8) | 2)
        address = (0xef90000000000000 | (4 << 48) | (6 << 36) |
                   (7 << 16) | (255 << 8) | 1)
        store = 0xef58000000000000 | (5 << 48) | (7 << 16) | (1 << 8) | 2
        load = 0xef48000000000000 | (5 << 48) | (7 << 16) | (1 << 8) | 4
        write = (0xeed8000000000000 | (4 << 48) | (1 << 45) |
                 (7 << 16) | (4 << 8))
        self.code.write_bytes(graphics(
            [pointer, address, store, load, mov32(0x12345678), write, EXIT])[0x50:])
        request = dict(self.request, stage='cs', local=[1, 1, 1], shared_memory=16)
        for process in (False, True):
            with self.subTest(process=process):
                reply = self.send(request, process=process)[0]
                self.assertFalse(reply['ok'], reply)
                self.assertIn('Storage buffer failed to track', reply['error'])
                self.assertFalse((self.output / 'original-exit_b.json').exists())

    def test_compute_entry_skips_noninstruction_prefix_and_checks_bounds(self):
        ldc = 0xef90000000000000 | (5 << 48) | (0x310 << 20) | (7 << 16) | (255 << 8) | 2
        store = (0xeed8000000000000 | (4 << 48) | (1 << 45) |
                 (7 << 16) | (2 << 8))
        prefix = b'\xff' * 0x100
        code = prefix + graphics([ldc, mov32(0x12345678), store, EXIT])[0x50:]
        self.code.write_bytes(code)
        request = dict(self.request, stage='cs', local=[1, 1, 1], entry_point=0x100)
        for process in (False, True):
            reply = self.send(request, process=process)[0]
            self.assertTrue(reply['ok'], reply)
            if METAL:
                result = subprocess.run(
                    [str(METAL), str(self.output / 'original-exit_b.metal'), 'compute',
                     '12345678', str(self.output / 'original-exit_b.json')],
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
        for entry in (1, len(code), 0xfffffff8):
            with self.subTest(entry=entry):
                reply = self.send(dict(request, entry_point=entry))[0]
                self.assertFalse(reply['ok'], reply)
                self.assertIn('compute entry point', reply['error'])
                self.assertFalse((self.output / 'original-exit_b.json').exists())
        reply = self.send(dict(request, stage='vs'))[0]
        self.assertFalse(reply['ok'], reply)
        self.assertIn('only valid for cs', reply['error'])

    def test_texture_aliases_keep_distinct_final_metal_field_names(self):
        # Sample the same handle normally and with depth comparison; both results
        # stay live through the sum written to fragment color 0.
        common = (0xd800000000000000 | (8 << 36) | (255 << 28) |
                  (255 << 20) | (7 << 16) | (2 << 8))
        sample = common | (2 << 53)
        compare = common | (6 << 53) | 4
        add = 0x5c58000000000000 | (4 << 20) | (7 << 16)
        self.code.write_bytes(graphics([mov32(0) | 2, mov32(0) | 3,
                                       sample, compare, add, EXIT], color=True))
        request = dict(self.request, stage='ps')
        reply = self.send(request)[0]
        self.assertTrue(reply['ok'], reply)
        bindings = self.reflection()['bindings']
        textures = [binding for binding in bindings if binding['kind'] == 'tex']
        self.assertEqual(len(textures), 2)
        self.assertEqual([binding['reg'] for binding in textures], [0, 0])
        self.assertEqual(len({binding['msl_name'] for binding in textures}), 2)
        metal = self.output / 'original-exit_b.metal'
        for binding in bindings:
            self.assertIn(f"{binding['msl_name']} [[id({binding['arg_id']})]]", metal.read_text())
        self.assertTrue(self.send(request, process=True)[0]['ok'])
        self.assertEqual(self.reflection()['bindings'], bindings)
        if METAL:
            compiled = subprocess.run([str(METAL), str(metal), str(self.output / 'original-exit_b.json')],
                                      capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)

    def test_precise_arithmetic_optimizes_without_contracting_on_modern_and_legacy_metal(self):
        load = 0xef90000000000000 | (4 << 48) | (3 << 36) | (7 << 16) | (255 << 8)
        multiply = 0x5c68000000000000 | (1 << 20) | (7 << 16)
        add = 0x5c58000000000000 | (2 << 20) | (7 << 16)
        self.code.write_bytes(graphics([
            load | (32 << 20), load | (36 << 20) | 1,
            load | (40 << 20) | 2, multiply, add, EXIT], color=True))
        request = dict(self.request, stage='ps')
        for process in (False, True):
            self.assertTrue(self.send(request, process=process)[0]['ok'])
            source = self.output / 'original-exit_b.metal'
            self.assertIn('nvnFPMul(', source.read_text())
            self.assertIn('#pragma metal fp contract(off)', source.read_text())
            if METAL:
                for version in ('2.3', '3.2'):
                    with self.subTest(process=process, version=version):
                        environment = dict(os.environ, TOTK_TEST_MSL_VERSION=version)
                        result = subprocess.run(
                            [str(METAL), str(source), 'float', '00000000',
                             str(self.output / 'original-exit_b.json')],
                            capture_output=True, text=True, timeout=60, env=environment)
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fragment_depth_is_not_converted_from_vertex_clip_space(self):
        for bits in (0x3e800000, 0x3f400000):
            code = graphics([mov32(bits) | 1, EXIT])
            struct.pack_into('<I', code, 0, (5 << 10) | 2)
            struct.pack_into('<I', code, 0x4c, 2)
            self.code.write_bytes(code)
            for depth_minus_w in (False, True):
                request = dict(self.request, stage='ps', depth_minus_w=depth_minus_w)
                for process in (False, True):
                    with self.subTest(depth=bits, depth_minus_w=depth_minus_w, process=process):
                        self.assertTrue(self.send(request, process=process)[0]['ok'])
                        if METAL:
                            result = subprocess.run(
                                [str(METAL), str(self.output / 'original-exit_b.metal'),
                                 'depth', f'{bits:08x}'],
                                capture_output=True, text=True, timeout=60)
                            self.assertEqual(result.returncode, 0, result.stderr)

    def test_mipmap_query_preserves_component_order_and_signed_fractional_levels(self):
        position = 0xe000000000000000 | (255 << 20) | (7 << 16) | (255 << 8)
        multiply = 0x5c68000000000000 | (6 << 20) | (7 << 16)
        query = (0xdf58000000000000 | (8 << 36) | (3 << 31) |
                 (2 << 28) | (255 << 20) | (7 << 16) | (4 << 8))
        # Ignore GPU-specific low LOD bits while retaining signed quarter-mip resolution.
        mask = 0x0400000000000000 | (0xffffffc0 << 20) | (7 << 16)
        for scale, expected in (
                (0x3ec00000, 'ffffff80,00000000,00000000,00000000'),
                (0x3f400000, '00000080,00000000,00000000,00000000')):
            code = graphics([
                position | (0x1c << 30) | 4,
                position | (0x1d << 30) | 5,
                mov32(scale) | 6,
                multiply | (4 << 8) | 4,
                multiply | (5 << 8) | 5,
                query, mask, mov32(0) | 2, mov32(0) | 3, EXIT], color=True)
            struct.pack_into('<I', code, 0, (5 << 10) | 2)
            struct.pack_into('<I', code, 0x48, 15)
            self.code.write_bytes(code)
            request = dict(self.request, stage='ps', fout={'0': 'uint'})
            for process in (False, True):
                with self.subTest(scale=scale, process=process):
                    self.assertTrue(self.send(request, process=process)[0]['ok'])
                    if METAL:
                        result = subprocess.run(
                            [str(METAL), str(self.output / 'original-exit_b.metal'),
                             'uint', expected, str(self.output / 'original-exit_b.json')],
                            capture_output=True, text=True, timeout=60)
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_depth_samples_fetches_and_gathers_preserve_channels_and_filtering(self):
        common = ((7 << 51) | (8 << 36) | (15 << 31) | (2 << 28) |
                  (255 << 20) | (7 << 16) | (4 << 8))
        cases = [
            ('implicit', 0xc000000000000000, 0x3f000000,
             '3ee00000,00000000,00000000,3f800000'),
            ('explicit', 0xc000000000000000 | (1 << 55), 0x3f000000,
             '3ee00000,00000000,00000000,3f800000'),
            ('fetch', 0xdc00000000000000, 0,
             '3e000000,00000000,00000000,3f800000'),
            ('gather-red', 0xc800000000000000, 0x3f000000,
             '3f200000,3f400000,3e800000,3e000000'),
            ('gather-green', 0xc800000000000000 | (1 << 56), 0x3f000000,
             '00000000,00000000,00000000,00000000'),
            ('gather-alpha', 0xc800000000000000 | (3 << 56), 0x3f000000,
             '3f800000,3f800000,3f800000,3f800000'),
        ]
        for name, opcode, coordinate, expected in cases:
            code = graphics([mov32(coordinate) | 4, mov32(coordinate) | 5,
                             opcode | common, EXIT], color=True)
            struct.pack_into('<I', code, 0x48, 15)
            self.code.write_bytes(code)
            request = dict(self.request, stage='ps', tex={'0': '2d_depth'})
            for process in (False, True):
                with self.subTest(operation=name, process=process):
                    reply = self.send(request, process=process)[0]
                    self.assertTrue(reply['ok'], reply)
                    reflection = self.reflection()
                    self.assertEqual(reflection['compare_slots'], [])
                    self.assertEqual(reflection['deps']['tex']['0'],
                                     '2d_depth' if name == 'implicit' else '*_depth')
                    textures = [b for b in reflection['bindings'] if b['kind'] == 'tex']
                    self.assertEqual(len(textures), 1)
                    self.assertIn('depth2d<float>', textures[0]['msl_type'])
                    if METAL:
                        result = subprocess.run(
                            [str(METAL), str(self.output / 'original-exit_b.metal'),
                             'float', expected, str(self.output / 'original-exit_b.json')],
                            capture_output=True, text=True, timeout=60)
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_immediate_gather_offsets_are_signed_and_select_the_correct_texels(self):
        for packed, expected in ((0x013f, '3f200000'), (0x3f01, '3e800000')):
            for compare in (False, True):
                with self.subTest(packed=packed, compare=compare):
                    gather = (0xc800000000000000 | (1 << 54) | (7 << 51) |
                              (8 << 36) | (15 << 31) | (2 << 28) | (6 << 20) |
                              (7 << 16) | (4 << 8) | ((1 << 50) if compare else 0))
                    code = graphics([mov32(0x3f000000) | 4, mov32(0x3f000000) | 5,
                                     mov32(packed) | 6, mov32(0x3f000000) | 7,
                                     gather, EXIT], color=True)
                    struct.pack_into('<I', code, 0x48, 15)
                    self.code.write_bytes(code)
                    reply = self.send(dict(self.request, stage='ps', tex={'0': '2d_depth'}))[0]
                    self.assertTrue(reply['ok'], reply)
                    spirv = self.output / 'original-exit.spv'
                    validated = subprocess.run(
                        ['spirv-val', '--target-env', 'vulkan1.1', str(spirv)],
                        capture_output=True, text=True, timeout=30)
                    self.assertEqual(validated.returncode, 0, validated.stderr)
                    disassembled = subprocess.run(
                        ['spirv-dis', '--raw-id', str(spirv)],
                        capture_output=True, text=True, timeout=30)
                    self.assertEqual(disassembled.returncode, 0, disassembled.stderr)
                    definitions = dict(re.findall(
                        r'(%\d+) = (\S+.*)', disassembled.stdout))
                    offsets = re.findall(r'\bConstOffset (%\d+)', disassembled.stdout)
                    self.assertEqual(len(offsets), 1, disassembled.stdout)
                    vector = definitions[offsets[0]].split()[1]
                    opcode, scalar, count = definitions[vector].split()
                    self.assertEqual((opcode, count), ('OpTypeVector', '2'))
                    self.assertEqual(definitions[scalar], 'OpTypeInt 32 1')
                    if METAL:
                        value = expected
                        if compare:
                            value = '3f800000' if packed == 0x013f else '00000000'
                        checked = subprocess.run(
                            [str(METAL), str(self.output / 'original-exit_b.metal'),
                             'float', ','.join([value] * 4),
                             str(self.output / 'original-exit_b.json')],
                            capture_output=True, text=True, timeout=60)
                        self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_array_texture_fetch_offsets_preserve_coordinates_and_layer(self):
        fetch = (0xdc00000000000000 | (7 << 51) | (8 << 36) | (1 << 35) |
                 (15 << 31) | (3 << 28) | (7 << 20) | (7 << 16) | (4 << 8))
        for layer in (0, 1):
            for x_offset, y_offset in ((0, 0), (1, 2), (-1, 1), (1, -1)):
                with self.subTest(layer=layer, x=x_offset, y=y_offset):
                    packed = (x_offset & 15) | ((y_offset & 15) << 4)
                    code = graphics([mov32(layer) | 4, mov32(1) | 5, mov32(1) | 6,
                                     mov32(packed) | 7, fetch, EXIT], color=True)
                    struct.pack_into('<I', code, 0x48, 15)
                    self.code.write_bytes(code)
                    reply = self.send(dict(self.request, stage='ps', tex={'0': '2darray'}))[0]
                    self.assertTrue(reply['ok'], reply)
                    validated = subprocess.run(
                        ['spirv-val', '--target-env', 'vulkan1.1',
                         str(self.output / 'original-exit.spv')],
                        capture_output=True, text=True, timeout=30)
                    self.assertEqual(validated.returncode, 0, validated.stderr)
                    if METAL:
                        red = 1 + (1 + x_offset) + 4 * (1 + y_offset) + 16 * layer
                        bits = struct.unpack('<I', struct.pack('<f', red))[0]
                        checked = subprocess.run(
                            [str(METAL), str(self.output / 'original-exit_b.metal'),
                             'float', f'{bits:08x},3e800000,3f400000,3f800000',
                             str(self.output / 'original-exit_b.json')],
                            capture_output=True, text=True, timeout=60)
                        self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_depth_aspect_is_per_resource_and_does_not_add_comparison_dependencies(self):
        common = ((7 << 51) | (15 << 31) | (2 << 28) | (7 << 16) | (8 << 8))
        sample = 0xc080000000000000 | common | (255 << 20) | (8 << 36)
        for compare in (False, True):
            other = (0xc080000000000000 | common | 4 |
                     ((1 << 50) | (10 << 20) | (8 << 36) if compare
                      else (255 << 20) | (10 << 36)))
            adds = [0x5c58000000000000 | ((4+i) << 20) | (7 << 16) |
                    (i << 8) | i for i in range(4)]
            code = graphics([mov32(0x3f000000) | 8, mov32(0x3f000000) | 9,
                             mov32(0x3f000000) | 10, sample, other, *adds, EXIT], color=True)
            struct.pack_into('<I', code, 0x48, 15)
            self.code.write_bytes(code)
            request = dict(self.request, stage='ps', tex={'0': '2d_depth', '1': '2d'})
            for process in (False, True):
                with self.subTest(compare=compare, process=process):
                    reply = self.send(request, process=process)[0]
                    self.assertTrue(reply['ok'], reply)
                    reflection = self.reflection()
                    self.assertEqual(reflection['compare_slots'], [0] if compare else [])
                    textures = [b for b in reflection['bindings'] if b['kind'] == 'tex']
                    self.assertEqual(len(textures), 2)
                    self.assertEqual(sum('depth2d' in b['msl_type'] for b in textures),
                                     2 if compare else 1)
                    if METAL:
                        expected = ('3f700000,3f000000,3f000000,40000000' if compare
                                    else '3f700000,3e800000,3f400000,40000000')
                        result = subprocess.run(
                            [str(METAL), str(self.output / 'original-exit_b.metal'),
                             'float', expected, str(self.output / 'original-exit_b.json')],
                            capture_output=True, text=True, timeout=60)
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_register_derived_ptp_gather_uses_each_offsets_corresponding_neighbor(self):
        offsets = (0xef90000000000000 | (5 << 48) | (3 << 36) |
                   (16 << 20) | (7 << 16) | (255 << 8) | 6)
        gather = (0xc800000000000000 | (2 << 54) | (7 << 51) |
                  (8 << 36) | (15 << 31) | (2 << 28) | (6 << 20) |
                  (7 << 16) | (4 << 8))
        for texture, compare, component, expected in (
                ('2d_depth', False, 0, '3f400000,3f200000,3f400000,3e000000'),
                ('2d_depth', True, 0, '3f800000,3f800000,3f800000,00000000'),
                ('2d', False, 1, '3e800000,3e800000,3e800000,3e800000')):
            code = graphics([mov32(0x3f000000) | 4, mov32(0x3f000000) | 5,
                             mov32(0x3f000000) | 8, offsets,
                             gather | (component << 56) | ((1 << 50) if compare else 0),
                             EXIT], color=True)
            struct.pack_into('<I', code, 0x48, 15)
            self.code.write_bytes(code)
            request = dict(self.request, stage='ps', tex={'0': texture})
            for process in (False, True):
                with self.subTest(texture=texture, compare=compare, process=process):
                    reply = self.send(request, process=process)[0]
                    self.assertTrue(reply['ok'], reply)
                    if METAL:
                        result = subprocess.run(
                            [str(METAL), str(self.output / 'original-exit_b.metal'),
                             'float', expected, str(self.output / 'original-exit_b.json')],
                            capture_output=True, text=True, timeout=60)
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_driver_reads_validate_accessed_bytes_not_entire_vec4(self):
        for offset, expected in ((0x168, True), (0x164, False)):
            with self.subTest(offset=offset):
                ldc = (0xef90000000000000 | (4 << 48) | (2 << 36) |
                       (offset << 20) | (7 << 16) | (255 << 8))
                self.code.write_bytes(graphics([ldc, EXIT], color=True))
                reply = self.send(dict(self.request, stage='ps'))[0]
                self.assertEqual(reply['ok'], expected, reply)
                if expected:
                    self.assertEqual(self.reflection()['drv_reads']['2']['vec4_offsets'], [0x160])
                else:
                    self.assertIn('unfillable driver data', reply['error'])

    def test_process_mode_compiles_identical_msl(self):
        self.assertTrue(self.send(self.request)[0]['ok'])
        expected = (self.output / 'original-exit_b.metal').read_bytes()
        self.assertTrue(self.send(self.request, process=True)[0]['ok'])
        self.assertEqual((self.output / 'original-exit_b.metal').read_bytes(), expected)

    def test_process_failure_cannot_leak_stdout_or_publish(self):
        script = self.root / 'broken cross'
        script.write_text('#!/bin/sh\nprintf "not JSON\\n"\nexit 7\n')
        script.chmod(0o700)
        completed = self.run_service(json.dumps(self.request) + '\n',
                                     process=True, cross=script)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('not JSON', completed.stderr)
        self.assertFalse(json.loads(completed.stdout)['ok'])
        self.assertFalse((self.output / 'original-exit_b.json').exists())
        self.assertEqual(list(self.output.glob('.yuzu-request-*')), [])

    def test_process_output_still_requires_matching_resource_bindings(self):
        ldc = 0xef90000000000000 | (4 << 48) | (3 << 36) | (7 << 16) | (255 << 8)
        self.code.write_bytes(graphics([ldc, EXIT], color=True))
        request = dict(self.request, stage='ps')
        self.assertTrue(self.send(request)[0]['ok'])
        binding = self.reflection()['bindings'][0]
        source = (self.output/'original-exit_b.metal').read_text()
        source = source.replace(binding['msl_name'], 'incorrect_resource')
        script = self.root/'incorrect cross'
        script.write_text('#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n'
                          'Path(sys.argv[sys.argv.index("--output")+1]).write_text('
                          + repr(source) + ')\n')
        script.chmod(0o700)
        reply = self.send(request, process=True, cross=script)[0]
        self.assertFalse(reply['ok'])
        self.assertIn('resource binding is absent', reply['error'])
        self.assertFalse((self.output/'original-exit_b.json').exists())

    def test_publication_failure_does_not_commit_partial_result(self):
        self.output.mkdir()
        (self.output / 'original-exit_b.metal').mkdir()
        self.assertFalse(self.send(self.request)[0]['ok'])
        self.assertFalse((self.output / 'original-exit_b.json').exists())
        self.assertEqual(list(self.output.glob('.yuzu-request-*')), [])

    def test_invalid_cli_has_no_protocol_reply(self):
        for extra in (('--unknown', 'value'), ('--debug-mode', '2'), ('--compile-mode', 'quick')):
            completed = self.run_service('', extra=extra)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertEqual(completed.stdout, '')
        completed = self.run_service('', compile_mode=None, mode_env='quick')
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(completed.stdout, '')

    def test_typed_fragment_output_preserves_register_bits(self):
        for kind, bits in (('float', 0x3f000000), ('uint', 0x3f000000),
                           ('sint', 0xbf000000)):
            with self.subTest(kind=kind):
                self.code.write_bytes(graphics([mov32(bits), EXIT], color=True))
                request = dict(self.request, stage='ps', fout={'0': kind},
                               early_fragment_tests=True)
                self.assertTrue(self.send(request)[0]['ok'])
                reflection = self.reflection()
                self.assertEqual(reflection['deps']['fout'], {'0': kind})
                metal = self.output / 'original-exit_b.metal'
                source = metal.read_text()
                self.assertIn('[[color(0)]]', source)
                self.assertIn({'sint': 'int4', 'uint': 'uint4', 'float': 'float4'}[kind],
                              source)
                data = (self.output / 'original-exit.spv').read_bytes()
                words = struct.unpack('<' + 'I' * (len(data) // 4), data)
                instructions, offset = [], 5
                while offset < len(words):
                    count, opcode = words[offset] >> 16, words[offset] & 0xffff
                    self.assertGreater(count, 0)
                    instructions.append((opcode, words[offset+1:offset+count]))
                    offset += count
                self.assertTrue(any(op == 16 and args[1] == 9
                                    for op, args in instructions))  # EarlyFragmentTests.
                if kind != 'float':
                    types = {args[0] for op, args in instructions
                             if op == 21 and args[1:] == (32, int(kind == 'sint'))}
                    self.assertTrue(any(op == 124 and args[0] in types
                                        for op, args in instructions))  # OpBitcast.
                if METAL:
                    result = subprocess.run(
                        [str(METAL), str(metal), kind, f'{bits:x}'],
                        text=True, capture_output=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
