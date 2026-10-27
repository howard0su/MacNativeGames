"""The active service must be the verified C++ backend, not an archived C# build."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import build_app


class ShaderBackendTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'darwin' and shutil.which('xcrun'), 'requires Foundation')
    def test_cpp_constant_buffer_names_do_not_bypass_the_shared_bank_guard(self):
        source = (build_app.GAME/'third_party/d2r_gfx/gfx/nvn/nvn_mtl.m').read_text()
        guard = source[source.index('static int c1_reads_bounded('):source.index('static void c1_store(')]
        fixture = '''#import <Foundation/Foundation.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
''' + guard + '''
int main(void) {
    @autoreleasepool {
        assert(c1_reads_bounded(@"uint x = (*args.cp_c1)._m0[31u].x;", 256));
        assert(c1_reads_bounded(@"uint x = (*args.cp_c1)._RESERVED_IDENTIFIER_FIXUP_m0[31u].x;", 256));
        assert(!c1_reads_bounded(@"uint x = (*args.cp_c1)._RESERVED_IDENTIFIER_FIXUP_m0[32u].x;", 256));
        assert(!c1_reads_bounded(@"uint x = (*args.cp_c1)._RESERVED_IDENTIFIER_FIXUP_m0[index].x;", 256));
        assert(!c1_reads_bounded(@"uint x = (*args.cp_c1).unknown[0].x;", 256));
    }
}
'''
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'guard.m').write_text(fixture)
            subprocess.run(['xcrun', 'clang', '-fobjc-arc', '-Wall', '-Wextra', '-Werror',
                            root/'guard.m', '-framework', 'Foundation', '-o', root/'guard'],
                           check=True, capture_output=True, text=True)
            subprocess.run([root/'guard'], check=True, capture_output=True, text=True)

    def test_accepts_cpp_and_rejects_archived_backend_stale_chain_and_changed_executable(self):
        with tempfile.TemporaryDirectory() as folder:
            tree = Path(folder)
            service = tree/'build/shader-service/totk_xlate'
            service.parent.mkdir(parents=True)
            service.write_bytes(b'original synthetic service')
            digest = hashlib.sha256(service.read_bytes()).hexdigest()
            service.with_suffix('.sha256').write_text(digest+'\n')
            metadata = dict(backend='yuzu-cpp', chain=build_app.SHADER_CHAIN, sha256=digest,
                            compile_modes=['full', 'online-fast'])
            service.with_suffix('.json').write_text(json.dumps(metadata))
            self.assertEqual(build_app.shader_translator(tree), service)
            for change in (dict(backend='ryujinx'), dict(chain='totk-native-v8-unverified'),
                           dict(chain='totk-yuzu-cpp-v1'), dict(chain='totk-yuzu-cpp-v2'),
                           dict(chain='totk-yuzu-cpp-v3'),
                           dict(chain='totk-yuzu-cpp-v4'),
                           dict(compile_modes=['full']), dict(compile_modes=None),
                           dict(sha256='0'*64)):
                with self.subTest(change=change):
                    service.with_suffix('.json').write_text(json.dumps(dict(metadata, **change)))
                    with self.assertRaisesRegex(build_app.BuildError, 'invalid Yuzu C\\+\\+ service'):
                        build_app.shader_translator(tree)
            service.with_suffix('.json').write_text(json.dumps(metadata))
            service.write_bytes(b'changed executable')
            with self.assertRaisesRegex(build_app.BuildError, 'SHA-256 differs'):
                build_app.shader_translator(tree)

    def test_source_build_installs_and_refreshes_generated_service(self):
        with tempfile.TemporaryDirectory() as folder:
            tree = Path(folder)
            (tree/'tooling/spirv-cross').mkdir(parents=True)
            native = tree/'build/totk_xlate/totk_xlate'
            payload = b'original synthetic native compiler'
            def run(command, log):
                if '--build' in command:
                    native.parent.mkdir(parents=True, exist_ok=True)
                    native.write_bytes(payload)
                    native.chmod(0o755)
                    (native.parent/'compiler-identity.txt').write_text('a'*64+'\n')
            with patch.object(build_app, 'run', side_effect=run) as commands:
                service = build_app.build_shader_service(tree, tree/'logs')
                self.assertEqual(service.read_bytes(), payload)
                self.assertEqual(commands.call_args_list[0].args[0][2],
                                 build_app.GAME/'tools/totk_xlate')
                self.assertIn('totk_xlate', commands.call_args_list[1].args[0])
                build_app.build_shader_service(tree, tree/'logs')
                payload = b'changed synthetic native compiler'
                self.assertEqual(build_app.build_shader_service(tree, tree/'logs'), service)
            self.assertEqual(service.read_bytes(), payload)
            metadata = json.loads(service.with_suffix('.json').read_text())
            self.assertEqual(metadata['sha256'], hashlib.sha256(payload).hexdigest())
            self.assertEqual(metadata['compiler_identity'], 'a'*64)
            self.assertEqual(build_app.shader_translator(tree), service)
            original = service.read_bytes()
            with patch.object(build_app, 'run', side_effect=build_app.BuildError('compile failed')):
                with self.assertRaisesRegex(build_app.BuildError, 'compile failed'):
                    build_app.build_shader_service(tree, tree/'logs')
            self.assertEqual(service.read_bytes(), original)

    def test_both_games_build_service_before_seed_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            tree = Path(folder)
            cross = tree/'build/spirv-cross/spirv-cross'
            cross.parent.mkdir(parents=True)
            cross.touch()
            service = tree/'build/shader-service/totk_xlate'
            for game in ('totk', 'botw'):
                with self.subTest(game=game), patch.object(
                        build_app, 'build_shader_service', return_value=service) as build, \
                        patch.object(build_app, 'run') as run:
                    if game == 'totk':
                        build_app.step_shaders(tree, tree/'logs')
                    else:
                        build_app.step_shaders_botw(tree, tree/'botw', tree/'logs')
                    build.assert_called_once_with(tree, tree/'logs')
                    command = run.call_args.args[0]
                    self.assertEqual(command[command.index('--translator') + 1], service)


if __name__ == '__main__':
    unittest.main()
