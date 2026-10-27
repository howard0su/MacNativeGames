# Shader translator: boundaries and service contract

**Active TOTK/shared-BOTW backend:** `tools/totk_xlate` compiles the copied
Yuzu Maxwell frontend/IR/SPIR-V backend and links SPIRV-Cross. App builds compile
the standalone `totk_xlate` executable from source. The application continues to
call it over JSON pipes, with no compiler linked into the engine. New caches
use `totk-yuzu-cpp-v6`. The retired C# project, patches, and archived
NativeAOT binary remain available in Git history.
The C++ service accepts `--compile-mode full|online-fast` (default `full`);
offline producers force `full`, and online misses also default to `full`.
`TOTK_SHADER_COMPILE_MODE` is the environment equivalent, overridden by the CLI.
Fast mode omits optional compiler verification, not correctness-critical
lowering, input/resource bounds, or the generated GPU loop watchdog.
Both modes retain the shader optimization pipeline. `online-fast` remains an
explicit developer override, not the runtime default; it is not an unoptimized
shader build.

The shader translator is a compiler service, not an emulator process. It reads
Switch Maxwell GPU machine code and produces Metal Shading Language (MSL) and
resource-binding reflection. The renderer compiles/loads Metal libraries and
creates render or compute pipelines separately.

```text
NVN shader bytes + relevant GPU state
    -> totk_xlate --service (JSON requests over pipes)
        -> Yuzu Maxwell frontend + IR + SPIR-V backend
        -> SPIRV-Cross -> MSL
        -> resource ABI adaptation + reflection
    -> renderer -> Metal library -> pipeline -> Apple GPU
```

Yuzu and SPIRV-Cross are independent projects. `compiler.cpp` translates Maxwell
to SPIR-V using Yuzu; `resource_abi.cpp` passes that SPIR-V to SPIRV-Cross and maps
its actual Metal argument IDs and final names into native reflection.
SPIRV-Cross does not decode Maxwell instructions.

## How the service becomes an executable

`games/totk/tools/totk_xlate/publish_native.sh` builds the standalone C++20
project using the copied compiler in `vendor/yuzu` and SPIRV-Cross at the pin in
`build_app.py`, then runs its native tests:

```sh
bash games/totk/tools/totk_xlate/publish_native.sh
```

The generated ARM64 executable, digest, and backend/mode manifest live in the
ignored `games/totk/build/shader-service/totk_xlate{,.sha256,.json}` directory,
not in version control.
`make -C games/totk shader-service` builds the CLI;
`make -C games/totk verify-shaders` builds and runs their native tests.
The native tests additionally require `spirv-val` from `brew install spirv-tools`.
Normal app builds invoke the same incremental source build in their work
folder, then verify its backend, cache chain, supported modes, and SHA-256.
Building requires CMake, a C++20 compiler, fmt, Boost headers, and pinned
SPIRV-Cross (fetched automatically if absent). No .NET is needed, and the
finished service links only system libc++/libSystem.

**Compiling the translator ahead of time is different from translating game
shaders ahead of time.** Build-time preparation and gameplay share the production
service protocol and transactional publisher.

| Caller | When it requests translation | Input state | Output destination |
| --- | --- | --- | --- |
| `games/totk/scripts/prepare_shader_seed.py` | A build translation-cache miss | Default, inferred, depth, or shared-list variants | Shader seed staging directory |
| `totk_shader_rt.m` | A draw/dispatch needs a variant absent from the seed/cache | Actual binding state | Player's translation cache |

The packager places the executable at `build/shader-service/totk_xlate`
inside macOS app resources. Each runtime worker retains its own service process.
The engine writes `.code` inputs, sends JSON requests and reads reflection after
successful replies. Shader artifacts are published with reflection last.
Prepared shaders and cache hits still avoid translation.

Device builds continue to use prepared shaders and recorded miss requests;
they do not link or launch the compiler. `TOTK_SHADER_NO_TRANSLATE=1` retains
the same offline-only behavior on macOS. Pretranslated MSL can still require
Metal compilation and pipeline creation when loaded.

`totk_xlate --identity` identifies the source/configuration of the service build.

## Process interface

```sh
totk_xlate --service \
  --spirv-cross /path/to/spirv-cross \
  --out /path/to/output \
  --chain translation-chain-name \
  --compile-mode full \
  --debug-mode 1
```

`--debug-mode` is an optional 0/1 compatibility control, defaulting to 1; it also
accepts `TOTK_SHADER_DEBUGMODE` but does not select Yuzu optimization passes.
`--compile-mode` controls optional verification as described above.
The path, output, and chain options are required even though
SPIRV-Cross normally runs in process. `TOTK_XLATE_SPIRV_CROSS=process` selects the
external executable for comparison in the desktop CLI only; this setting never
changes the engine's subprocess interface.

Each nonblank stdin line is one JSON request. Each request gets one JSON stdout
line, flushed immediately. Outputs are files, not the reply body:

```json
{"ok": true}
```

```json
{"ok": false, "error": "..."}
```

A request error is reported and the service continues reading requests. EOF
ends the process. Invalid CLI arguments produce exit code 2. Stdout must contain
only protocol replies; diagnostics belong on stderr. The native host currently
treats a failed translation, unexpected EOF, or missing reflection as fatal.

### Request

```json
{
  "variant": "<shader-key>-<binding-state-hash>",
  "stage": "ps",
  "code": "/absolute/path/to/shader.code",
  "c1": "",
  "tex": {"0": "2d_depth"},
  "img": {},
  "attr": {},
  "fout": {"0": "float"},
  "depth_minus_w": true,
  "early_fragment_tests": true
}
```

| Field | Meaning |
| --- | --- |
| `variant` | Output stem chosen by the caller; runtime combines the shader key with a truncated SHA-256 of binding-state answers. |
| `stage` | `vs`, `ps`, or `cs`. Geometry and tessellation are rejected. |
| `code` | Shader file path. Graphics input starts with the 0x50-byte SPH followed by instructions, with the preceding 0x30-byte NVN header removed. Compute keeps the original code region, including any prefix before `entry_point`. |
| `c1` | Hex bytes of constant buffer 1; used to resolve indirect branches and retained for runtime binding. Defaults to empty. |
| `c1_buffer_size` | Optional caller-declared accessible c1 bank size, including zero padding; must contain `c1` and be at most 64 KiB. NVN producers declare their existing 64-KiB bank. Dynamic reads not statically bounded to the supplied data get explicit bounds checks and zero results outside the declared bank; the actual load is clamped so it cannot access outside it. Without this field, legacy strict bounds checks remain unchanged. |
| `tex` | Slot-to-kind map: `1d`, `1darray`, `2d`, `2darray`, `2dms`, `2dmsarray`, `3d`, `cube`, `cubearray`, `buffer`; supported depth kinds have `_depth` appended. |
| `img` | Slot-to-object map with `type` and `fmt`; format names are Ryujinx `TextureFormat` names. |
| `attr`, `fout` | Vertex location / color target to `float`, `sint`, or `uint`. Unanswered types default to float; supplied types affect code and dependencies. |
| `local` | Compute workgroup dimensions `[x,y,z]`; required for compute requests. |
| `entry_point` | Compute program byte offset within `code`; defaults to zero for legacy requests. Must be 8-byte aligned and leave complete instructions. Offline and online NVN producers read it from control+0x708. |
| `local_memory`, `shared_memory` | Optional compute allocation sizes in bytes; required when the shader actually accesses the respective memory. |
| `depth_minus_w` | Convert vertex clip z from `-w..w` to Metal's `0..w`; defaults to false. Explicit fragment depth is already in window space and is not converted. |
| `early_fragment_tests` | Force early fragment tests; defaults to false. |

The state dictionaries default to empty. Existing fallback type behavior is not
proof that every shader can be translated without state. In particular,
translation must preserve all state dependencies that affect the result.

The NVN producers preserve compute workgroup dimensions at control+0x748,
shared bytes at +0x754, and positive local bytes at +0x758, including explicit
zeros. Nonzero negative-local allocation (+0x75c) is unsupported and rejected,
not guessed or combined with positive offsets. These fields match the
[public NVN control layout](https://github.com/nvnprogram/uam-nvn/blob/f4470ae120392f76ecfdce74390dbb13a461a9df/source/nvn_control.h).

Depth aspect is independent of comparison sampling. Bound `_depth` resources
use native Metal depth types, but only comparison instructions populate
`compare_slots`. Noncomparison sample/read results are `(depth,0,0,1)`;
gathers preserve neighboring values and register-derived per-texel offsets.
Process-mode diagnostics must use the C++ build's `totk_spirv_cross_driver`,
which shares its NVN type adapter. A stock converter returning incompatible
resource types is rejected instead of publishing misleading reflection.

### Output

Success produces `<variant>_b.metal` and `<variant>_b.json`. The current
implementation also leaves intermediate `<variant>.spv` and
`<variant>.res.json` files.

Reflection includes `key`, `route: "B"`, `chain`, `backend: "yuzu-cpp"`, `compile_mode`, `stage`, `entry`,
`msl_entry_kind`, `threadgroup_size`, `c1`, `notes`, `bindings`, `vertex_inputs`,
`raster`, `drv_reads`, `drv_unfillable`, `deps`, and
`compare_slots`.

FP64 lowering runs before SSA in a single frontend invocation. The retired
`softfloat64` retry diagnostic is no longer emitted; older cached reflection
containing it remains compatible.

Bindings identify actual Metal argument IDs (`arg_id`), names/types, and NVN
resources. Supported kinds include `cbuf`, `sbuf`, `ubo_ptr`, `sbuf_ubo`, `tex`,
`tbuf`, `img`, `samp`, `samp_tex`, `support`, `c1`, and `drvcb`. Storage writes
must be reflected accurately because they affect synchronization and reuse.
Resources are in descriptor set 0 / Metal argument buffer 0.

`drv_reads` records driver-buffer accesses; unsupported accesses appear in
`drv_unfillable` and are rejected by the native reflection loader. `deps`
records binding-state answers needed to reuse a variant. Texture dependency
`"*"` matches non-depth kinds; `"*_depth"` matches depth kinds. An unanswered
vertex input can match an existing variant. See [shader variants](shader-variants.md).

The translation `chain` is a cache compatibility identity, not a negotiated
protocol version. Changing compiler output under an existing variant name
requires a new identity and compatible seed/library/pipeline invalidation.

## Internal compiler interface

`compiler.cpp` implements `Shader::Environment` using bounded shader/c1 reads,
texture/image provenance, explicit compute allocations, and graphics state.
The translation path is:

```text
Flow::CFG -> Maxwell::TranslateProgram -> Backend::SPIRV::EmitSPIRV
```

The target profile is configured for Metal/SPIRV-Cross, not Yuzu's emulator or
Vulkan renderer. The frontend retains required SSA, constant propagation, FP64
lowering, and resource analysis. Unsupported state fails explicitly rather than
falling back to the retired translator.

The removed C# batch, `--serve`, and `--disasm` development modes are not native
service interfaces. Offline preparation uses `prepare_shader_seed.py`.

## Sources and replacement work

- `games/totk/tools/totk_xlate/{compiler.cpp,service.cpp,resource_abi.cpp}`
- `games/totk/tools/totk_xlate/{publish_native.sh,spirv_cross_inproc.cpp}`
- `games/totk/scripts/prepare_shader_seed.py`
- `games/totk/third_party/d2r_gfx/gfx/nvn/{totk_shader_rt.m,nvn_refl.m,nvn_mtl.m}`
- `build_app.py`

Geometry/tessellation stages, unsupported subgroup operations and untracked
global-memory pointers are rejected. FP64 arithmetic uses FP32 lowering, not
full-precision software FP64. See `vendor/yuzu/README.md` for source provenance,
local compiler adaptations and retained compatibility limitations. Synthetic
compiler tests do not establish all-shader or gameplay equivalence.
