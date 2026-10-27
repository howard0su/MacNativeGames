# Shader-only Yuzu source import

Imported from the user-supplied Yuzu checkout at revision
`7533ff093d76e2807ccb87bbc6fd17dab2447cc6`. `SOURCE.json` records each source
file's original raw SHA-256, HEAD blob, and comparison before/after CRLF
normalization. All selected inputs matched HEAD after line-ending normalization.

Only the Maxwell frontend, IR, optimization passes, SPIR-V backend, minimal
common header closure, Sirit, and unified SPIR-V headers are included. Emulator
engines, HLE, CPU, Qt, Vulkan renderer, and other shader backends are excluded.
The settings-dependent resolution rescaling pass is deliberately not imported.

Local adaptations:

- `emit_spirv_image.cpp`: signed constant texture-gather offsets, ported from
  Eden `025bc799f782be40544d2718bdcbb8f4d0d033d2`
  (https://git.eden-emu.dev/eden-emu/eden/commit/025bc799f782be40544d2718bdcbb8f4d0d033d2).
  Keep the local runtime PTP gather lowering. Eden additions retain their
  GPL-3.0-or-later notices; the original Yuzu import manifest is unchanged.
- `emit_spirv_image.cpp`: apply Array2D texel-fetch offsets rather than adding
  the coordinates to themselves, preserving the array layer. Ported from Eden
  `80bafc8fe876371c90d64fb9b7288e03d2ba8da8`
  (https://git.eden-emu.dev/eden-emu/eden/commit/80bafc8fe876371c90d64fb9b7288e03d2ba8da8).
- `ir_emitter.h/.cpp`, `opcodes.inc`, `emit_spirv_integer.cpp`,
  `emit_spirv_instructions.h`: width-aware integer absolute value for signed
  64-bit I2F sources, ported from Eden
  `0e6ea2d9d66bc7bd4019ab10324e75e73e96f517`
  (https://git.eden-emu.dev/eden-emu/eden/commit/0e6ea2d9d66bc7bd4019ab10324e75e73e96f517).
  Track `IAbs64` as int64 usage; only the active native-int64 SPIR-V path is
  ported, without the other backends or non-native-int64 lowering.
- `translate_program.cpp`: no global emulator settings/resolution scaling;
  verify IR by default, with an explicit host flag to omit invariant verification
  for online-fast compilation. Required lowering/optimization stays enabled.
  `host_translate_info.h` also adds an opt-in diagnostic flag for logging IR
  before resource lowering without changing generated code.
- `emit_spirv.cpp`: no global settings dependency; keep loop safety checks.
- `runtime_info.h`, `spirv_emit_context.cpp`, `emit_spirv_context_get_set.cpp`,
  `emit_spirv_special.cpp`: typed fragment color declarations and bit-preserving
  stores for signed/unsigned integer targets; reject integer alpha-test state.
- `environment.h`, `texture_pass.cpp`: metadata queries carry constant-buffer
  provenance rather than reading unavailable raw texture handles.
- `environment.h`, `load_store_local_shared.cpp`: require an actual compute
  local-memory allocation before translating local loads/stores; memory-free
  compute kernels do not need an invented allocation.
- `global_memory_to_storage_buffer_pass.cpp`: successful pointer tracking
  outside the biased driver table is informational, allowing supported user
  uniform pointers through the host ABI checks. Untracked pointers still warn
  and fail; unsupported pointer locations still fail the ABI mapping.
  Proven compute uniform-table loads are normalized to banked uniform reads
  before tracking pointers loaded from them. IR diagnostics can show the
  post-normalization program and the failing block. Shared/local pointer
  spills now have separate descriptor-provenance analysis, informed by
  Ryujinx's `GlobalToStorage` pass. It keeps the actual memory operations and
  pointer arithmetic, requires an initializing write that dominates each load,
  checks every potentially overlapping writer, and verifies both descriptor
  words. Shared-record stride proofs account for 32-bit address wraparound;
  local slots must be constant. Conflicting, partial, atomic, uninitialized,
  or unproven aliased writes still reject the translation. Memory-load
  addresses are not followed as if they were the loaded pointer's provenance.
  No zero-load/no-op-store or multi-target best-effort fallback is added.
- `spirv_emit_context.h/.cpp`, `emit_spirv_image.cpp`: keep actual depth image
  types independent of comparison opcodes; define noncomparison sample/fetch
  channels and depth gather components. PTP gathers use four corresponding
  offset footprints, including runtime offsets, instead of ignoring nonconstant
  offsets. Sparse PTP remains explicitly unsupported.
- `texture_mipmap_level.cpp`: Maxwell TMML returns unclamped then clamped LOD
  in signed 24.8 fixed point. Reverse the query components and multiply by 256
  before signed conversion, preserving negative and fractional LODs.
- `spirv_emit_context.h/.cpp`, `emit_spirv_image.cpp`: explicit storage-image
  formats determine signed/unsigned SPIR-V scalar and read/write types even when
  usage metadata is non-integer. Preserve register bits across signed/float
  image accesses, and declare required 1D/extended-format capabilities. This
  focused correction reproduces the integer-image changes in 19 recorded BOTW
  compute shaders; it does not establish a captured-gameplay visual improvement.
- `lower_fp64_to_fp32.cpp`: correct exponent rebiasing, nearest-even conversion,
  gradual underflow, signed zeros, special values, immediate operands and
  min/max/select lowering. Double arithmetic still uses the existing FP32
  fallback; this is not full-precision software FP64.
- Maxwell arithmetic: preserve extended carry/cumulative-zero and numeric
  floating-point flags; complete the implemented XMAD, VMAD and mixed-sign
  VMNMX cases, IMUL/IMAD operand forms, immediate float conversion/range reduction,
  predicate masks, and integer min/max, funnel-shift, compare and logical flags.
  Unproven modifiers remain explicit errors.
- `control_flow.cpp`: preserve conditional indirect-branch fallthrough and
  avoid requiring a branch table for a statically untaken branch.
- Tessellation patch loads: track and declare all four outer and two inner
  levels for native TES/TCS, including reading TCS output levels. Reject patch
  loads outside tessellation stages. This does not add a production Metal
  tessellation pipeline or expand the standalone service's accepted stages.
- Surface/image operations: scale typed byte addresses by the full pixel size;
  correct bound/bindless atomic size and reduction-opcode fields, byte addressing
  and array-layer extraction; translate supported CAS32 with retained side
  effects; implement bounded unsigned increment/decrement with CAS retry loops.
  Preserve signed typeless-image metadata and match atomic pointer/scalar types
  to the image declaration, with bit-preserving register results.
- Resource queries/declarations: declare sampled 1D, cube-array and multisample
  array capabilities from the actual resource type, normalize rectangle
  dimension queries, and reject out-of-range constant-buffer banks before
  indexing or shifting a mask.
- TXQ sample counts: translate the sample-count component using actual
  `Environment::ReadTextureSamples` state. Missing/invalid state and other
  texture-header components are errors, not guessed single-sample results.
  The production adapter does not yet supply that state.
- Existing unsupported-target compatibility fallbacks are preserved: typeless
  image reads/global loads can return zero, unsupported global writes can be
  discarded, and unsupported wide atomics retain their legacy zero/non-atomic
  paths. These specific paths log a nonfatal compatibility diagnostic instead
  of throwing through the standalone logger. This prioritizes continuing
  gameplay; it does not establish accurate rendering or atomicity. Other
  warnings and errors retain their existing behavior.

These correctness fixes are active by default, without a feature opt-in.
Ryujinx-informed adaptations retain their source notices and
`LICENSES/Ryujinx-MIT.txt`; `SOURCE.json` continues to describe the original import.
The unfinished descriptor-array, geometry-to-compute and fragment-interlock
expansions remain isolated and are not part of this main-branch port.

Synthetic FP64, instruction, branch, tessellation and scalar-resource regression
sources are registered in the standalone CMake project. The scalar resource
fixture excludes the isolated descriptor-array contracts. These tests do not
establish all-shader or gameplay equivalence.

The standalone target and throwing/stderr diagnostics are under
`games/totk/tools/totk_xlate`. Compiler declarations and third-party headers
retain their licenses. The repository's MIT license does not relicense these
files. Consult `LICENSES/`, `LICENSE.txt`, and each dependency's license.

The import powers the standalone translator; it is not linked into the engine.
Distribution obligations require review. See `docs/shader-translator.md`.
