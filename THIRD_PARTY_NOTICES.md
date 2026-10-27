# Third-party code

Code in this repository that comes from other projects keeps its license; the license file sits next to the code.

| Code | From | License | File |
|---|---|---|---|
| `games/totk/third_party/d2r` (NSO rewriter, verification tools) | [d2r-mobile](https://github.com/jhaizhou-ops/d2r-mobile), adapted | MIT | `games/totk/third_party/d2r/LICENSE` |
| `games/totk/third_party/d2r_gfx` (NVN on Metal, texture tools, host headers) | d2r-mobile, extended for this game | MIT | `games/totk/third_party/d2r_gfx/LICENSE` |
| texture layout sizes in `games/totk/third_party/d2r_gfx/gfx/nvn/nvn_tex.c` | deko3d | zlib | `games/totk/third_party/d2r_gfx/LICENSE.deko3d` |
| `games/totk/third_party/d2r_gfx/third_party/bcdec` | [bcdec](https://github.com/iOrange/bcdec) 0.985 | MIT or public domain | `games/totk/third_party/d2r_gfx/third_party/bcdec/LICENSE` |
| `games/totk/tools/totk_xlate/spirv_cross_inproc.cpp` (C++ SPIRV-Cross CLI bridge for probes) | d2r-mobile, adapted and retained independently of the removed C# translator | MIT | `games/totk/tools/totk_xlate/LICENSE` |
| `vendor/ryujinx` (Ryujinx.Graphics.Shader, Ryujinx.Common, Spv.Generator) | Ryujinx, as pinned by d2r-mobile; retained for legacy game tooling and reference | MIT | `vendor/ryujinx/LICENSE.txt` |
| `vendor/yuzu/src` (shader-only C++ compiler import) | supplied Yuzu checkout; provenance in `vendor/yuzu/SOURCE.json` | GPL-2.0-or-later; individual utility headers retain their notices | `vendor/yuzu/LICENSES/`, source headers |
| Eden shader-compiler fixes in `vendor/yuzu/src/shader_recompiler` | [Eden](https://git.eden-emu.dev/eden-emu/eden); selected commits in `vendor/yuzu/README.md` | GPL-3.0-or-later additions; original Yuzu notices retained | `vendor/yuzu/LICENSES/GPL-3.0-or-later.txt`, source headers |
| `vendor/yuzu/externals/sirit` | Sirit and SPIR-V headers bundled with that checkout | BSD-3-Clause (Sirit), MIT-style Khronos license (SPIR-V headers) | `vendor/yuzu/externals/sirit/LICENSE.txt`, `vendor/yuzu/externals/sirit/externals/SPIRV-Headers/LICENSE` |
| `games/totk/build/shader-service/totk_xlate` (generated standalone compiler; not tracked in Git) | `games/totk/tools/totk_xlate`, copied Yuzu/Sirit sources, pinned SPIRV-Cross, fmt and Boost headers; no .NET | GPL-2.0-or-later compiler sources, GPL-3.0-or-later adaptations and dependency notices; distribution obligations require review | `vendor/yuzu/LICENSES/`, source/dependency headers |
| `games/{by3,odyssey}/prebuilt/macos-arm64/totk_xlate` (unchanged legacy NativeAOT binaries) | Ryujinx + former C# service; contains .NET runtime libraries compiled in by NativeAOT | MIT (Ryujinx, d2r-mobile, .NET) | `vendor/ryujinx/LICENSE.txt`, `games/totk/tools/totk_xlate/LICENSE` |
| in `games/totk/runtime/threads.c`: SignalToAddress that wakes all waiters subtracts 2, SystemEvent bindings that end with their write handle, signals that wake only the waiters of the signalled object (checked by `games/totk/tests/system_event_bind`) | [nxnative](https://github.com/seedds/nxnative) commits `d47f899`, `49ee623`, `df7badd`, rewritten for this runtime (via PR #159) | MIT | `games/totk/third_party/licenses/nxnative.txt` |
| spin-wait search and copy in `games/totk/third_party/d2r/patch_nso.py` ("Spin waits") and its test `tests/test_spin_waits.py` | [nxnative](https://github.com/seedds/nxnative) commits `066f2af`, `6d34984`, adapted to this rewriter | MIT | `games/totk/third_party/licenses/nxnative.txt` |
| audio effects in `games/totk/runtime/audio_reverb.c`, `games/totk/runtime/audio_effects.c` (delay, reverb, compressor) and the mixer | ported from Ryujinx.Audio | MIT | `games/totk/third_party/licenses/Ryujinx-Audio.txt` |

Fetched or linked at build time (not in this repository):

| Code | Use | License |
|---|---|---|
| [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross) at `aa217ae` | SPIR-V to Metal Shading Language | Apache-2.0 |
| [hactool](https://github.com/SciresM/hactool) at `1d64a83` (with explicit private title-key-file hook), including its bundled mbedTLS | Base + update BKTR merge at build time only | ISC (hactool), Apache-2.0 (mbedTLS); see fetched source LICENSE files |
| libopus (Homebrew), linked statically | Opus audio | BSD-3-Clause |
| libvpx (Homebrew), linked statically | VP9 movies | BSD-3-Clause |
| fmt (developer installation), included header-only | Yuzu compiler diagnostics and formatting | MIT |
| Boost headers, including Boost.JSON (developer installation) | Yuzu compiler containers and native service JSON | Boost Software License 1.0 |
