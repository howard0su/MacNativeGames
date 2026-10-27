// Reuse SPIRV-Cross's CLI defaults for the probe and process-mode test driver.
// The production service calls CompilerMSL directly with the same pinned options.
#include "spirv_cpp.hpp"
#include "spirv_cross_util.hpp"
#include "spirv_glsl.hpp"
#include "spirv_hlsl.hpp"
#include "spirv_msl.hpp"
#include "spirv_parser.hpp"
#include "spirv_reflect.hpp"
#include "nvn_msl.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{
// main.cpp calls exit() on bad arguments or a missing entry point; inside the service that would end it, so it throws.
struct CliExit : std::runtime_error
{
	explicit CliExit(int code) : std::runtime_error("spirv-cross stopped (exit " + std::to_string(code) + ")") {}
};
[[noreturn]] void cli_exit(int code) { throw CliExit(code); }
} // namespace

#define exit(code) cli_exit(code)
#define main spirv_cross_cli_main
// Use the same metadata-driven NVN backend in the real process-mode driver.
#define CompilerMSL NvnCompilerMSL
#include "main.cpp"
#undef CompilerMSL
#undef main
#undef exit

// spirv-cross --msl --msl-version 20300 --msl-argument-buffers --msl-texture-buffer-native --msl-pad-fragment-output
// on the SPIR-V words. Returns 0 and the MSL in *out, or 1 and the message the program printed in *err;
// free either with totk_spirv_cross_free.
extern "C" int totk_spirv_cross_msl(const uint32_t *words, size_t count, char **out, char **err)
{
	*out = nullptr;
	*err = nullptr;
	try
	{
		CLIArguments args;
		args.msl = true;
		args.msl_version = 20300;
		args.set_msl_version = true;
		args.msl_argument_buffers = true;
		args.msl_texture_buffer_native = true;
		args.msl_pad_fragment_output = true;
		std::string msl = compile_iteration(args, std::vector<uint32_t>(words, words + count));
		*out = strdup(msl.c_str()); // the program wrote it with fprintf("%s"): up to the first NUL, as here
		return 0;
	}
	catch (const std::exception &e)
	{
		*err = strdup((std::string("SPIRV-Cross threw an exception: ") + e.what()).c_str());
		return 1;
	}
}

extern "C" void totk_spirv_cross_free(char *p)
{
	free(p);
}
