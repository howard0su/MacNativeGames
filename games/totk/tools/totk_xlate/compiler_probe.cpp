#include <fstream>
#include <iostream>
#include <stdexcept>

#include "compiler.h"

#ifdef TOTK_PROBE_MSL
extern "C" int totk_spirv_cross_msl(const uint32_t*, size_t, char**, char**);
extern "C" void totk_spirv_cross_free(char*);
#endif

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "usage: totk_yuzu_probe vs|ps|cs shader.code output.spv [output.metal]\n";
        return 2;
    }
    try {
#ifndef TOTK_PROBE_MSL
        if (argc == 5) {
            throw std::runtime_error("configure SPIRV_CROSS_SRC to enable MSL output");
        }
#endif
        Totk::CompileRequest request;
        request.stage = argv[1];
        request.code = Totk::ReadShader(argv[2]);
        const auto result = Totk::Compile(request);
        std::ofstream output(argv[3], std::ios::binary);
        output.write(reinterpret_cast<const char*>(result.words.data()),
                     static_cast<std::streamsize>(result.words.size() * sizeof(u32)));
        output.close();
        if (!output) {
            throw std::runtime_error("cannot write SPIR-V output");
        }
#ifdef TOTK_PROBE_MSL
        if (argc == 5) {
            char* msl = nullptr;
            char* error = nullptr;
            const int rc = totk_spirv_cross_msl(result.words.data(), result.words.size(), &msl, &error);
            const std::string message = error ? error : "SPIRV-Cross returned no source";
            const std::string source = msl ? msl : "";
            totk_spirv_cross_free(msl);
            totk_spirv_cross_free(error);
            if (rc != 0 || source.empty()) {
                throw std::runtime_error(message);
            }
            std::ofstream metal(argv[4], std::ios::binary);
            metal << source;
            metal.close();
            if (!metal) {
                throw std::runtime_error("cannot write MSL output");
            }
        }
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
