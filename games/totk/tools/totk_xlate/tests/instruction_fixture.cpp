#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"

extern "C" int totk_spirv_cross_msl(const uint32_t*, size_t, char**, char**);
extern "C" void totk_spirv_cross_free(char*);

class InstructionEnvironment final : public Shader::Environment {
public:
    explicit InstructionEnvironment(const char* path, Shader::Stage shader_stage) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot read instruction fixture");
        code = std::vector<char>(std::istreambuf_iterator<char>(input), {});
        if (code.size() < sizeof(sph) || code.size() % sizeof(u64) != 0) {
            throw std::runtime_error("invalid instruction fixture size");
        }
        std::memcpy(&sph, code.data(), sizeof(sph));
        stage = shader_stage;
        // Restore the removed NVN prefix to align scheduling bundles.
        start_address = address_bias;
    }

    u32 EntryPoint() const { return start_address + sizeof(sph); }

    u64 ReadInstruction(u32 address) override {
        if (address < address_bias || address % sizeof(u64) != 0) {
            throw std::runtime_error("invalid instruction fixture address");
        }
        address -= address_bias;
        if (address > code.size() || code.size() - address < sizeof(u64)) {
            throw std::runtime_error("instruction fixture address out of bounds");
        }
        u64 value;
        std::memcpy(&value, code.data() + address, sizeof(value));
        return value;
    }
    u32 ReadCbufValue(u32, u32) override { throw std::runtime_error("unexpected cbuf read"); }
    Shader::TextureType ReadTextureType(const Shader::TextureBufferDescriptor&) override {
        throw std::runtime_error("unexpected texture type query");
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(const Shader::TextureBufferDescriptor&) override {
        throw std::runtime_error("unexpected texture format query");
    }
    bool IsTexturePixelFormatInteger(const Shader::TextureBufferDescriptor&) override {
        throw std::runtime_error("unexpected texture format query");
    }
    u32 ReadViewportTransformState() override { throw std::runtime_error("unexpected viewport query"); }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {1, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override { throw std::runtime_error("unexpected fixture dump"); }

private:
    static constexpr u32 address_bias = 0x30;
    std::vector<char> code;
};

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5) {
            throw std::runtime_error("usage: instruction_fixture ps|tes|tcs input.code output.spv [output.metal]");
        }
        const std::string_view name{argv[1]};
        const Shader::Stage stage = name == "ps" ? Shader::Stage::Fragment
                                  : name == "tes" ? Shader::Stage::TessellationEval
                                  : name == "tcs" ? Shader::Stage::TessellationControl
                                  : throw std::runtime_error("unsupported fixture stage");
        InstructionEnvironment environment{argv[2], stage};
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
        Shader::ObjectPool<Shader::IR::Block> blocks;
        Shader::ObjectPool<Shader::IR::Inst> instructions;
        Shader::Maxwell::Flow::CFG cfg{environment, flow_blocks, environment.EntryPoint()};
        Shader::HostTranslateInfo host{};
        host.support_int64 = true;
        if (const char* mode = std::getenv("TOTK_SHADER_COMPILE_MODE")) {
            if (std::string_view{mode} == "online-fast") host.verify_ir = false;
            else if (std::string_view{mode} != "full") throw std::runtime_error("invalid compile mode");
        }
        auto program = Shader::Maxwell::TranslateProgram(instructions, blocks, environment, cfg, host);
        Shader::Profile profile{};
        profile.supported_spirv = 0x00010300;
        profile.support_int64 = true;
        profile.support_vote = true;
        profile.need_fastmath_off = true;
        const auto words = Shader::Backend::SPIRV::EmitSPIRV(profile, program);
        std::ofstream spirv(argv[3], std::ios::binary);
        spirv.write(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(u32));
        spirv.close();
        if (!spirv) throw std::runtime_error("cannot write fixture SPIR-V");
        if (argc == 4) return 0;
        char* source = nullptr;
        char* error = nullptr;
        const int status = totk_spirv_cross_msl(words.data(), words.size(), &source, &error);
        const std::string message = error ? error : "SPIRV-Cross returned no source";
        const std::string metal = source ? source : "";
        totk_spirv_cross_free(source);
        totk_spirv_cross_free(error);
        if (status != 0 || metal.empty()) throw std::runtime_error(message);
        std::ofstream output(argv[4], std::ios::binary);
        output << metal;
        output.close();
        if (!output) throw std::runtime_error("cannot write fixture Metal");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
