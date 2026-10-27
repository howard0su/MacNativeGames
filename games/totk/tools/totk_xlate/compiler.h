#pragma once

#include <array>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/shader_info.h"

namespace Totk {

enum class CompileMode { Full, OnlineFast };

constexpr const char* CompileModeName(CompileMode mode) {
    return mode == CompileMode::Full ? "full" : "online-fast";
}

struct ImageState {
    std::string type{"2d"};
    std::string format{"R8G8B8A8Unorm"};
};

struct CompileRequest {
    CompileMode mode{CompileMode::Full};
    std::string stage;
    std::vector<char> code;
    std::vector<unsigned char> c1;
    std::optional<u32> c1_buffer_size;
    std::array<Shader::AttributeType, 32> attributes{};
    std::array<Shader::AttributeType, 8> colors{};
    std::map<u32, std::string> textures;
    std::map<u32, ImageState> images;
    std::optional<std::array<u32, 3>> workgroup;
    u32 entry_point{};
    std::optional<u32> local_memory;
    std::optional<u32> shared_memory;
    bool depth_minus_w{};
    bool early_fragment_tests{};
};

struct CompileResult {
    std::vector<u32> words;
    Shader::Info info;
    std::set<u32> texture_queries;
    std::set<u32> image_queries;
    struct ConstantReads {
        std::set<u32> vec4_offsets;
        std::set<std::pair<u32, u32>> byte_ranges;
        bool dynamic{};
    };
    std::map<u32, ConstantReads> constant_reads;
};

u32 TextureSlot(u32 bank, u32 offset);
template <typename Descriptor>
u32 TextureRegister(const Descriptor& desc) {
    if (desc.cbuf_index == 2 && desc.cbuf_offset >= 0x568 && desc.has_secondary) {
        return TextureSlot(desc.secondary_cbuf_index, desc.secondary_cbuf_offset);
    }
    return TextureSlot(desc.cbuf_index, desc.cbuf_offset);
}

u32 ImageSlot(u32 bank, u32 offset);
Shader::TextureType TextureType(std::string name);
std::vector<char> ReadShader(const std::string& path);
CompileResult Compile(const CompileRequest& request);

} // namespace Totk
