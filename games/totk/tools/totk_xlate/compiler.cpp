#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "compiler.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"

namespace Totk {

u32 TextureSlot(u32 bank, u32 offset) {
    if (bank == 2 && offset >= 0x168 && offset < 0x568 && (offset - 0x168) % 8 == 0) {
        return (offset - 0x168) / 8;
    }
    if (bank == 2 && offset >= 0x20 && offset < 0x120 && (offset - 0x20) % 8 == 0) {
        return (offset - 0x20) / 8;
    }
    throw std::runtime_error("texture handle has no supported NVN slot");
}

u32 ImageSlot(u32 bank, u32 offset) {
    if (bank == 2 && offset >= 0x120 && offset < 0x168 && (offset - 0x120) % 8 == 0) {
        return (offset - 0x120) / 8;
    }
    throw std::runtime_error("image handle has no supported NVN slot");
}

Shader::TextureType TextureType(std::string name) {
    if (name.ends_with("_depth")) name.resize(name.size() - 6);
    static const std::map<std::string, Shader::TextureType> types{
        {"1d", Shader::TextureType::Color1D}, {"1darray", Shader::TextureType::ColorArray1D},
        {"2d", Shader::TextureType::Color2D}, {"2darray", Shader::TextureType::ColorArray2D},
        {"2dms", Shader::TextureType::Color2D}, {"2dmsarray", Shader::TextureType::ColorArray2D},
        {"3d", Shader::TextureType::Color3D}, {"cube", Shader::TextureType::ColorCube},
        {"cubearray", Shader::TextureType::ColorArrayCube}, {"buffer", Shader::TextureType::Buffer}};
    const auto type = types.find(name);
    if (type == types.end()) throw std::runtime_error("unsupported texture type: " + name);
    return type->second;
}

std::vector<char> ReadShader(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    const auto length = file.tellg();
    if (length < 0 || length > 16 * 1024 * 1024) {
        throw std::runtime_error("invalid shader size");
    }
    std::vector<char> bytes(static_cast<size_t>(length));
    file.seekg(0);
    if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("cannot read " + path);
    }
    return bytes;
}

namespace {

u32 MaximumOffset(Shader::IR::Value value, u32& budget) {
    constexpr u32 unknown = std::numeric_limits<u32>::max();
    value = value.Resolve();
    if (value.IsImmediate()) {
        return value.Type() == Shader::IR::Type::U32 ? value.U32() : unknown;
    }
    if (budget == 0) return unknown;
    --budget;
    const auto* inst = value.TryInstRecursive();
    if (!inst) return unknown;
    const auto bound = [&budget](Shader::IR::Value argument) {
        return MaximumOffset(argument, budget);
    };
    using Shader::IR::Opcode;
    switch (inst->GetOpcode()) {
    case Opcode::UMin32:
    case Opcode::BitwiseAnd32:
        return std::min(bound(inst->Arg(0)), bound(inst->Arg(1)));
    case Opcode::IAdd32: {
        const u64 sum = u64{bound(inst->Arg(0))} + bound(inst->Arg(1));
        return sum <= unknown ? static_cast<u32>(sum) : unknown;
    }
    case Opcode::ShiftLeftLogical32:
    case Opcode::ShiftRightLogical32: {
        const auto shift = inst->Arg(1).Resolve();
        if (!shift.IsImmediate() || shift.U32() >= 32) return unknown;
        const u32 maximum = bound(inst->Arg(0));
        const u32 count = shift.U32();
        if (inst->GetOpcode() == Opcode::ShiftRightLogical32) return maximum >> count;
        return maximum <= (unknown >> count) ? maximum << count : unknown;
    }
    case Opcode::SelectU32:
        return std::max(bound(inst->Arg(1)), bound(inst->Arg(2)));
    default:
        return unknown;
    }
}

u32 ConstantReadWidth(std::string_view name) {
    return name.ends_with("x2") ? 8u :
           name.ends_with("8") ? 1u : name.ends_with("16") ? 2u : 4u;
}

void GuardDynamicC1Reads(Shader::IR::Program& program, const CompileRequest& request) {
    if (!request.c1_buffer_size) return;
    const u32 capacity = *request.c1_buffer_size;
    if (capacity == 0 || capacity > 65536 || capacity < request.c1.size()) {
        throw std::runtime_error("c1 buffer allocation must contain the supplied data and fit 64 KiB");
    }
    namespace IR = Shader::IR;
    for (auto* block : program.blocks) {
        for (auto& inst : block->Instructions()) {
            const auto name = IR::NameOf(inst.GetOpcode());
            if (!name.starts_with("GetCbuf") || !inst.Arg(0).IsImmediate() ||
                inst.Arg(0).U32() != 1 || inst.Arg(1).IsImmediate()) continue;
            const u32 width = ConstantReadWidth(name);
            u32 budget = 256;
            const u32 maximum = MaximumOffset(inst.Arg(1), budget);
            if (maximum <= request.c1.size() && request.c1.size() - maximum >= width) continue;
            if (capacity < width) {
                throw std::runtime_error("c1 buffer allocation is smaller than a constant read");
            }
            const auto point = IR::Block::InstructionList::s_iterator_to(inst);
            IR::IREmitter ir{*block, point};
            const IR::U32 offset{inst.Arg(1)};
            const auto limit = ir.Imm32(capacity - width);
            const auto valid = ir.ILessThanEqual(offset, limit, false);
            // The host explicitly provides a zero-padded bank. Clamp the actual
            // access too: a select alone must not issue an out-of-bounds load.
            const auto safe_offset = ir.UMin(offset, limit);
            const auto load = block->PrependNewInst(point, inst);
            load->SetArg(1, safe_offset);
            const IR::Value value{&*load};
            if (inst.Type() == IR::Type::U32x2) {
                inst.ReplaceUsesWith(ir.CompositeConstruct(
                    ir.Select(valid, ir.CompositeExtract(value, 0), ir.Imm32(0u)),
                    ir.Select(valid, ir.CompositeExtract(value, 1), ir.Imm32(0u))));
            } else {
                const IR::Value zero = inst.Type() == IR::Type::F32
                    ? IR::Value{ir.Imm32(0.0f)} : IR::Value{ir.Imm32(0u)};
                inst.ReplaceUsesWith(ir.Select(valid, value, zero));
            }
        }
    }
}

class Environment final : public Shader::Environment {
public:
    std::set<u32> texture_queries;
    std::set<u32> image_queries;

    explicit Environment(const CompileRequest& request_) : request{request_}, code{request.code} {
        static_assert(std::endian::native == std::endian::little);
        const auto& name = request.stage;
        if (name == "vs") {
            stage = Shader::Stage::VertexB;
        } else if (name == "ps") {
            stage = Shader::Stage::Fragment;
        } else if (name == "cs") {
            stage = Shader::Stage::Compute;
            if (!request.workgroup) {
                throw std::runtime_error("compute workgroup size is required");
            }
            if (request.entry_point % sizeof(u64) || request.entry_point > code.size() ||
                code.size() - request.entry_point < 2 * sizeof(u64)) {
                throw std::runtime_error("compute entry point is unaligned or outside shader code");
            }
        } else {
            throw std::runtime_error("stage must be vs, ps, or cs");
        }
        if (stage != Shader::Stage::Compute) {
            if (code.size() < sizeof(sph)) {
                throw std::runtime_error("missing graphics shader header");
            }
            std::memcpy(&sph, code.data(), sizeof(sph));
            // Reconstitute the removed NVN prefix so scheduling bundles are
            // aligned relative to the instruction stream, not the SPH file.
            address_bias = 0x30;
            start_address = address_bias;
        }
        const size_t header_size = stage == Shader::Stage::Compute ? 0 : sizeof(sph);
        if (code.size() < header_size + 2 * sizeof(u64)) {
            throw std::runtime_error("missing or unaligned shader instructions");
        }
    }

    u32 EntryPoint() const {
        return stage == Shader::Stage::Compute ? request.entry_point : start_address + sizeof(sph);
    }

    u64 ReadInstruction(u32 address) override {
        if (address < address_bias) {
            throw std::runtime_error("instruction read before shader file");
        }
        address -= address_bias;
        if (address % sizeof(u64)) {
            throw std::runtime_error("unaligned instruction address");
        }
        if (address > code.size() || code.size() - address < sizeof(u64)) {
            throw std::runtime_error(
                "instruction read outside shader file: offset=" + std::to_string(address) +
                ", size=" + std::to_string(code.size()) +
                ", address_bias=" + std::to_string(address_bias));
        }
        u64 word;
        std::memcpy(&word, code.data() + address, sizeof(word));
        return word;
    }

    u32 ReadCbufValue(u32 bank, u32 offset) override {
        if (bank != 1 || offset > request.c1.size() || request.c1.size() - offset < sizeof(u32)) {
            throw std::runtime_error("unavailable constant-buffer state");
        }
        u32 value;
        std::memcpy(&value, request.c1.data() + offset, sizeof(value));
        return value;
    }
    Shader::TextureType ReadTextureType(const Shader::TextureBufferDescriptor& handle) override {
        if (handle.cbuf_index == 2 && handle.cbuf_offset >= 0x120 && handle.cbuf_offset < 0x168) {
            const auto slot = ImageSlot(handle.cbuf_index, handle.cbuf_offset);
            image_queries.insert(slot);
            const auto found = request.images.find(slot);
            return TextureType(found == request.images.end() ? "2d" : found->second.type);
        }
        const auto slot = TextureSlot(handle.cbuf_index, handle.cbuf_offset);
        texture_queries.insert(slot);
        const auto found = request.textures.find(slot);
        return TextureType(found == request.textures.end() ? "2d" : found->second);
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(const Shader::TextureBufferDescriptor& handle) override {
        const auto slot = ImageSlot(handle.cbuf_index, handle.cbuf_offset);
        image_queries.insert(slot);
        const auto found = request.images.find(slot);
        const auto format = found == request.images.end() ? "R8G8B8A8Unorm" : found->second.format;
        static const std::map<std::string, Shader::TexturePixelFormat> formats{
            {"R8G8B8A8Unorm", Shader::TexturePixelFormat::A8B8G8R8_UNORM},
            {"R8G8B8A8Snorm", Shader::TexturePixelFormat::A8B8G8R8_SNORM},
            {"R8G8B8A8Sint", Shader::TexturePixelFormat::A8B8G8R8_SINT},
            {"R8G8B8A8Uint", Shader::TexturePixelFormat::A8B8G8R8_UINT},
            {"R32Float", Shader::TexturePixelFormat::R32_FLOAT},
            {"R32Sint", Shader::TexturePixelFormat::R32_SINT},
            {"R32Uint", Shader::TexturePixelFormat::R32_UINT},
            {"R32G32Float", Shader::TexturePixelFormat::R32G32_FLOAT},
            {"R32G32Sint", Shader::TexturePixelFormat::R32G32_SINT},
            {"R32G32Uint", Shader::TexturePixelFormat::R32G32_UINT},
            {"R32G32B32A32Float", Shader::TexturePixelFormat::R32G32B32A32_FLOAT},
            {"R32G32B32A32Sint", Shader::TexturePixelFormat::R32G32B32A32_SINT},
            {"R32G32B32A32Uint", Shader::TexturePixelFormat::R32G32B32A32_UINT},
            {"R16G16B16A16Float", Shader::TexturePixelFormat::R16G16B16A16_FLOAT},
            {"R16G16B16A16Sint", Shader::TexturePixelFormat::R16G16B16A16_SINT},
            {"R16G16B16A16Uint", Shader::TexturePixelFormat::R16G16B16A16_UINT},
            {"R16G16B16A16Unorm", Shader::TexturePixelFormat::R16G16B16A16_UNORM},
            {"R16G16B16A16Snorm", Shader::TexturePixelFormat::R16G16B16A16_SNORM},
            {"R16Float", Shader::TexturePixelFormat::R16_FLOAT},
            {"R16Sint", Shader::TexturePixelFormat::R16_SINT},
            {"R16Uint", Shader::TexturePixelFormat::R16_UINT},
            {"R16Unorm", Shader::TexturePixelFormat::R16_UNORM},
            {"R16Snorm", Shader::TexturePixelFormat::R16_SNORM},
            {"R16G16Float", Shader::TexturePixelFormat::R16G16_FLOAT},
            {"R16G16Sint", Shader::TexturePixelFormat::R16G16_SINT},
            {"R16G16Uint", Shader::TexturePixelFormat::R16G16_UINT},
            {"R8Sint", Shader::TexturePixelFormat::R8_SINT},
            {"R8Uint", Shader::TexturePixelFormat::R8_UINT},
            {"R8Unorm", Shader::TexturePixelFormat::R8_UNORM},
            {"R8Snorm", Shader::TexturePixelFormat::R8_SNORM},
            {"R8G8Sint", Shader::TexturePixelFormat::R8G8_SINT},
            {"R8G8Uint", Shader::TexturePixelFormat::R8G8_UINT},
            {"R8G8Unorm", Shader::TexturePixelFormat::R8G8_UNORM}};
        const auto type = formats.find(format);
        if (type == formats.end()) throw std::runtime_error("unsupported image format: " + format);
        return type->second;
    }
    bool IsTexturePixelFormatInteger(const Shader::TextureBufferDescriptor& handle) override {
        ReadTexturePixelFormat(handle);
        const auto found = request.images.find(ImageSlot(handle.cbuf_index, handle.cbuf_offset));
        return found != request.images.end() &&
               (found->second.format.ends_with("Sint") || found->second.format.ends_with("Uint"));
    }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 2; }
    u32 LocalMemorySize() const override {
        if (stage == Shader::Stage::Compute) {
            return request.local_memory.value_or(0);
        }
        const u64 size = sph.LocalMemorySize() + sph.common3.shader_local_memory_crs_size;
        if (size > std::numeric_limits<u32>::max()) {
            throw std::runtime_error("graphics local memory size overflows");
        }
        return static_cast<u32>(size);
    }
    u32 SharedMemorySize() const override {
        return request.shared_memory.value_or(0);
    }
    bool HasLocalMemoryState() const override {
        return stage != Shader::Stage::Compute || request.local_memory.has_value();
    }
    std::array<u32, 3> WorkgroupSize() const override {
        if (!request.workgroup) throw std::runtime_error("compute workgroup size is required");
        return *request.workgroup;
    }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {
        throw std::runtime_error("emulator shader dumping is unavailable");
    }

private:
    const CompileRequest& request;
    const std::vector<char>& code;
    u32 address_bias{};
};

} // namespace

CompileResult Compile(const CompileRequest& request) {
    Environment env{request};
    Shader::ObjectPool<Shader::IR::Inst> inst_pool;
    Shader::ObjectPool<Shader::IR::Block> block_pool;
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::Maxwell::Flow::CFG cfg{env, flow_pool, env.EntryPoint()};
    Shader::HostTranslateInfo host{};
    host.verify_ir = request.mode == CompileMode::Full;
    const char* dump_ir = std::getenv("TOTK_XLATE_DUMP_IR");
    host.dump_ir = dump_ir && std::string_view(dump_ir) == "1";
    host.support_float64 = false;
    host.support_float16 = true;
    host.support_int64 = true;
    host.min_ssbo_alignment = 1;
    host.support_snorm_render_buffer = true;
    host.support_conditional_barrier = false;
    auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
    Shader::Profile profile{};
    profile.unified_descriptor_binding = true;
    profile.support_int64 = true;
    profile.support_derivative_control = true;
    profile.max_user_clip_distances = 8;
    Shader::RuntimeInfo runtime{};
    if (request.stage == "vs") {
        runtime.generic_input_types = request.attributes;
    }
    runtime.fragment_output_types = request.colors;
    // Fragment depth is already in window space, independent of the vertex clip-depth range.
    runtime.convert_depth_mode = request.stage == "vs" && request.depth_minus_w;
    runtime.force_early_z = request.stage == "ps" && request.early_fragment_tests;
    runtime.previous_stage_stores.mask.set();
    Shader::Backend::Bindings bindings{};
    if (request.stage == "cs" && request.shared_memory.value_or(0) == 0) {
        for (const auto* block : program.blocks) {
            for (const auto& inst : block->Instructions()) {
                const auto name = Shader::IR::NameOf(inst.GetOpcode());
                if (name.starts_with("LoadShared") || name.starts_with("WriteShared") ||
                    name.starts_with("SharedAtomic")) {
                    throw std::runtime_error("compute shared memory allocation is required by this shader");
                }
            }
        }
    }
    GuardDynamicC1Reads(program, request);
    const auto logical_info = program.info;
    for (auto& desc : program.info.texture_descriptors) {
        const auto found = request.textures.find(TextureRegister(desc));
        if (found != request.textures.end() && found->second.ends_with("_depth")) {
            // Resource aspect and comparison opcodes are independent. Preserve the
            // original descriptors for compare_slots and logical resource names.
            desc.is_depth = true;
        }
    }
    CompileResult result{
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings), logical_info,
        env.texture_queries, env.image_queries, {}};
    for (const auto* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            const auto opcode = inst.GetOpcode();
            const auto name = Shader::IR::NameOf(opcode);
            if (!name.starts_with("GetCbuf")) continue;
            if (!inst.Arg(0).IsImmediate()) {
                throw std::runtime_error("indirect constant-buffer bank needs explicit resource support");
            }
            const auto bank = inst.Arg(0).U32();
            auto& reads = result.constant_reads[bank];
            const auto width = ConstantReadWidth(name);
            if (!inst.Arg(1).IsImmediate()) {
                reads.dynamic = true;
                if (bank == 1) {
                    u32 budget = 256;
                    const auto maximum = MaximumOffset(inst.Arg(1), budget);
                    const auto capacity = request.c1_buffer_size.value_or(request.c1.size());
                    if (maximum > capacity || capacity - maximum < width) {
                        throw std::runtime_error(
                            "dynamic c1 read cannot be bounded within supplied constant segment");
                    }
                }
                continue;
            }
            const auto offset = inst.Arg(1).U32();
            if (bank == 1 && (offset > request.c1.size() || request.c1.size() - offset < width)) {
                throw std::runtime_error("c1 read exceeds supplied constant segment");
            }
            if (offset > std::numeric_limits<u32>::max() - (width - 1)) {
                throw std::runtime_error("constant-buffer read range overflows");
            }
            reads.byte_ranges.emplace(offset, width);
            reads.vec4_offsets.insert(offset & ~15u);
            reads.vec4_offsets.insert((offset + width - 1) & ~15u);
        }
    }
    return result;
}

} // namespace Totk
