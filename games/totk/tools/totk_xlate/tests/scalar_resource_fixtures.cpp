#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace {
using namespace Shader;
namespace IR = Shader::IR;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Fixture {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::IREmitter ir{block};
    IR::Program program;
    Profile profile;

    Fixture() {
        program.stage = Stage::Compute;
        program.workgroup_size = {4, 1, 1};
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        program.syntax_list = {
            {.data = {.block = &block}, .type = IR::AbstractSyntaxNode::Type::Block},
            {.type = IR::AbstractSyntaxNode::Type::Return},
        };
        program.info.uses_local_invocation_id = true;
        program.info.storage_buffers_descriptors.push_back({0, 0, 1, true});
        program.info.used_storage_buffer_types = IR::Type::U32;
        profile.supported_spirv = 0x00010300;
        profile.unified_descriptor_binding = true;
        profile.support_typeless_image_loads = true;
    }

    IR::Value Inst(IR::Opcode opcode, std::initializer_list<IR::Value> args,
                   IR::TextureInstInfo flags = {}) {
        block.AppendNewInst(opcode, args);
        auto& inst = block.Instructions().back();
        inst.SetFlags(flags);
        return IR::Value{&inst};
    }

    void Output(IR::Value value, u32 offset = 0) {
        const auto address = ir.IAdd(ir.IMul(ir.LocalInvocationIdX(), ir.Imm32(16)),
                                    ir.Imm32(offset));
        Inst(value.Type() == IR::Type::U32x4 ? IR::Opcode::WriteStorage128
                                           : IR::Opcode::WriteStorage32,
             {ir.Imm32(0), address, value});
    }

    void Save(const std::filesystem::path& path) {
        const auto words = Backend::SPIRV::EmitSPIRV(profile, program);
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(u32));
        Require(file.good(), "Failed to write generated module");
    }
};

void Storage(Fixture& fixture, bool buffer, ImageFormat format, bool integer,
             bool signed_format) {
    auto& info = fixture.program.info;
    info.uses_image_buffers = buffer;
    info.uses_typeless_image_reads = format == ImageFormat::Typeless;
    info.uses_typeless_image_writes = format == ImageFormat::Typeless;
    if (buffer) {
        info.image_buffer_descriptors.push_back({
            .format = format, .is_written = true, .is_read = true, .is_integer = integer,
            .cbuf_index = 0, .cbuf_offset = 0, .count = 1, .size_shift = 3,
            .is_signed = signed_format,
        });
    } else {
        info.image_descriptors.push_back({
            .type = TextureType::Color2D, .format = format, .is_written = true,
            .is_read = true, .is_integer = integer, .cbuf_index = 0, .cbuf_offset = 0,
            .count = 1, .size_shift = 3, .is_signed = signed_format,
        });
    }
}

IR::TextureInstInfo Flags(bool buffer, ImageFormat format) {
    IR::TextureInstInfo flags{};
    flags.type.Assign(buffer ? TextureType::Buffer : TextureType::Color2D);
    flags.image_format.Assign(format);
    return flags;
}

void SignedImages(const std::filesystem::path& root) {
    struct Format {
        const char* name;
        ImageFormat format;
        bool integer;
        bool is_signed;
    };
    for (const auto& f : {
             Format{"s8", ImageFormat::R8_SINT, true, true},
             Format{"s16", ImageFormat::R16_SINT, true, true},
             Format{"u8", ImageFormat::R8_UINT, true, false},
             Format{"u16", ImageFormat::R16_UINT, true, false},
             Format{"u32", ImageFormat::R32_UINT, true, false},
             Format{"u32_signed_metadata", ImageFormat::R32_UINT, true, true},
             Format{"s32", ImageFormat::Typeless, true, true},
             Format{"f32", ImageFormat::Typeless, false, false},
         }) {
        for (bool buffer : {false, true}) {
            Fixture fixture;
            Storage(fixture, buffer, f.format, f.integer, f.is_signed);
            auto& ir = fixture.ir;
            const auto x = ir.LocalInvocationIdX();
            const IR::Value coords = buffer ? IR::Value{x}
                                            : ir.CompositeConstruct(x, ir.Imm32(0));
            const auto flags = Flags(buffer, f.format);
            const auto value = fixture.Inst(IR::Opcode::ImageRead, {ir.Imm32(0), coords}, flags);
            fixture.Output(value);
            const auto output_x = ir.IAdd(x, ir.Imm32(4));
            const IR::Value output_coords = buffer ? IR::Value{output_x}
                                                   : ir.CompositeConstruct(output_x, ir.Imm32(0));
            fixture.Inst(IR::Opcode::ImageWrite, {ir.Imm32(0), output_coords, value}, flags);
            fixture.Save(root / (std::string("image_") + f.name + (buffer ? "_buffer" : "_2d") + ".spv"));
        }
    }
}

class ScalarEnvironment final : public Environment {
public:
    ScalarEnvironment() { stage = Stage::Compute; }

    std::optional<u32> samples;
    TextureType texture_type{TextureType::Color2D};
    u64 ReadInstruction(u32) override { throw std::runtime_error("Unexpected instruction read"); }
    u32 ReadCbufValue(u32, u32) override { throw std::runtime_error("Unexpected cbuf state read"); }
    TextureType ReadTextureType(const TextureBufferDescriptor&) override { return texture_type; }
    TexturePixelFormat ReadTexturePixelFormat(const TextureBufferDescriptor&) override {
        return TexturePixelFormat::R8_SINT;
    }
    bool IsTexturePixelFormatInteger(const TextureBufferDescriptor&) override { return true; }
    std::optional<u32> ReadTextureSamples(const TextureBufferDescriptor&) override { return samples; }
    u32 ReadViewportTransformState() override { return 0; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {4, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return {}; }
    void Dump(u64, u64) override {}
};

void SurfaceFrontend(const std::filesystem::path& root) {
    for (bool buffer : {false, true}) {
        for (bool bindless : {false, true}) {
            for (u64 size = 0; size < 7; ++size) {
                Fixture fixture;
                ScalarEnvironment env;
                auto& ir = fixture.ir;
                Maxwell::TranslatorVisitor visitor{env, fixture.block};
                const u32 shift = size < 2 ? 0 : size < 4 ? 1 : static_cast<u32>(size - 2);
                const u32 components = size == 5 ? 2 : size == 6 ? 4 : 1;
                visitor.X(IR::Reg{0}, ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(shift)));
                visitor.X(IR::Reg{1}, ir.Imm32(0));
                if (bindless) visitor.X(IR::Reg{8}, IR::U32{ir.GetCbuf(ir.Imm32(0), ir.Imm32(64))});
                const u64 instruction = 4 | (size << 20) | (u64{1} << 23) |
                    (u64{1} << 52) | (u64{buffer ? 1U : 3U} << 33) |
                    (bindless ? u64{8} << 39 : (u64{1} << 51) | (u64{16} << 36));
                visitor.SULD(instruction);
                const auto reg = [&](u32 index) -> IR::U32 {
                    return index < components ? visitor.X(IR::Reg{4 + index}) : ir.Imm32(0);
                };
                fixture.Output(ir.CompositeConstruct(reg(0), reg(1), reg(2), reg(3)));
                visitor.X(IR::Reg{0}, ir.ShiftLeftLogical(
                    ir.IAdd(ir.LocalInvocationIdX(), ir.Imm32(4)), ir.Imm32(shift)));
                visitor.SUST(instruction);
                Optimization::SsaRewritePass(fixture.program);
                Optimization::ConstantPropagationPass(env, fixture.program);
                Optimization::TexturePass(env, fixture.program, {});
                Optimization::DeadCodeEliminationPass(fixture.program);
                Optimization::CollectShaderInfoPass(env, fixture.program);
                Optimization::VerificationPass(fixture.program);
                fixture.Save(root / ("surface_ba" + std::to_string(size) +
                    (buffer ? "_buffer" : "_2d") + (bindless ? "_bindless.spv" : "_bound.spv")));
            }
            for (u32 operation = 0; operation < 19; ++operation) {
                Fixture fixture;
                ScalarEnvironment env;
                auto& ir = fixture.ir;
                Maxwell::TranslatorVisitor visitor{env, fixture.block};
                const auto lane = ir.LocalInvocationIdX();
                visitor.X(IR::Reg{0}, ir.ShiftLeftLogical(lane, ir.Imm32(2)));
                visitor.X(IR::Reg{1}, ir.Imm32(0));
                if (bindless) visitor.X(IR::Reg{8}, IR::U32{ir.GetCbuf(ir.Imm32(0), ir.Imm32(64))});
                const bool cas = operation >= 17;
                const bool reduction = operation >= 9 && !cas;
                const u32 op = reduction ? operation - 9 : operation;
                const u64 size = !cas && (op == 1 || op == 2) ? 1 : 0;
                u64 instruction = (u64{buffer ? 1U : 3U} << 33) |
                    (bindless ? u64{8} << 39 : u64{16} << 36);
                if (cas) {
                    const IR::U32 initial{ir.Select(
                        ir.IEqual(lane, ir.Imm32(0)), ir.Imm32(0x80000000),
                        ir.Select(ir.IEqual(lane, ir.Imm32(1)), ir.Imm32(0xffffffff),
                            ir.Select(ir.IEqual(lane, ir.Imm32(2)),
                                      ir.Imm32(0), ir.Imm32(0x7fffffff))))};
                    visitor.X(IR::Reg{4}, operation == 17 ? initial : ir.BitwiseXor(initial, ir.Imm32(1)));
                    visitor.X(IR::Reg{5}, ir.Imm32(0x12345678));
                    instruction |= 12 | (u64{4} << 20) | (u64{1} << 28) | (u64{7} << 30);
                    if (bindless) instruction |= u64{1} << 54;
                    visitor.SUATOM_cas(instruction);
                } else if (reduction) {
                    visitor.X(IR::Reg{4}, ir.Imm32(7));
                    instruction |= 4 | (size << 20) | (u64{1} << 23) | (u64{op} << 24);
                    if (!bindless) instruction |= u64{1} << 51;
                    visitor.SURED(instruction);
                } else {
                    visitor.X(IR::Reg{4}, ir.Imm32(7));
                    instruction |= 12 | (u64{4} << 20) | (u64{1} << 28) | (u64{op} << 29);
                    instruction |= bindless ? (u64{1} << 54) | (size << 36) : size << 51;
                    visitor.SUATOM(instruction);
                }
                fixture.Output(reduction ? ir.Imm32(0) : visitor.X(IR::Reg{12}));
                Optimization::SsaRewritePass(fixture.program);
                Optimization::ConstantPropagationPass(env, fixture.program);
                Optimization::TexturePass(env, fixture.program, {});
                Optimization::DeadCodeEliminationPass(fixture.program);
                Optimization::CollectShaderInfoPass(env, fixture.program);
                Optimization::VerificationPass(fixture.program);
                fixture.Save(root / ("surface_atomic" + std::to_string(operation) +
                    (buffer ? "_buffer" : "_2d") + (bindless ? "_bindless.spv" : "_bound.spv")));
            }
        }
    }
}

void SampleCountQueries(const std::filesystem::path& root) {
    const std::array<std::optional<u32>, 6> variants{std::nullopt, 0U, 3U, 1U, 2U, 4U};
    for (bool bindless : {false, true}) {
        for (bool frontend : {false, true}) {
            for (const auto samples : variants) {
                Fixture fixture;
                ScalarEnvironment env;
                env.samples = samples;
                const IR::Value handle = bindless
                    ? IR::Value{fixture.ir.GetCbuf(fixture.ir.Imm32(0), fixture.ir.Imm32(64))}
                    : IR::Value{fixture.ir.Imm32(64)};
                if (frontend) {
                    Maxwell::TranslatorVisitor visitor{env, fixture.block};
                    if (bindless) visitor.X(IR::Reg{8}, IR::U32{handle});
                    const u64 instruction = u64{4} | (u64{8} << 8) | (u64{2} << 22) |
                                            (u64{4} << 31) | (u64{16} << 36);
                    if (bindless) visitor.TXQ_b(instruction);
                    else visitor.TXQ(instruction);
                    fixture.Output(visitor.X(IR::Reg{4}));
                    Optimization::SsaRewritePass(fixture.program);
                    Optimization::ConstantPropagationPass(env, fixture.program);
                } else {
                    fixture.Output(fixture.ir.ImageQuerySamples(handle));
                }
                bool rejected{};
                try {
                    Optimization::TexturePass(env, fixture.program, {});
                } catch (const std::exception&) { rejected = true; }
                const bool valid = samples && (*samples == 1 || *samples == 2 || *samples == 4);
                Require(rejected != valid, "Sample query mishandled unavailable/invalid state");
                if (valid) {
                    Optimization::DeadCodeEliminationPass(fixture.program);
                    Optimization::CollectShaderInfoPass(env, fixture.program);
                    const auto& desc = fixture.program.info.texture_descriptors.at(0);
                    Require(desc.is_multisample == (*samples > 1),
                            "Sample query lost multisample typing");
                    fixture.Save(root / ("query_samples_" + std::to_string(*samples) +
                                         (bindless ? "_bindless" : "") +
                                         (frontend ? "_frontend.spv" : ".spv")));
                }
            }
        }
        for (u64 mask : {u64{1}, u64{8}, u64{12}}) {
            Fixture fixture;
            ScalarEnvironment env;
            env.samples = 4;
            Maxwell::TranslatorVisitor visitor{env, fixture.block};
            const u64 instruction = u64{4} | (u64{8} << 8) | (u64{2} << 22) |
                                    (mask << 31) | (u64{16} << 36);
            bool rejected{};
            try {
                if (bindless) visitor.TXQ_b(instruction);
                else visitor.TXQ(instruction);
            } catch (const NotImplementedException&) {
                rejected = true;
            }
            Require(rejected, "TXQ accepted unavailable texture-header components");
        }
    }
}

void ScalarResourceContracts(const std::filesystem::path& root) {
    struct Sampled {
        const char* name;
        TextureType type;
        bool multisample;
    };
    for (const auto& sampled : {
             Sampled{"sampled_1d", TextureType::Color1D, false},
             Sampled{"sampled_array1d", TextureType::ColorArray1D, false},
             Sampled{"sampled_arraycube", TextureType::ColorArrayCube, false},
             Sampled{"sampled_msarray", TextureType::ColorArray2D, true},
         }) {
        Fixture fixture;
        TextureDescriptor descriptor{};
        descriptor.type = sampled.type;
        descriptor.is_multisample = sampled.multisample;
        descriptor.count = 1;
        fixture.program.info.texture_descriptors.push_back(descriptor);
        fixture.Save(root / (std::string(sampled.name) + ".spv"));
    }
    Fixture rectangle;
    ScalarEnvironment env;
    env.texture_type = TextureType::Color2DRect;
    rectangle.Output(rectangle.ir.ImageQueryDimension(
        rectangle.ir.Imm32(64), rectangle.ir.Imm32(0), rectangle.ir.Imm1(true)));
    Optimization::TexturePass(env, rectangle.program, {});
    Require(rectangle.program.info.texture_descriptors.at(0).type == TextureType::Color2D,
            "Rectangle dimension query retained a frontend-only texture type");
    rectangle.Save(root / "query_rectangle.spv");

    Fixture bank;
    (void)bank.ir.GetCbuf(bank.ir.Imm32(static_cast<u32>(Info::MAX_CBUFS)), bank.ir.Imm32(0));
    bool rejected{};
    try {
        Optimization::CollectShaderInfoPass(env, bank.program);
    } catch (const InvalidArgument&) {
        rejected = true;
    }
    Require(rejected, "Out-of-range constant-buffer bank was accepted");

    Fixture typeless;
    Storage(typeless, false, ImageFormat::Typeless, true, true);
    typeless.profile.support_typeless_image_loads = false;
    const auto zero = typeless.Inst(
        IR::Opcode::ImageRead,
        {typeless.ir.Imm32(0),
         typeless.ir.CompositeConstruct(typeless.ir.Imm32(0), typeless.ir.Imm32(0))},
        Flags(false, ImageFormat::Typeless));
    typeless.Output(zero);
    typeless.Save(root / "typeless_fallback.spv");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) return 2;
        const std::filesystem::path root{argv[1]};
        std::filesystem::create_directories(root);
        SignedImages(root);
        SurfaceFrontend(root);
        SampleCountQueries(root);
        ScalarResourceContracts(root);
        std::cout << "Generated scalar resource modules and checked explicit failures\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
