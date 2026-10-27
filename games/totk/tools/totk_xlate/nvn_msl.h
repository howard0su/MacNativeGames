#pragma once

#include <stdexcept>
#include "spirv_msl.hpp"

namespace spirv_cross {

// The Vulkan-oriented backend deliberately ignores OpTypeImage's Depth operand.
// Our frontend knows the bound NVN texture aspect and requires native depth types.
class NvnCompilerMSL : public CompilerMSL {
public:
    using CompilerMSL::CompilerMSL;

protected:
    // Modern Metal can forbid contraction without disabling all helper optimization.
    void emit_header() override {
        CompilerMSL::emit_header();
        if (!uses_precise_math) return;
        statement(R"(
#if __METAL_VERSION__ >= 320
#pragma metal fp math_mode(safe)
#pragma metal fp contract(off)
#define NVN_PRECISE_MATH
#else
#define NVN_PRECISE_MATH [[clang::optnone]]
#endif
template<typename T> NVN_PRECISE_MATH T nvnFPMul(T l, T r) { return fma(l, r, T(0)); }
template<typename T> NVN_PRECISE_MATH T nvnFPAdd(T l, T r) { return fma(T(1), l, r); }
template<typename T> NVN_PRECISE_MATH T nvnFPSub(T l, T r) { return fma(T(-1), r, l); }
#undef NVN_PRECISE_MATH
)");
    }

    void emit_instruction(const Instruction& instruction) override {
        const char* function = nullptr;
        switch (static_cast<spv::Op>(instruction.op)) {
        case spv::OpFMul: function = "nvnFPMul"; break;
        case spv::OpFAdd: function = "nvnFPAdd"; break;
        case spv::OpFSub: function = "nvnFPSub"; break;
        default: break;
        }
        const auto* operands = stream(instruction);
        if (function && has_legacy_nocontract(operands[0], operands[1])) {
            if (!uses_precise_math) {
                uses_precise_math = true;
                force_recompile();
            }
            emit_binary_func_op(operands[0], operands[1], operands[2], operands[3], function);
            return;
        }
        CompilerMSL::emit_instruction(instruction);
    }

    std::string image_type_glsl(const SPIRType& type, uint32_t id, bool member) override {
        auto name = CompilerMSL::image_type_glsl(type, id, member);
        if (type.image.depth && type.array.empty()) {
            if (!name.starts_with("texture2d") && !name.starts_with("texturecube")) {
                throw std::runtime_error("NVN depth texture dimension is unsupported by Metal");
            }
            name.replace(0, 7, "depth");
        }
        return name;
    }

    std::string to_function_name(const TextureFunctionNameArguments& args) override {
        if (!args.base.imgtype->image.depth || !args.has_dref) {
            return CompilerMSL::to_function_name(args);
        }
        if (args.has_array_offsets) {
            throw std::runtime_error("depth comparison gather with constant offsets is unsupported");
        }
        auto native = args;
        native.has_dref = false;
        return CompilerMSL::to_function_name(native) + "_compare";
    }

    std::string to_function_args(const TextureFunctionArguments& args, bool* forward) override {
        auto native = args;
        if (args.base.imgtype->image.depth && args.base.is_gather && !args.dref) {
            native.component = 0;
        }
        return CompilerMSL::to_function_args(native, forward);
    }

private:
    bool uses_precise_math = false;
};

} // namespace spirv_cross
