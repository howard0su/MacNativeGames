// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/frontend/ir/modifiers.h"

namespace Shader::Backend::SPIRV {
namespace {
Id Image(EmitContext& ctx, IR::TextureInstInfo info) {
    if (info.type == TextureType::Buffer) {
        const ImageBufferDefinition def{ctx.image_buffers.at(info.descriptor_index)};
        return def.id;
    } else {
        const ImageDefinition def{ctx.images.at(info.descriptor_index)};
        return def.id;
    }
}

std::pair<Id, Id> AtomicArgs(EmitContext& ctx) {
    const Id scope{ctx.Const(static_cast<u32>(spv::Scope::Device))};
    const Id semantics{ctx.u32_zero_value};
    return {scope, semantics};
}

Id ImageAtomicU32(EmitContext& ctx, IR::Inst* inst, const IR::Value& index, Id coords, Id value,
                  Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id), Id compare = {}) {
    if (!index.IsImmediate() || index.U32() != 0) {
        // TODO: handle layers
        throw NotImplementedException("Image indexing");
    }
    const auto info{inst->Flags<IR::TextureInstInfo>()};
    const Id image{Image(ctx, info)};
    const Id result_vector{info.type == TextureType::Buffer
                               ? ctx.image_buffers.at(info.descriptor_index).result_type
                               : ctx.images.at(info.descriptor_index).result_type};
    if (result_vector.value == ctx.F32[4].value) {
        throw InvalidArgument("Integer image atomic used with a floating-point image");
    }
    const Id scalar_type{result_vector.value == ctx.S32[4].value ? ctx.S32[1] : ctx.U32[1]};
    const Id pointer_type{ctx.TypePointer(spv::StorageClass::Image, scalar_type)};
    const Id pointer{ctx.OpImageTexelPointer(pointer_type, image, coords, ctx.Const(0U))};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    if (scalar_type.value != ctx.U32[1].value) {
        value = ctx.OpBitcast(scalar_type, value);
        if (Sirit::ValidId(compare)) compare = ctx.OpBitcast(scalar_type, compare);
    }
    const Id result{Sirit::ValidId(compare)
        ? ctx.OpAtomicCompareExchange(scalar_type, pointer, scope, semantics, semantics,
                                      value, compare)
        : (ctx.*atomic_func)(scalar_type, pointer, scope, semantics, value)};
    return scalar_type.value == ctx.U32[1].value ? result : ctx.OpBitcast(ctx.U32[1], result);
}

Id ImageAtomicIncDec(EmitContext& ctx, IR::Inst* inst, const IR::Value& index,
                     Id coords, Id limit, bool increment) {
    if (!index.IsImmediate() || index.U32() != 0) {
        throw NotImplementedException("Image indexing");
    }
    const auto info{inst->Flags<IR::TextureInstInfo>()};
    const Id image{Image(ctx, info)};
    const Id vector_type{info.type == TextureType::Buffer
                             ? ctx.image_buffers.at(info.descriptor_index).result_type
                             : ctx.images.at(info.descriptor_index).result_type};
    if (vector_type.value != ctx.U32[4].value) {
        throw InvalidArgument("Image increment/decrement requires an unsigned integer image");
    }
    const Id pointer{ctx.OpImageTexelPointer(
        ctx.TypePointer(spv::StorageClass::Image, ctx.U32[1]), image, coords, ctx.Const(0U))};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    const Id header{ctx.OpLabel()};
    const Id retry{ctx.OpLabel()};
    const Id merge{ctx.OpLabel()};
    ctx.OpBranch(header);
    ctx.AddLabel(header);
    ctx.OpLoopMerge(merge, retry, spv::LoopControlMask::MaskNone);
    ctx.OpBranch(retry);
    ctx.AddLabel(retry);
    const Id old{ctx.OpAtomicLoad(ctx.U32[1], pointer, scope, semantics)};
    Id replacement{};
    if (increment) {
        replacement = ctx.OpSelect(
            ctx.U32[1], ctx.OpUGreaterThanEqual(ctx.U1, old, limit), ctx.u32_zero_value,
            ctx.OpIAdd(ctx.U32[1], old, ctx.Const(1U)));
    } else {
        const Id wrap{ctx.OpLogicalOr(ctx.U1, ctx.OpIEqual(ctx.U1, old, ctx.u32_zero_value),
                                     ctx.OpUGreaterThan(ctx.U1, old, limit))};
        replacement = ctx.OpSelect(ctx.U32[1], wrap, limit,
                                   ctx.OpISub(ctx.U32[1], old, ctx.Const(1U)));
    }
    const Id result{ctx.OpAtomicCompareExchange(ctx.U32[1], pointer, scope, semantics, semantics,
                                               replacement, old)};
    ctx.OpBranchConditional(ctx.OpIEqual(ctx.U1, result, old), merge, header);
    ctx.AddLabel(merge);
    return result;
}
} // Anonymous namespace

Id EmitImageAtomicIAdd32(EmitContext& ctx, IR::Inst* inst, const IR::Value& index, Id coords,
                         Id value) {
    return ImageAtomicU32(ctx, inst, index, coords, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitImageAtomicSMin32(EmitContext& ctx, IR::Inst* inst, const IR::Value& index, Id coords,
                         Id value) {
    return ImageAtomicU32(ctx, inst, index, coords, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitImageAtomicUMin32(EmitContext& ctx, IR::Inst* inst, const IR::Value& index, Id coords,
                         Id value) {
    return ImageAtomicU32(ctx, inst, index, coords, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitImageAtomicSMax32(EmitContext& ctx, IR::Inst* inst, const IR::Value& index, Id coords,
                         Id value) {
    return ImageAtomicU32(ctx, inst, index, coords, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitImageAtomicUMax32(EmitContext& ctx, IR::Inst* inst, const IR::Value& index, Id coords,
                         Id value) {
    return ImageAtomicU32(ctx, inst, index, coords, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitImageAtomicInc32(EmitContext& ctx, IR::Inst* inst, const IR::Value& index, Id coords, Id value) {
    return ImageAtomicIncDec(ctx, inst, index, coords, value, true);
}

Id EmitImageAtomicDec32(EmitContext& ctx, IR::Inst* inst, const IR::Value& index, Id coords, Id value) {
    return ImageAtomicIncDec(ctx, inst, index, coords, value, false);
}

Id EmitImageAtomicAnd32(EmitContext& ctx, IR::Inst* inst, const IR::Value& index, Id coords,
                        Id value) {
    return ImageAtomicU32(ctx, inst, index, coords, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitImageAtomicOr32(EmitContext& ctx, IR::Inst* inst, const IR::Value& index, Id coords,
                       Id value) {
    return ImageAtomicU32(ctx, inst, index, coords, value, &Sirit::Module::OpAtomicOr);
}

Id EmitImageAtomicXor32(EmitContext& ctx, IR::Inst* inst, const IR::Value& index, Id coords,
                        Id value) {
    return ImageAtomicU32(ctx, inst, index, coords, value, &Sirit::Module::OpAtomicXor);
}

Id EmitImageAtomicExchange32(EmitContext& ctx, IR::Inst* inst, const IR::Value& index, Id coords,
                             Id value) {
    return ImageAtomicU32(ctx, inst, index, coords, value, &Sirit::Module::OpAtomicExchange);
}

Id EmitImageAtomicCompareExchange32(EmitContext& ctx, IR::Inst* inst, const IR::Value& index,
                                    Id coords, Id compare, Id replacement) {
    return ImageAtomicU32(ctx, inst, index, coords, replacement, nullptr, compare);
}

Id EmitBindlessImageAtomicCompareExchange32(EmitContext&) {
    throw LogicError("Bindless image atomic was not tracked");
}

Id EmitBoundImageAtomicCompareExchange32(EmitContext&) {
    throw LogicError("Bound image atomic was not tracked");
}

Id EmitBindlessImageAtomicIAdd32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBindlessImageAtomicSMin32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBindlessImageAtomicUMin32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBindlessImageAtomicSMax32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBindlessImageAtomicUMax32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBindlessImageAtomicInc32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBindlessImageAtomicDec32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBindlessImageAtomicAnd32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBindlessImageAtomicOr32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBindlessImageAtomicXor32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBindlessImageAtomicExchange32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBoundImageAtomicIAdd32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBoundImageAtomicSMin32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBoundImageAtomicUMin32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBoundImageAtomicSMax32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBoundImageAtomicUMax32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBoundImageAtomicInc32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBoundImageAtomicDec32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBoundImageAtomicAnd32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBoundImageAtomicOr32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBoundImageAtomicXor32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

Id EmitBoundImageAtomicExchange32(EmitContext&) {
    throw NotImplementedException("SPIR-V Instruction");
}

} // namespace Shader::Backend::SPIRV
