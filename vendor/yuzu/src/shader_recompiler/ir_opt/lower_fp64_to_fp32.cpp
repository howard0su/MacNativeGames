// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/bit_cast.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/opcodes.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {

IR::F32 PackedF64ToF32(IR::IREmitter& ir, const IR::Value& packed) {
    const IR::U32 lo{ir.CompositeExtract(packed, 0)};
    const IR::U32 hi{ir.CompositeExtract(packed, 1)};
    const IR::U32 sign{ir.BitwiseAnd(hi, ir.Imm32(0x80000000))};
    const IR::U32 exp{ir.BitFieldExtract(hi, ir.Imm32(20), ir.Imm32(11))};
    const IR::U32 mantissa_hi{ir.BitFieldExtract(hi, ir.Imm32(0), ir.Imm32(20))};
    const IR::U32 fraction{ir.BitwiseOr(ir.ShiftLeftLogical(mantissa_hi, ir.Imm32(3)),
                                      ir.ShiftRightLogical(lo, ir.Imm32(29)))};
    const IR::U32 significand{ir.BitwiseOr(fraction, ir.Imm32(0x800000))};
    const IR::U32 tail{ir.BitwiseAnd(lo, ir.Imm32(0x1fffffff))};
    const auto round = [&](const IR::U32& retained, const IR::U32& discarded,
                           const IR::U32& halfway, const IR::U1& sticky) {
        const IR::U1 odd{ir.INotEqual(ir.BitwiseAnd(retained, ir.Imm32(1)), ir.Imm32(0))};
        const IR::U1 increment{
            ir.LogicalOr(ir.IGreaterThan(discarded, halfway, false),
                         ir.LogicalAnd(ir.IEqual(discarded, halfway), ir.LogicalOr(sticky, odd)))};
        return IR::U32{ir.IAdd(retained, IR::U32{ir.Select(increment, ir.Imm32(1), ir.Imm32(0))})};
    };

    // Adding the rounded hidden bit also carries into the exponent at binade boundaries.
    const IR::U32 normal{
        ir.IAdd(ir.ShiftLeftLogical(ir.ISub(exp, ir.Imm32(897)), ir.Imm32(23)),
                round(significand, tail, ir.Imm32(0x10000000), ir.Imm1(false)))};

    // Clamp even unselected shifts: every emitted SPIR-V shift must be in range.
    const IR::U32 shift{ir.UMin(ir.UMax(ir.ISub(ir.Imm32(897), exp), ir.Imm32(1)),
                               ir.Imm32(24))};
    const IR::U32 halfway{ir.ShiftLeftLogical(ir.Imm32(1), ir.ISub(shift, ir.Imm32(1)))};
    const IR::U32 mask{ir.ISub(ir.ShiftLeftLogical(ir.Imm32(1), shift), ir.Imm32(1))};
    const IR::U32 subnormal{
        round(IR::U32{ir.ShiftRightLogical(significand, shift)}, ir.BitwiseAnd(significand, mask),
              halfway, ir.INotEqual(tail, ir.Imm32(0)))};
    const IR::U32 tiny{
        ir.Select(ir.ILessThan(exp, ir.Imm32(873), false), ir.Imm32(0), subnormal)};
    const IR::U32 finite{ir.Select(ir.ILessThan(exp, ir.Imm32(897), false), tiny, normal)};
    const IR::U32 bounded{
        ir.Select(ir.IGreaterThanEqual(exp, ir.Imm32(1151), false), ir.Imm32(0x7f800000), finite)};
    const IR::U1 is_nan{ir.INotEqual(ir.BitwiseOr(mantissa_hi, lo), ir.Imm32(0))};
    // Preserve the available payload and sign, quieting even NaNs whose payload was truncated.
    const IR::U32 special{
        ir.Select(is_nan, ir.BitwiseOr(fraction, ir.Imm32(0x7fc00000)), ir.Imm32(0x7f800000))};
    const IR::U32 magnitude{ir.Select(ir.IEqual(exp, ir.Imm32(0x7ff)), special, bounded)};
    return ir.BitCast<IR::F32>(ir.BitwiseOr(sign, magnitude));
}

IR::Value F32ToPackedF64(IR::IREmitter& ir, const IR::Value& raw) {
    const IR::U32 value{ir.BitCast<IR::U32>(IR::F32(raw))};
    const IR::U32 sign{ir.BitFieldExtract(value, ir.Imm32(31), ir.Imm32(1))};
    const IR::U32 exp{ir.BitFieldExtract(value, ir.Imm32(23), ir.Imm32(8))};
    const IR::U32 fraction{ir.BitFieldExtract(value, ir.Imm32(0), ir.Imm32(23))};
    const IR::U1 is_subnormal{ir.IEqual(exp, ir.Imm32(0))};
    const IR::U32 shift{
        ir.ISub(ir.Imm32(23), ir.FindUMsb(ir.BitwiseOr(fraction, ir.Imm32(1))))};
    const IR::U32 normalized{
        ir.BitwiseAnd(ir.ShiftLeftLogical(fraction, shift), ir.Imm32(0x7fffff))};
    const IR::U32 nan_fraction{
        ir.Select(ir.LogicalAnd(ir.IEqual(exp, ir.Imm32(0xff)),
                               ir.INotEqual(fraction, ir.Imm32(0))),
                  ir.BitwiseOr(fraction, ir.Imm32(0x400000)), fraction)};
    const IR::U32 mantissa{ir.Select(is_subnormal, normalized, nan_fraction)};
    const IR::U32 mantissa_hi{ir.BitFieldExtract(mantissa, ir.Imm32(3), ir.Imm32(20))};
    const IR::U32 mantissa_lo{ir.BitFieldExtract(mantissa, ir.Imm32(0), ir.Imm32(3))};
    const IR::U32 subnormal_exp{
        ir.Select(ir.IEqual(fraction, ir.Imm32(0)), ir.Imm32(0),
                  ir.ISub(ir.Imm32(897), shift))};
    const IR::U32 exp_if_subnorm{
        ir.Select(is_subnormal, subnormal_exp, ir.IAdd(exp, ir.Imm32(896)))};
    const IR::U32 exp_if_infnan{
        ir.Select(ir.IEqual(exp, ir.Imm32(0xff)), ir.Imm32(0x7ff), exp_if_subnorm)};
    const IR::U32 lo{ir.ShiftLeftLogical(mantissa_lo, ir.Imm32(29))};
    const IR::U32 hi{
        ir.BitwiseOr(ir.ShiftLeftLogical(sign, ir.Imm32(31)),
                     ir.BitwiseOr(ir.ShiftLeftLogical(exp_if_infnan, ir.Imm32(20)), mantissa_hi))};
    return ir.CompositeConstruct(lo, hi);
}

IR::Opcode Replace(IR::Opcode op) {
    switch (op) {
    case IR::Opcode::FPAbs64:
        return IR::Opcode::FPAbs32;
    case IR::Opcode::FPAdd64:
        return IR::Opcode::FPAdd32;
    case IR::Opcode::FPCeil64:
        return IR::Opcode::FPCeil32;
    case IR::Opcode::FPFloor64:
        return IR::Opcode::FPFloor32;
    case IR::Opcode::FPFma64:
        return IR::Opcode::FPFma32;
    case IR::Opcode::FPMax64:
        return IR::Opcode::FPMax32;
    case IR::Opcode::FPMin64:
        return IR::Opcode::FPMin32;
    case IR::Opcode::FPMul64:
        return IR::Opcode::FPMul32;
    case IR::Opcode::FPNeg64:
        return IR::Opcode::FPNeg32;
    case IR::Opcode::FPRoundEven64:
        return IR::Opcode::FPRoundEven32;
    case IR::Opcode::FPSaturate64:
        return IR::Opcode::FPSaturate32;
    case IR::Opcode::FPClamp64:
        return IR::Opcode::FPClamp32;
    case IR::Opcode::FPTrunc64:
        return IR::Opcode::FPTrunc32;
    case IR::Opcode::SelectF64:
        return IR::Opcode::SelectF32;
    case IR::Opcode::CompositeConstructF64x2:
        return IR::Opcode::CompositeConstructF32x2;
    case IR::Opcode::CompositeConstructF64x3:
        return IR::Opcode::CompositeConstructF32x3;
    case IR::Opcode::CompositeConstructF64x4:
        return IR::Opcode::CompositeConstructF32x4;
    case IR::Opcode::CompositeExtractF64x2:
        return IR::Opcode::CompositeExtractF32x2;
    case IR::Opcode::CompositeExtractF64x3:
        return IR::Opcode::CompositeExtractF32x3;
    case IR::Opcode::CompositeExtractF64x4:
        return IR::Opcode::CompositeExtractF32x4;
    case IR::Opcode::CompositeInsertF64x2:
        return IR::Opcode::CompositeInsertF32x2;
    case IR::Opcode::CompositeInsertF64x3:
        return IR::Opcode::CompositeInsertF32x3;
    case IR::Opcode::CompositeInsertF64x4:
        return IR::Opcode::CompositeInsertF32x4;
    case IR::Opcode::FPOrdEqual64:
        return IR::Opcode::FPOrdEqual32;
    case IR::Opcode::FPUnordEqual64:
        return IR::Opcode::FPUnordEqual32;
    case IR::Opcode::FPOrdNotEqual64:
        return IR::Opcode::FPOrdNotEqual32;
    case IR::Opcode::FPUnordNotEqual64:
        return IR::Opcode::FPUnordNotEqual32;
    case IR::Opcode::FPOrdLessThan64:
        return IR::Opcode::FPOrdLessThan32;
    case IR::Opcode::FPUnordLessThan64:
        return IR::Opcode::FPUnordLessThan32;
    case IR::Opcode::FPOrdGreaterThan64:
        return IR::Opcode::FPOrdGreaterThan32;
    case IR::Opcode::FPUnordGreaterThan64:
        return IR::Opcode::FPUnordGreaterThan32;
    case IR::Opcode::FPOrdLessThanEqual64:
        return IR::Opcode::FPOrdLessThanEqual32;
    case IR::Opcode::FPUnordLessThanEqual64:
        return IR::Opcode::FPUnordLessThanEqual32;
    case IR::Opcode::FPOrdGreaterThanEqual64:
        return IR::Opcode::FPOrdGreaterThanEqual32;
    case IR::Opcode::FPUnordGreaterThanEqual64:
        return IR::Opcode::FPUnordGreaterThanEqual32;
    case IR::Opcode::FPIsNan64:
        return IR::Opcode::FPIsNan32;
    case IR::Opcode::ConvertS16F64:
        return IR::Opcode::ConvertS16F32;
    case IR::Opcode::ConvertS32F64:
        return IR::Opcode::ConvertS32F32;
    case IR::Opcode::ConvertS64F64:
        return IR::Opcode::ConvertS64F32;
    case IR::Opcode::ConvertU16F64:
        return IR::Opcode::ConvertU16F32;
    case IR::Opcode::ConvertU32F64:
        return IR::Opcode::ConvertU32F32;
    case IR::Opcode::ConvertU64F64:
        return IR::Opcode::ConvertU64F32;
    case IR::Opcode::ConvertF32F64:
        return IR::Opcode::Identity;
    case IR::Opcode::ConvertF64F32:
        return IR::Opcode::Identity;
    case IR::Opcode::ConvertF64S8:
        return IR::Opcode::ConvertF32S8;
    case IR::Opcode::ConvertF64S16:
        return IR::Opcode::ConvertF32S16;
    case IR::Opcode::ConvertF64S32:
        return IR::Opcode::ConvertF32S32;
    case IR::Opcode::ConvertF64S64:
        return IR::Opcode::ConvertF32S64;
    case IR::Opcode::ConvertF64U8:
        return IR::Opcode::ConvertF32U8;
    case IR::Opcode::ConvertF64U16:
        return IR::Opcode::ConvertF32U16;
    case IR::Opcode::ConvertF64U32:
        return IR::Opcode::ConvertF32U32;
    case IR::Opcode::ConvertF64U64:
        return IR::Opcode::ConvertF32U64;
    default:
        return op;
    }
}

void Lower(IR::Block& block, IR::Inst& inst) {
    for (size_t index = 0; index < inst.NumArgs(); ++index) {
        const IR::Value arg{inst.Arg(index)};
        if (!arg.IsImmediate() || arg.Type() != IR::Type::F64) {
            continue;
        }
        IR::IREmitter ir(block, IR::Block::InstructionList::s_iterator_to(inst));
        const u64 bits{Common::BitCast<u64>(arg.F64())};
        const IR::Value packed{ir.CompositeConstruct(ir.Imm32(static_cast<u32>(bits)),
                                                     ir.Imm32(static_cast<u32>(bits >> 32)))};
        inst.SetArg(index, PackedF64ToF32(ir, packed));
    }
    switch (inst.GetOpcode()) {
    case IR::Opcode::PackDouble2x32: {
        IR::IREmitter ir(block, IR::Block::InstructionList::s_iterator_to(inst));
        inst.ReplaceUsesWith(PackedF64ToF32(ir, inst.Arg(0)));
        break;
    }
    case IR::Opcode::UnpackDouble2x32: {
        IR::IREmitter ir(block, IR::Block::InstructionList::s_iterator_to(inst));
        inst.ReplaceUsesWith(F32ToPackedF64(ir, inst.Arg(0)));
        break;
    }
    default:
        inst.ReplaceOpcode(Replace(inst.GetOpcode()));
        break;
    }
}

} // Anonymous namespace

void LowerFp64ToFp32(IR::Program& program) {
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            Lower(*block, inst);
        }
    }
}

} // namespace Shader::Optimization
