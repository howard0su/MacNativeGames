// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/common_types.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/integer_multiply.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/video_helper.h"

namespace Shader::Maxwell {
void TranslatorVisitor::VMAD(u64 insn) {
    union {
        u64 raw;
        BitField<0, 8, IR::Reg> dest_reg;
        BitField<20, 16, u64> src_b_imm;
        BitField<28, 2, u64> src_b_selector;
        BitField<29, 2, VideoWidth> src_b_width;
        BitField<36, 2, u64> src_a_selector;
        BitField<37, 2, VideoWidth> src_a_width;
        BitField<47, 1, u64> cc;
        BitField<48, 1, u64> src_a_sign;
        BitField<49, 1, u64> src_b_sign;
        BitField<50, 1, u64> is_src_b_reg;
        BitField<51, 2, u64> scale;
        BitField<53, 1, u64> src_c_neg;
        BitField<54, 1, u64> src_a_neg;
        BitField<55, 1, u64> sat;
    } const vmad{insn};

    if (vmad.cc != 0) {
        throw NotImplementedException("VMAD CC");
    }
    if (vmad.scale == 3) {
        throw NotImplementedException("VMAD reserved scale");
    }
    const bool is_b_imm{vmad.is_src_b_reg == 0};
    const IR::U32 src_a{GetReg8(insn)};
    const IR::U32 src_b{is_b_imm ? ir.Imm32(static_cast<u32>(vmad.src_b_imm)) : GetReg20(insn)};
    const IR::U32 src_c{GetReg39(insn)};

    const u32 a_selector{static_cast<u32>(vmad.src_a_selector)};
    // Immediate values can't have a selector
    const u32 b_selector{is_b_imm ? 0U : static_cast<u32>(vmad.src_b_selector)};
    const VideoWidth a_width{vmad.src_a_width};
    const VideoWidth b_width{GetVideoSourceWidth(vmad.src_b_width, is_b_imm)};

    const bool src_a_signed{vmad.src_a_sign != 0};
    const bool src_b_signed{vmad.src_b_sign != 0};
    const IR::U32 op_a{ExtractVideoOperandValue(ir, src_a, a_width, a_selector, src_a_signed)};
    const IR::U32 op_b{ExtractVideoOperandValue(ir, src_b, b_width, b_selector, src_b_signed)};

    const bool po{vmad.src_a_neg != 0 && vmad.src_c_neg != 0};
    const bool neg_product{vmad.src_a_neg != 0 && !po};
    const bool neg_c{vmad.src_c_neg != 0 && !po};
    const bool signed_result{src_a_signed || src_b_signed || neg_product || neg_c};
    const auto sign_extend = [&](const IR::U32& value) {
        return IR::U32{ir.ShiftRightArithmetic(value, ir.Imm32(31))};
    };
    const auto product{MultiplyInteger(ir, op_a, op_b, src_a_signed, src_b_signed)};
    IR::U32 low{product.low};
    IR::U32 high{product.high};
    // The extra limb preserves unsigned products above INT64_MAX under
    // negation/saturation.
    IR::U32 upper{src_a_signed || src_b_signed ? sign_extend(high) : ir.Imm32(0)};
    const auto add = [&](IR::U32 lo, IR::U32 hi, IR::U32 top) {
        const IR::U32 old_low{low};
        low = ir.IAdd(low, lo);
        const IR::U32 carry{ir.Select(ir.ILessThan(low, old_low, false), ir.Imm32(1), ir.Imm32(0))};
        const IR::U32 partial{ir.IAdd(high, hi)};
        const IR::U32 c1{ir.Select(ir.ILessThan(partial, high, false), ir.Imm32(1), ir.Imm32(0))};
        high = ir.IAdd(partial, carry);
        const IR::U32 c2{ir.Select(ir.ILessThan(high, partial, false), ir.Imm32(1), ir.Imm32(0))};
        upper = ir.IAdd(ir.IAdd(ir.IAdd(upper, top), c1), c2);
    };
    if (neg_product) {
        low = ir.BitwiseNot(low);
        high = ir.BitwiseNot(high);
        upper = ir.BitwiseNot(upper);
        add(ir.Imm32(1), ir.Imm32(0), ir.Imm32(0));
    }
    // PTX vmad defines c as sign-extended for a signed final result, including
    // -c.
    const IR::U32 c{neg_c ? ir.BitwiseNot(src_c) : src_c};
    const IR::U32 c_high{signed_result ? sign_extend(c) : ir.Imm32(0)};
    add(c, c_high, c_high);
    if (neg_c || po) {
        add(ir.Imm32(1), ir.Imm32(0), ir.Imm32(0));
    }
    if (vmad.scale != 0) {
        const u32 shift{vmad.scale == 1 ? 7U : 15U};
        low = ir.BitwiseOr(ir.ShiftRightLogical(low, ir.Imm32(shift)),
                           ir.ShiftLeftLogical(high, ir.Imm32(32 - shift)));
        high = ir.BitwiseOr(ir.ShiftRightLogical(high, ir.Imm32(shift)),
                            ir.ShiftLeftLogical(upper, ir.Imm32(32 - shift)));
        upper = signed_result ? ir.ShiftRightArithmetic(upper, ir.Imm32(shift))
                              : ir.ShiftRightLogical(upper, ir.Imm32(shift));
    }
    if (vmad.sat != 0) {
        const IR::U32 expected_high{signed_result ? sign_extend(low) : ir.Imm32(0)};
        const IR::U1 overflow{
            ir.LogicalOr(ir.INotEqual(high, expected_high), ir.INotEqual(upper, expected_high))};
        const IR::U32 clamp{signed_result ? ir.Select(ir.ILessThan(upper, ir.Imm32(0), true),
                                                      ir.Imm32(0x80000000U), ir.Imm32(0x7fffffffU))
                                          : ir.Imm32(0xffffffffU)};
        low = IR::U32{ir.Select(overflow, clamp, low)};
    }
    X(vmad.dest_reg, low);
}

} // namespace Shader::Maxwell
