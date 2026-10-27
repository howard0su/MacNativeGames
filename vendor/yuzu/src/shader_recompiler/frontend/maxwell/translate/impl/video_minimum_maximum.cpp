// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/common_types.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/video_helper.h"

namespace Shader::Maxwell {
namespace {
enum class VideoMinMaxOps : u64 {
    MRG_16H,
    MRG_16L,
    MRG_8B0,
    MRG_8B2,
    ACC,
    MIN,
    MAX,
};

[[nodiscard]] IR::U32 MinMax(IR::IREmitter& ir, const IR::U32& a, const IR::U32& b, bool signed_a,
                             bool signed_b, bool maximum) {
    IR::U1 less{ir.ILessThan(a, b, signed_a && signed_b)};
    if (signed_a != signed_b) {
        const IR::U1 negative{ir.ILessThan(signed_a ? a : b, ir.Imm32(0), true)};
        less =
            signed_a ? ir.LogicalOr(negative, less) : ir.LogicalAnd(ir.LogicalNot(negative), less);
    }
    return IR::U32{ir.Select(less, maximum ? b : a, maximum ? a : b)};
}

[[nodiscard]] IR::U32 ApplyVideoMinMaxOp(IR::IREmitter& ir, const IR::U32& lhs, const IR::U32& rhs,
                                         VideoMinMaxOps op, bool lhs_signed, bool rhs_signed) {
    switch (op) {
    case VideoMinMaxOps::MRG_16H:
        return ir.BitFieldInsert(rhs, lhs, ir.Imm32(16), ir.Imm32(16));
    case VideoMinMaxOps::MRG_16L:
        return ir.BitFieldInsert(rhs, lhs, ir.Imm32(0), ir.Imm32(16));
    case VideoMinMaxOps::MRG_8B0:
        return ir.BitFieldInsert(rhs, lhs, ir.Imm32(0), ir.Imm32(8));
    case VideoMinMaxOps::MRG_8B2:
        return ir.BitFieldInsert(rhs, lhs, ir.Imm32(16), ir.Imm32(8));
    case VideoMinMaxOps::ACC:
        return ir.IAdd(lhs, rhs);
    case VideoMinMaxOps::MIN:
        return MinMax(ir, lhs, rhs, lhs_signed, rhs_signed, false);
    case VideoMinMaxOps::MAX:
        return MinMax(ir, lhs, rhs, lhs_signed, rhs_signed, true);
    default:
        throw NotImplementedException("VMNMX op {}", op);
    }
}
} // Anonymous namespace

void TranslatorVisitor::VMNMX(u64 insn) {
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
        BitField<51, 3, VideoMinMaxOps> op;
        BitField<54, 1, u64> dest_sign;
        BitField<55, 1, u64> sat;
        BitField<56, 1, u64> mx;
    } const vmnmx{insn};

    if (vmnmx.cc != 0) {
        throw NotImplementedException("VMNMX CC");
    }

    const bool is_b_imm{vmnmx.is_src_b_reg == 0};
    const IR::U32 src_a{GetReg8(insn)};
    const IR::U32 src_b{is_b_imm ? ir.Imm32(static_cast<u32>(vmnmx.src_b_imm)) : GetReg20(insn)};
    const IR::U32 src_c{GetReg39(insn)};

    const VideoWidth a_width{vmnmx.src_a_width};
    const VideoWidth b_width{GetVideoSourceWidth(vmnmx.src_b_width, is_b_imm)};

    const u32 a_selector{static_cast<u32>(vmnmx.src_a_selector)};
    // Immediate values can't have a selector
    const u32 b_selector{is_b_imm ? 0U : static_cast<u32>(vmnmx.src_b_selector)};

    const bool src_a_signed{vmnmx.src_a_sign != 0};
    const bool src_b_signed{vmnmx.src_b_sign != 0};
    const IR::U32 op_a{ExtractVideoOperandValue(ir, src_a, a_width, a_selector, src_a_signed)};
    const IR::U32 op_b{ExtractVideoOperandValue(ir, src_b, b_width, b_selector, src_b_signed)};

    const bool maximum{vmnmx.mx != 0};
    const bool dest_signed{vmnmx.dest_sign != 0};
    bool result_signed{src_a_signed == src_b_signed ? src_a_signed : !maximum};
    IR::U32 lhs{MinMax(ir, op_a, op_b, src_a_signed, src_b_signed, maximum)};
    if (vmnmx.sat != 0) {
        // Scalar-video saturation precedes the secondary op and uses the merge field's width.
        const u32 width{vmnmx.op <= VideoMinMaxOps::MRG_16L   ? 16U
                        : vmnmx.op <= VideoMinMaxOps::MRG_8B2 ? 8U
                                                              : 32U};
        const u32 upper{dest_signed   ? (1U << (width - 1)) - 1U
                        : width == 32 ? 0xffffffffU
                                      : (1U << width) - 1U};
        if (result_signed) {
            const u32 lower{dest_signed ? 0xffffffffU << (width - 1) : 0U};
            lhs = ir.SMax(lhs, ir.Imm32(lower));
            if (dest_signed || width != 32) {
                lhs = ir.SMin(lhs, ir.Imm32(upper));
            }
        } else {
            lhs = ir.UMin(lhs, ir.Imm32(upper));
        }
        result_signed = dest_signed;
    }
    X(vmnmx.dest_reg, ApplyVideoMinMaxOp(ir, lhs, src_c, vmnmx.op, result_signed, dest_signed));
}

} // namespace Shader::Maxwell
