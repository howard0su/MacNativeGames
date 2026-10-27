// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/bit_field.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace Shader::Maxwell {
namespace {
enum class Mode : u64 {
    PR,
    CC,
};
void P2R(TranslatorVisitor& v, u64 insn, const IR::U32& mask_value) {
    union {
        u64 raw;
        BitField<0, 8, IR::Reg> dest_reg;
        BitField<8, 8, IR::Reg> src;
        BitField<40, 1, Mode> mode;
        BitField<41, 2, u64> byte_selector;
    } const p2r{insn};

    const IR::U32 mask{v.ir.BitwiseAnd(mask_value, v.ir.Imm32(0xffU))};
    const bool pr_mode{p2r.mode == Mode::PR};
    const u32 num_items{pr_mode ? 8U : 4U};
    const u32 offset{static_cast<u32>(p2r.byte_selector) * 8};
    IR::U32 insert{v.ir.Imm32(0)};
    for (u32 index = 0; index < num_items; ++index) {
        const IR::U1 cond{[&v, index, pr_mode] {
            if (pr_mode) {
                return v.ir.GetPred(IR::Pred{index});
            }
            switch (index) {
            case 0:
                return v.ir.GetZFlag();
            case 1:
                return v.ir.GetSFlag();
            case 2:
                return v.ir.GetCFlag();
            case 3:
                return v.ir.GetOFlag();
            }
            throw LogicError("Unreachable P2R index");
        }()};
        const IR::U32 bit{v.ir.Select(cond, v.ir.Imm32(1U << (index + offset)), v.ir.Imm32(0))};
        insert = v.ir.BitwiseOr(insert, bit);
    }
    const IR::U32 shifted_mask{v.ir.ShiftLeftLogical(mask, v.ir.Imm32(offset))};
    const IR::U32 masked_out{v.ir.BitwiseAnd(v.X(p2r.src), v.ir.BitwiseNot(shifted_mask))};
    v.X(p2r.dest_reg, v.ir.BitwiseOr(masked_out, v.ir.BitwiseAnd(insert, shifted_mask)));
}
} // Anonymous namespace

void TranslatorVisitor::P2R_reg(u64 insn) {
    P2R(*this, insn, GetReg20(insn));
}

void TranslatorVisitor::P2R_cbuf(u64 insn) {
    P2R(*this, insn, GetCbuf(insn));
}

void TranslatorVisitor::P2R_imm(u64 insn) {
    P2R(*this, insn, GetImm20(insn));
}

} // namespace Shader::Maxwell
