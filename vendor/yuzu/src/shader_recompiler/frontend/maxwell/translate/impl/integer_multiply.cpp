// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/maxwell/translate/impl/integer_multiply.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace Shader::Maxwell {
namespace {

void Multiply(TranslatorVisitor& v, u64 insn, IR::U32 b, IR::U32 c, bool mad, bool immediate32) {
    const auto bit = [&](unsigned pos) { return ((insn >> pos) & 1) != 0; };
    const bool cc{bit(immediate32 ? 52 : 47)};
    if (cc || (mad && !immediate32 && (bit(49) || bit(50)))) {
        throw NotImplementedException("IMUL/IMAD CC, X or SAT");
    }
    const bool signed_a{bit(immediate32 ? 54 : mad ? 48 : 40)};
    const bool signed_b{bit(immediate32 ? (mad ? 57 : 55) : mad ? 53 : 41)};
    const bool high{bit(immediate32 ? 53 : mad ? 54 : 39)};
    const u32 mode{mad ? static_cast<u32>((insn >> (immediate32 ? 55 : 51)) & 3) : 0};
    if (mode == 3) {
        throw NotImplementedException("IMAD PO");
    }
    if (mode == 2) {
        b = v.ir.INeg(b);
    } else if (mode == 1) {
        c = v.ir.INeg(c);
    }
    const IR::U32 a{v.GetReg8(insn)};
    const IR::U32 product{high ? MultiplyInteger(v.ir, a, b, signed_a, signed_b).high
                               : v.ir.IMul(a, b)};
    v.X(IR::Reg{static_cast<u32>(insn & 255)}, mad ? IR::U32{v.ir.IAdd(product, c)} : product);
}

} // namespace

void TranslatorVisitor::IMUL_reg(u64 insn) {
    Multiply(*this, insn, GetReg20(insn), ir.Imm32(0), false, false);
}
void TranslatorVisitor::IMUL_cbuf(u64 insn) {
    Multiply(*this, insn, GetCbuf(insn), ir.Imm32(0), false, false);
}
void TranslatorVisitor::IMUL_imm(u64 insn) {
    Multiply(*this, insn, GetImm20(insn), ir.Imm32(0), false, false);
}
void TranslatorVisitor::IMUL32I(u64 insn) {
    Multiply(*this, insn, GetImm32(insn), ir.Imm32(0), false, true);
}
void TranslatorVisitor::IMAD_reg(u64 insn) {
    Multiply(*this, insn, GetReg20(insn), GetReg39(insn), true, false);
}
void TranslatorVisitor::IMAD_rc(u64 insn) {
    Multiply(*this, insn, GetReg39(insn), GetCbuf(insn), true, false);
}
void TranslatorVisitor::IMAD_cr(u64 insn) {
    Multiply(*this, insn, GetCbuf(insn), GetReg39(insn), true, false);
}
void TranslatorVisitor::IMAD_imm(u64 insn) {
    Multiply(*this, insn, GetImm20(insn), GetReg39(insn), true, false);
}
void TranslatorVisitor::IMAD32I(u64 insn) {
    Multiply(*this, insn, GetImm32(insn), X(IR::Reg{static_cast<u32>(insn & 255)}), true, true);
}

} // namespace Shader::Maxwell
