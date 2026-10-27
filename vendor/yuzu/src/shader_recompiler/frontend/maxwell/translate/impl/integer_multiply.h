// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/frontend/ir/ir_emitter.h"

namespace Shader::Maxwell {

struct IntegerProduct {
    IR::U32 low;
    IR::U32 high;
};

inline IntegerProduct MultiplyInteger(IR::IREmitter& ir, const IR::U32& a, const IR::U32& b,
                                      bool signed_a, bool signed_b) {
    const auto half = [&](const IR::U32& value, u32 offset) {
        return ir.BitFieldExtract(value, ir.Imm32(offset), ir.Imm32(16));
    };
    const IR::U32 a0{half(a, 0)}, a1{half(a, 16)};
    const IR::U32 b0{half(b, 0)}, b1{half(b, 16)};
    const IR::U32 p00{ir.IMul(a0, b0)};
    const IR::U32 middle{ir.IAdd(ir.IMul(a1, b0), half(p00, 16))};
    const IR::U32 middle2{ir.IAdd(ir.IMul(a0, b1), half(middle, 0))};
    IR::U32 high{ir.IAdd(ir.IAdd(ir.IMul(a1, b1), half(middle, 16)), half(middle2, 16))};
    // Signed products differ from the unsigned product by b*2^32 and/or a*2^32.
    if (signed_a) {
        high =
            ir.ISub(high, IR::U32{ir.Select(ir.ILessThan(a, ir.Imm32(0), true), b, ir.Imm32(0))});
    }
    if (signed_b) {
        high =
            ir.ISub(high, IR::U32{ir.Select(ir.ILessThan(b, ir.Imm32(0), true), a, ir.Imm32(0))});
    }
    return {ir.IMul(a, b), high};
}

} // namespace Shader::Maxwell
