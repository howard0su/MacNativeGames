// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace Shader::Maxwell {

// Flag semantics adapted from Ryujinx (MIT); see LICENSES/Ryujinx-MIT.txt.
inline void SetIntegerZnFlags(TranslatorVisitor& v, const IR::U32& value, bool cc,
                              bool extended = false) {
    if (!cc) {
        return;
    }
    IR::U1 zero{v.ir.IEqual(value, v.ir.Imm32(0))};
    if (extended) {
        zero = v.ir.LogicalAnd(zero, v.ir.GetZFlag());
    }
    v.ir.SetZFlag(zero);
    v.ir.SetSFlag(v.ir.ILessThan(value, v.ir.Imm32(0), true));
}

} // namespace Shader::Maxwell
