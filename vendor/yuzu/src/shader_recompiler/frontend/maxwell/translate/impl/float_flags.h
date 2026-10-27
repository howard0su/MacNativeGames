// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <type_traits>

#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace Shader::Maxwell {

// Flag semantics adapted from Ryujinx (MIT); see LICENSES/Ryujinx-MIT.txt.
template <typename Float>
void SetFloatFlags(TranslatorVisitor& v, const Float& value, bool cc) {
    if (!cc) {
        return;
    }
    const Float zero{[&] {
        if constexpr (std::is_same_v<Float, IR::F64>) {
            return v.ir.Imm64(0.0);
        } else {
            return v.ir.Imm32(0.0f);
        }
    }()};
    v.ir.SetZFlag(v.ir.FPEqual(value, zero));
    v.ir.SetSFlag(v.ir.FPLessThan(value, zero));
}

} // namespace Shader::Maxwell
