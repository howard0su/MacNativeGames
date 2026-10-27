// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <string_view>
#include <tuple>

#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>

#include "common/alignment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {
/// Address in constant buffers to the storage buffer descriptor
struct StorageBufferAddr {
    auto operator<=>(const StorageBufferAddr&) const noexcept = default;

    u32 index;
    u32 offset;
};

/// Block iterator to a global memory instruction and the storage buffer it uses
struct StorageInst {
    StorageBufferAddr storage_buffer;
    IR::Inst* inst;
    IR::Block* block;
};

/// Bias towards a certain range of constant buffers when looking for storage buffers
struct Bias {
    u32 index;
    u32 offset_begin;
    u32 offset_end;
    u32 alignment;
};

using boost::container::flat_set;
using boost::container::small_vector;
using StorageBufferSet =
    flat_set<StorageBufferAddr, std::less<StorageBufferAddr>, small_vector<StorageBufferAddr, 16>>;
using StorageInstVector = small_vector<StorageInst, 24>;
using StorageWritesSet =
    flat_set<StorageBufferAddr, std::less<StorageBufferAddr>, small_vector<StorageBufferAddr, 16>>;

struct StorageInfo {
    StorageBufferSet set;
    StorageInstVector to_replace;
    StorageWritesSet writes;
};

/// Returns true when the instruction is a global memory instruction
bool IsGlobalMemory(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadGlobalS8:
    case IR::Opcode::LoadGlobalU8:
    case IR::Opcode::LoadGlobalS16:
    case IR::Opcode::LoadGlobalU16:
    case IR::Opcode::LoadGlobal32:
    case IR::Opcode::LoadGlobal64:
    case IR::Opcode::LoadGlobal128:
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return true;
    default:
        return false;
    }
}

/// Returns true when the instruction is a global memory instruction
bool IsGlobalMemoryWrite(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return true;
    default:
        return false;
    }
}

/// Converts a global memory opcode to its storage buffer equivalent
IR::Opcode GlobalToStorage(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::LoadGlobalS8:
        return IR::Opcode::LoadStorageS8;
    case IR::Opcode::LoadGlobalU8:
        return IR::Opcode::LoadStorageU8;
    case IR::Opcode::LoadGlobalS16:
        return IR::Opcode::LoadStorageS16;
    case IR::Opcode::LoadGlobalU16:
        return IR::Opcode::LoadStorageU16;
    case IR::Opcode::LoadGlobal32:
        return IR::Opcode::LoadStorage32;
    case IR::Opcode::LoadGlobal64:
        return IR::Opcode::LoadStorage64;
    case IR::Opcode::LoadGlobal128:
        return IR::Opcode::LoadStorage128;
    case IR::Opcode::WriteGlobalS8:
        return IR::Opcode::WriteStorageS8;
    case IR::Opcode::WriteGlobalU8:
        return IR::Opcode::WriteStorageU8;
    case IR::Opcode::WriteGlobalS16:
        return IR::Opcode::WriteStorageS16;
    case IR::Opcode::WriteGlobalU16:
        return IR::Opcode::WriteStorageU16;
    case IR::Opcode::WriteGlobal32:
        return IR::Opcode::WriteStorage32;
    case IR::Opcode::WriteGlobal64:
        return IR::Opcode::WriteStorage64;
    case IR::Opcode::WriteGlobal128:
        return IR::Opcode::WriteStorage128;
    case IR::Opcode::GlobalAtomicIAdd32:
        return IR::Opcode::StorageAtomicIAdd32;
    case IR::Opcode::GlobalAtomicSMin32:
        return IR::Opcode::StorageAtomicSMin32;
    case IR::Opcode::GlobalAtomicUMin32:
        return IR::Opcode::StorageAtomicUMin32;
    case IR::Opcode::GlobalAtomicSMax32:
        return IR::Opcode::StorageAtomicSMax32;
    case IR::Opcode::GlobalAtomicUMax32:
        return IR::Opcode::StorageAtomicUMax32;
    case IR::Opcode::GlobalAtomicInc32:
        return IR::Opcode::StorageAtomicInc32;
    case IR::Opcode::GlobalAtomicDec32:
        return IR::Opcode::StorageAtomicDec32;
    case IR::Opcode::GlobalAtomicAnd32:
        return IR::Opcode::StorageAtomicAnd32;
    case IR::Opcode::GlobalAtomicOr32:
        return IR::Opcode::StorageAtomicOr32;
    case IR::Opcode::GlobalAtomicXor32:
        return IR::Opcode::StorageAtomicXor32;
    case IR::Opcode::GlobalAtomicExchange32:
        return IR::Opcode::StorageAtomicExchange32;
    case IR::Opcode::GlobalAtomicIAdd64:
        return IR::Opcode::StorageAtomicIAdd64;
    case IR::Opcode::GlobalAtomicSMin64:
        return IR::Opcode::StorageAtomicSMin64;
    case IR::Opcode::GlobalAtomicUMin64:
        return IR::Opcode::StorageAtomicUMin64;
    case IR::Opcode::GlobalAtomicSMax64:
        return IR::Opcode::StorageAtomicSMax64;
    case IR::Opcode::GlobalAtomicUMax64:
        return IR::Opcode::StorageAtomicUMax64;
    case IR::Opcode::GlobalAtomicAnd64:
        return IR::Opcode::StorageAtomicAnd64;
    case IR::Opcode::GlobalAtomicOr64:
        return IR::Opcode::StorageAtomicOr64;
    case IR::Opcode::GlobalAtomicXor64:
        return IR::Opcode::StorageAtomicXor64;
    case IR::Opcode::GlobalAtomicExchange64:
        return IR::Opcode::StorageAtomicExchange64;
    case IR::Opcode::GlobalAtomicIAdd32x2:
        return IR::Opcode::StorageAtomicIAdd32x2;
    case IR::Opcode::GlobalAtomicSMin32x2:
        return IR::Opcode::StorageAtomicSMin32x2;
    case IR::Opcode::GlobalAtomicUMin32x2:
        return IR::Opcode::StorageAtomicUMin32x2;
    case IR::Opcode::GlobalAtomicSMax32x2:
        return IR::Opcode::StorageAtomicSMax32x2;
    case IR::Opcode::GlobalAtomicUMax32x2:
        return IR::Opcode::StorageAtomicUMax32x2;
    case IR::Opcode::GlobalAtomicAnd32x2:
        return IR::Opcode::StorageAtomicAnd32x2;
    case IR::Opcode::GlobalAtomicOr32x2:
        return IR::Opcode::StorageAtomicOr32x2;
    case IR::Opcode::GlobalAtomicXor32x2:
        return IR::Opcode::StorageAtomicXor32x2;
    case IR::Opcode::GlobalAtomicExchange32x2:
        return IR::Opcode::StorageAtomicExchange32x2;
    case IR::Opcode::GlobalAtomicAddF32:
        return IR::Opcode::StorageAtomicAddF32;
    case IR::Opcode::GlobalAtomicAddF16x2:
        return IR::Opcode::StorageAtomicAddF16x2;
    case IR::Opcode::GlobalAtomicMinF16x2:
        return IR::Opcode::StorageAtomicMinF16x2;
    case IR::Opcode::GlobalAtomicMaxF16x2:
        return IR::Opcode::StorageAtomicMaxF16x2;
    case IR::Opcode::GlobalAtomicAddF32x2:
        return IR::Opcode::StorageAtomicAddF32x2;
    case IR::Opcode::GlobalAtomicMinF32x2:
        return IR::Opcode::StorageAtomicMinF32x2;
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return IR::Opcode::StorageAtomicMaxF32x2;
    default:
        throw InvalidArgument("Invalid global memory opcode {}", opcode);
    }
}

/// Returns true when a storage buffer address satisfies a bias
bool MeetsBias(const StorageBufferAddr& storage_buffer, const Bias& bias) noexcept {
    return storage_buffer.index == bias.index && storage_buffer.offset >= bias.offset_begin &&
           storage_buffer.offset < bias.offset_end;
}

struct LowAddrInfo {
    IR::U32 value;
    s32 imm_offset;
};

/// Tries to track the first 32-bits of a global memory instruction
std::optional<LowAddrInfo> TrackLowAddress(IR::Inst* inst) {
    // The first argument is the low level GPU pointer to the global memory instruction
    const IR::Value addr{inst->Arg(0)};
    if (addr.IsImmediate()) {
        // Not much we can do if it's an immediate
        return std::nullopt;
    }
    // This address is expected to either be a PackUint2x32, a IAdd64, or a CompositeConstructU32x2
    IR::Inst* addr_inst{addr.InstRecursive()};
    s32 imm_offset{0};
    if (addr_inst->GetOpcode() == IR::Opcode::IAdd64) {
        // If it's an IAdd64, get the immediate offset it is applying and grab the address
        // instruction. This expects for the instruction to be canonicalized having the address on
        // the first argument and the immediate offset on the second one.
        const IR::U64 imm_offset_value{addr_inst->Arg(1)};
        if (!imm_offset_value.IsImmediate()) {
            return std::nullopt;
        }
        imm_offset = static_cast<s32>(static_cast<s64>(imm_offset_value.U64()));
        const IR::U64 iadd_addr{addr_inst->Arg(0)};
        if (iadd_addr.IsImmediate()) {
            return std::nullopt;
        }
        addr_inst = iadd_addr.InstRecursive();
    }
    // With IAdd64 handled, now PackUint2x32 is expected
    if (addr_inst->GetOpcode() == IR::Opcode::PackUint2x32) {
        // PackUint2x32 is expected to be generated from a vector
        const IR::Value vector{addr_inst->Arg(0)};
        if (vector.IsImmediate()) {
            return std::nullopt;
        }
        addr_inst = vector.InstRecursive();
    }
    // The vector is expected to be a CompositeConstructU32x2
    if (addr_inst->GetOpcode() != IR::Opcode::CompositeConstructU32x2) {
        return std::nullopt;
    }
    // Grab the first argument from the CompositeConstructU32x2, this is the low address.
    return LowAddrInfo{
        .value{IR::U32{addr_inst->Arg(0)}},
        .imm_offset = imm_offset,
    };
}

/// Tries to track the storage buffer address used by a global memory instruction
std::optional<StorageBufferAddr> Track(const IR::Value& value, const Bias* bias) {
    const auto pred{[bias](const IR::Inst* inst) -> std::optional<StorageBufferAddr> {
        if (inst->GetOpcode() != IR::Opcode::GetCbufU32 &&
            inst->GetOpcode() != IR::Opcode::GetCbufU32x2) {
            return std::nullopt;
        }
        const IR::Value index{inst->Arg(0)};
        const IR::Value offset{inst->Arg(1)};
        if (!index.IsImmediate()) {
            // Definitely not a storage buffer if it's read from a
            // non-immediate index
            return std::nullopt;
        }
        if (!offset.IsImmediate()) {
            // TODO: Support SSBO arrays
            return std::nullopt;
        }
        const StorageBufferAddr storage_buffer{
            .index = index.U32(),
            .offset = offset.U32(),
        };
        const u32 alignment{bias ? bias->alignment : 8U};
        if (!Common::IsAligned(storage_buffer.offset, alignment)) {
            // The SSBO pointer has to be aligned
            return std::nullopt;
        }
        if (bias && !MeetsBias(storage_buffer, *bias)) {
            // We have to blacklist some addresses in case we wrongly
            // point to them
            return std::nullopt;
        }
        return storage_buffer;
    }};
    if (value.IsImmediate()) {
        return std::nullopt;
    }
    std::queue<const IR::Inst*> pending;
    flat_set<const IR::Inst*, std::less<const IR::Inst*>,
             small_vector<const IR::Inst*, 16>> visited;
    pending.push(value.InstRecursive());
    while (!pending.empty()) {
        const IR::Inst* inst{pending.front()};
        pending.pop();
        if (!visited.insert(inst).second) {
            continue;
        }
        if (const auto result{pred(inst)}) {
            return result;
        }
        // The address used to load a spill is not the pointer stored in that spill.
        if (inst->GetOpcode() == IR::Opcode::LoadLocal ||
            std::string_view{IR::NameOf(inst->GetOpcode())}.starts_with("LoadShared")) {
            continue;
        }
        for (size_t arg = inst->NumArgs(); arg--;) {
            if (!inst->Arg(arg).IsImmediate()) {
                pending.push(inst->Arg(arg).InstRecursive());
            }
        }
    }
    return std::nullopt;
}

// Recover descriptor identity, not the spilled value: the original memory operations and
// runtime pointer arithmetic must remain intact. Unlike SSA edges, memory edges also require
// an initializing write and agreement from every potentially overlapping writer.
class MemoryPointerProvenance {
    struct Address {
        IR::Inst* base{};
        u32 offset{};
    };
    struct Write {
        bool shared;
        Address address;
        u32 width;
        IR::Value value;
        const IR::Inst* inst;
    };
    struct Origin {
        std::optional<StorageBufferAddr> buffer;
        bool valid{true};
    };

public:
    explicit MemoryPointerProvenance(IR::Program& program_, bool dump_ir_)
        : program{program_}, dump_ir{dump_ir_} {}

    std::optional<StorageBufferAddr> TrackPointer(const IR::Value& low, const IR::Value& high,
                                                const Bias* bias) {
        Initialize();
        const Origin result{Find(low, bias, 0, 0)};
        if (!result.valid || !result.buffer) {
            return std::nullopt;
        }
        const Origin upper{Find(high, bias, 4, 0)};
        if (!upper.valid || !upper.buffer || upper.buffer->index != result.buffer->index ||
            upper.buffer->offset != result.buffer->offset + 4) {
            return std::nullopt;
        }
        return result.buffer;
    }

private:
    static Address Normalize(IR::Value value) {
        Address address;
        for (unsigned depth = 0; depth < 64; ++depth) {
            value = value.Resolve();
            if (value.IsImmediate()) {
                address.offset += value.U32();
                return address;
            }
            IR::Inst* inst{value.Inst()};
            if ((inst->GetOpcode() == IR::Opcode::IAdd32 ||
                 inst->GetOpcode() == IR::Opcode::ISub32) && inst->Arg(1).IsImmediate()) {
                address.offset += inst->GetOpcode() == IR::Opcode::IAdd32
                    ? inst->Arg(1).U32() : 0U - inst->Arg(1).U32();
                value = inst->Arg(0);
            } else if (inst->GetOpcode() == IR::Opcode::IAdd32 && inst->Arg(0).IsImmediate()) {
                address.offset += inst->Arg(0).U32();
                value = inst->Arg(1);
            } else {
                address.base = inst;
                return address;
            }
        }
        address.base = value.InstRecursive();
        return address;
    }

    static u64 Maximum(IR::Value value, unsigned depth = 0) {
        constexpr u64 limit{std::numeric_limits<u32>::max()};
        if (depth >= 32) {
            return limit;
        }
        value = value.Resolve();
        if (value.IsImmediate()) {
            return value.U32();
        }
        const IR::Inst& inst{*value.Inst()};
        if (inst.GetOpcode() == IR::Opcode::BitFieldUExtract && inst.Arg(2).IsImmediate()) {
            const u32 bits{inst.Arg(2).U32()};
            return bits < 32 ? (u64{1} << bits) - 1 : limit;
        }
        if (inst.GetOpcode() == IR::Opcode::BitwiseAnd32) {
            return std::min(Maximum(inst.Arg(0), depth + 1), Maximum(inst.Arg(1), depth + 1));
        }
        if (inst.GetOpcode() == IR::Opcode::IMul32 || inst.GetOpcode() == IR::Opcode::IAdd32) {
            const u64 left{Maximum(inst.Arg(0), depth + 1)};
            const u64 right{Maximum(inst.Arg(1), depth + 1)};
            return std::min(limit, inst.GetOpcode() == IR::Opcode::IMul32
                                       ? left * right : left + right);
        }
        return limit;
    }

    static u64 Multiple(IR::Value value, unsigned depth = 0) {
        if (depth >= 32) {
            return 1;
        }
        value = value.Resolve();
        if (value.IsImmediate()) {
            return value.U32();
        }
        const IR::Inst& inst{*value.Inst()};
        if (inst.GetOpcode() == IR::Opcode::IMul32) {
            const u64 multiple{Multiple(inst.Arg(0), depth + 1) *
                               Multiple(inst.Arg(1), depth + 1)};
            const bool wraps{Maximum(inst.Arg(0)) * Maximum(inst.Arg(1)) >
                             std::numeric_limits<u32>::max()};
            return wraps ? std::gcd(multiple, u64{1} << 32) : multiple;
        }
        if (inst.GetOpcode() == IR::Opcode::ShiftLeftLogical32 && inst.Arg(1).IsImmediate()) {
            const u32 shift{inst.Arg(1).U32()};
            return shift < 32 ? u64{1} << shift : 1;
        }
        return 1;
    }

    static u64 Stride(const Address& address, u32 width) {
        if (!address.base) {
            return u64{address.offset} + width - 1 > std::numeric_limits<u32>::max()
                ? u64{1} << 32 : 0;
        }
        const IR::Value base{address.base};
        const u64 multiple{Multiple(base)};
        return Maximum(base) + address.offset + width - 1 > std::numeric_limits<u32>::max()
            ? std::gcd(multiple, u64{1} << 32) : multiple;
    }

    static bool MayOverlap(const Address& a, u32 width_a, const Address& b, u32 width_b,
                           bool shared) {
        if (!shared && a.base == b.base) {
            for (u32 i = 0; i < width_a; ++i) {
                for (u32 j = 0; j < width_b; ++j) {
                    if (a.offset + i == b.offset + j) {
                        return true;
                    }
                }
            }
            return false;
        }
        const u64 stride{std::gcd(Stride(a, width_a), Stride(b, width_b))};
        for (u32 i = 0; i < width_a; ++i) {
            for (u32 j = 0; j < width_b; ++j) {
                if (stride ? (u64{a.offset} + i) % stride == (u64{b.offset} + j) % stride
                           : a.offset + i == b.offset + j) {
                    return true;
                }
            }
        }
        return false;
    }

    void Initialize() {
        if (initialized) {
            return;
        }
        initialized = true;
        for (IR::Block* block : program.blocks) {
            size_t order{0};
            for (IR::Inst& inst : *block) {
                locations.emplace(&inst, std::pair{block, order++});
                const IR::Opcode op{inst.GetOpcode()};
                const bool shared{op != IR::Opcode::WriteLocal};
                u32 width{};
                switch (op) {
                case IR::Opcode::WriteLocal:
                case IR::Opcode::WriteSharedU8:
                    width = 1;
                    break;
                case IR::Opcode::WriteSharedU16:
                    width = 2;
                    break;
                case IR::Opcode::WriteSharedU32:
                    width = 4;
                    break;
                case IR::Opcode::WriteSharedU64:
                    width = 8;
                    break;
                case IR::Opcode::WriteSharedU128:
                    width = 16;
                    break;
                default:
                    if (!std::string_view{IR::NameOf(op)}.starts_with("SharedAtomic")) {
                        continue;
                    }
                    width = op == IR::Opcode::SharedAtomicExchange64 ||
                            op == IR::Opcode::SharedAtomicExchange32x2 ? 8 : 4;
                    break;
                }
                const Address address{Normalize(inst.Arg(0))};
                if (op == IR::Opcode::WriteLocal || op == IR::Opcode::WriteSharedU32) {
                    writes.push_back({shared, address, width, inst.Arg(1), &inst});
                } else if (op == IR::Opcode::WriteSharedU64 || op == IR::Opcode::WriteSharedU128) {
                    const IR::Value vector{inst.Arg(1).Resolve()};
                    const auto construct{width == 8 ? IR::Opcode::CompositeConstructU32x2
                                                    : IR::Opcode::CompositeConstructU32x4};
                    if (!vector.IsImmediate() && vector.Inst()->GetOpcode() == construct) {
                        for (u32 word = 0; word < width / 4; ++word) {
                            writes.push_back({true, {address.base, address.offset + word * 4},
                                              4, vector.Inst()->Arg(word), &inst});
                        }
                    } else {
                        writes.push_back({true, address, width, {}, &inst});
                    }
                } else {
                    writes.push_back({shared, address, width, {}, &inst});
                }
            }
        }
    }

    bool Dominates(const IR::Inst* store, const IR::Inst* load) {
        const auto [store_block, store_order]{locations.at(store)};
        const auto [load_block, load_order]{locations.at(load)};
        if (store_block == load_block) {
            return store_order < load_order;
        }
        const auto key{std::pair{store_block, load_block}};
        if (const auto it{dominance.find(key)}; it != dominance.end()) {
            return it->second;
        }
        std::vector<IR::Block*> pending{load_block};
        std::set<IR::Block*> visited;
        while (!pending.empty()) {
            IR::Block* block{pending.back()};
            pending.pop_back();
            if (block == store_block || !visited.insert(block).second) {
                continue;
            }
            if (block->ImmPredecessors().empty()) {
                dominance.emplace(key, false);
                return false;
            }
            for (IR::Block* predecessor : block->ImmPredecessors()) {
                pending.push_back(predecessor);
            }
        }
        dominance.emplace(key, true);
        return true;
    }

    Origin FromMemory(const IR::Inst& load, const Bias* bias, u32 phase, unsigned depth,
                      u32 word = 0) {
        const auto key{std::tuple{&load, bias != nullptr, phase, word}};
        if (const auto it{origins.find(key)}; it != origins.end()) {
            return it->second;
        }
        // Leave a failed entry while resolving to reject cyclic memory dependencies.
        origins.emplace(key, Origin{std::nullopt, false});
        const bool shared{load.GetOpcode() != IR::Opcode::LoadLocal};
        Address address{Normalize(load.Arg(0))};
        address.offset += word * 4;
        if (!shared && address.base) {
            return {std::nullopt, false};
        }
        Origin result;
        bool initialized_here{false};
        const u32 width{shared ? 4U : 1U};
        for (const Write& write : writes) {
            if (write.shared != shared ||
                !MayOverlap(address, width, write.address, write.width, shared)) {
                continue;
            }
            if (write.address.base != address.base || write.address.offset != address.offset ||
                write.width != width || write.value.IsEmpty()) {
                if (dump_ir) {
                    LOG_INFO(Shader, "Memory pointer load {:x} has an overlapping write {:x}",
                             reinterpret_cast<uintptr_t>(&load),
                             reinterpret_cast<uintptr_t>(write.inst));
                }
                return {std::nullopt, false};
            }
            const Origin source{Find(write.value, bias, phase, depth + 1)};
            if (!source.valid || !source.buffer ||
                (result.buffer && result.buffer != source.buffer)) {
                if (dump_ir) {
                    LOG_INFO(Shader, "Memory pointer load {:x} has an unresolved/conflicting write {:x}",
                             reinterpret_cast<uintptr_t>(&load),
                             reinterpret_cast<uintptr_t>(write.inst));
                }
                return {std::nullopt, false};
            }
            result.buffer = source.buffer;
            initialized_here |= Dominates(write.inst, &load);
        }
        result.valid = initialized_here && result.buffer.has_value();
        if (dump_ir && !result.valid) {
            LOG_INFO(Shader, "Memory pointer load {:x} has no dominating initializer",
                     reinterpret_cast<uintptr_t>(&load));
        }
        origins[key] = result;
        return result;
    }

    static Origin ConstantBufferWord(const IR::Inst& inst, const Bias* bias, u32 phase,
                                     u32 extra = 0) {
        if (inst.Arg(0).IsImmediate() && inst.Arg(1).IsImmediate()) {
            const StorageBufferAddr buffer{inst.Arg(0).U32(), inst.Arg(1).U32() + extra};
            if (buffer.offset >= phase &&
                Common::IsAligned(buffer.offset - phase, bias ? bias->alignment : 8U) &&
                (!bias || MeetsBias({buffer.index, buffer.offset - phase}, *bias))) {
                return {buffer};
            }
        }
        return {};
    }

    Origin Find(IR::Value value, const Bias* bias, u32 phase, unsigned depth) {
        if (depth >= 64 || value.IsEmpty()) {
            return {std::nullopt, false};
        }
        value = value.Resolve();
        if (value.IsImmediate()) {
            return {};
        }
        const IR::Inst& inst{*value.Inst()};
        const IR::Opcode op{inst.GetOpcode()};
        if ((op == IR::Opcode::CompositeExtractU32x2 ||
             op == IR::Opcode::CompositeExtractU32x4) && inst.Arg(1).IsImmediate()) {
            const IR::Inst* vector{inst.Arg(0).TryInstRecursive()};
            const u32 word{inst.Arg(1).U32()};
            if (vector && (vector->GetOpcode() == IR::Opcode::LoadSharedU64 ||
                           vector->GetOpcode() == IR::Opcode::LoadSharedU128)) {
                return FromMemory(*vector, bias, phase, depth + 1, word);
            }
            if (vector && vector->GetOpcode() == IR::Opcode::GetCbufU32x2 && word < 2) {
                return ConstantBufferWord(*vector, bias, phase, word * 4);
            }
        }
        if (op == IR::Opcode::GetCbufU32) {
            return ConstantBufferWord(inst, bias, phase);
        }
        if (op == IR::Opcode::LoadSharedU32 || op == IR::Opcode::LoadLocal) {
            return FromMemory(inst, bias, phase, depth + 1);
        }
        if (op != IR::Opcode::IAdd32 && op != IR::Opcode::ISub32 &&
            op != IR::Opcode::SelectU32 && op != IR::Opcode::Phi) {
            return {};
        }
        Origin result;
        const bool merge{op == IR::Opcode::SelectU32 || op == IR::Opcode::Phi};
        bool first{true};
        for (size_t arg = op == IR::Opcode::SelectU32 ? 1 : 0; arg < inst.NumArgs(); ++arg) {
            const Origin source{Find(inst.Arg(arg), bias, phase, depth + 1)};
            if (!source.valid || (merge && !first && result.buffer != source.buffer) ||
                (result.buffer && source.buffer && result.buffer != source.buffer)) {
                return {std::nullopt, false};
            }
            if (source.buffer) {
                result.buffer = source.buffer;
            }
            first = false;
        }
        return result;
    }

    IR::Program& program;
    bool dump_ir;
    bool initialized{};
    std::vector<Write> writes;
    std::map<const IR::Inst*, std::pair<IR::Block*, size_t>> locations;
    std::map<std::pair<IR::Block*, IR::Block*>, bool> dominance;
    std::map<std::tuple<const IR::Inst*, bool, u32, u32>, Origin> origins;
};

IR::Value HighAddress(IR::Value address) {
    if (address.IsImmediate()) {
        return {};
    }
    IR::Inst* inst{address.InstRecursive()};
    if (inst->GetOpcode() == IR::Opcode::IAdd64) {
        if (!inst->Arg(1).IsImmediate()) {
            return {};
        }
        inst = inst->Arg(0).TryInstRecursive();
    }
    if (inst && inst->GetOpcode() == IR::Opcode::PackUint2x32) {
        inst = inst->Arg(0).TryInstRecursive();
    }
    return inst && inst->GetOpcode() == IR::Opcode::CompositeConstructU32x2
        ? inst->Arg(1) : IR::Value{};
}

/// Collects the storage buffer used by a global memory instruction and the instruction itself
void CollectStorageBuffers(IR::Block& block, IR::Inst& inst, StorageInfo& info, bool dump_ir,
                           MemoryPointerProvenance& memory) {
    // NVN puts storage buffers in a specific range, we have to bias towards these addresses to
    // avoid getting false positives
    static constexpr Bias nvn_bias{
        .index = 0,
        .offset_begin = 0x110,
        .offset_end = 0x610,
        .alignment = 16,
    };
    // Track the low address of the instruction
    const std::optional<LowAddrInfo> low_addr_info{TrackLowAddress(&inst)};
    if (!low_addr_info) {
        // Failed to track the low address, use NVN fallbacks
        return;
    }
    // First try to find storage buffers in the NVN address
    const IR::U32 low_addr{low_addr_info->value};
    std::optional<StorageBufferAddr> storage_buffer{Track(low_addr, &nvn_bias)};
    if (!storage_buffer) {
        storage_buffer = memory.TrackPointer(low_addr, HighAddress(inst.Arg(0)), &nvn_bias);
    }
    if (!storage_buffer) {
        // If it fails, track without a bias
        storage_buffer = Track(low_addr, nullptr);
        if (!storage_buffer) {
            storage_buffer = memory.TrackPointer(low_addr, HighAddress(inst.Arg(0)), nullptr);
        }
        if (!storage_buffer) {
            if (dump_ir) {
                LOG_INFO(Shader, "Untracked {} (IR node {:x}) after uniform lowering:\n{}",
                         IR::NameOf(inst.GetOpcode()), reinterpret_cast<uintptr_t>(&inst),
                         IR::DumpBlock(block));
            }
            // If that also fails, use NVN fallbacks
            LOG_WARNING(Shader, "Storage buffer failed to track, using global memory fallbacks");
            return;
        }
        LOG_INFO(Shader, "Storage buffer tracked without bias, index {} offset {}",
                 storage_buffer->index, storage_buffer->offset);
    }
    // Collect storage buffer and the instruction
    if (IsGlobalMemoryWrite(inst)) {
        info.writes.insert(*storage_buffer);
    }
    info.set.insert(*storage_buffer);
    info.to_replace.push_back(StorageInst{
        .storage_buffer{*storage_buffer},
        .inst = &inst,
        .block = &block,
    });
}

/// Returns the offset in indices (not bytes) for an equivalent storage instruction
IR::U32 StorageOffset(IR::Block& block, IR::Inst& inst, StorageBufferAddr buffer, u32 alignment) {
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    IR::U32 offset;
    if (const std::optional<LowAddrInfo> low_addr{TrackLowAddress(&inst)}) {
        offset = low_addr->value;
        if (low_addr->imm_offset != 0) {
            offset = ir.IAdd(offset, ir.Imm32(low_addr->imm_offset));
        }
    } else {
        offset = ir.UConvert(32, IR::U64{inst.Arg(0)});
    }
    // Subtract the least significant 32 bits from the guest offset. The result is the storage
    // buffer offset in bytes.
    IR::U32 low_cbuf{ir.GetCbuf(ir.Imm32(buffer.index), ir.Imm32(buffer.offset))};

    // Align the offset base to match the host alignment requirements
    low_cbuf = ir.BitwiseAnd(low_cbuf, ir.Imm32(~(alignment - 1U)));
    return ir.ISub(offset, low_cbuf);
}

/// Replace a global memory load instruction with its storage buffer equivalent
void ReplaceLoad(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                 const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::Value value{&*block.PrependNewInst(it, new_opcode, {storage_index, offset})};
    inst.ReplaceUsesWith(value);
}

/// Replace a global memory write instruction with its storage buffer equivalent
void ReplaceWrite(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                  const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    block.PrependNewInst(it, new_opcode, {storage_index, offset, inst.Arg(1)});
    inst.Invalidate();
}

/// Replace an atomic operation on global memory instruction with its storage buffer equivalent
void ReplaceAtomic(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                   const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::Value value{
        &*block.PrependNewInst(it, new_opcode, {storage_index, offset, inst.Arg(1)})};
    inst.ReplaceUsesWith(value);
}

/// Replace a global memory instruction with its storage buffer equivalent
void Replace(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
             const IR::U32& offset) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadGlobalS8:
    case IR::Opcode::LoadGlobalU8:
    case IR::Opcode::LoadGlobalS16:
    case IR::Opcode::LoadGlobalU16:
    case IR::Opcode::LoadGlobal32:
    case IR::Opcode::LoadGlobal64:
    case IR::Opcode::LoadGlobal128:
        return ReplaceLoad(block, inst, storage_index, offset);
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
        return ReplaceWrite(block, inst, storage_index, offset);
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return ReplaceAtomic(block, inst, storage_index, offset);
    default:
        throw InvalidArgument("Invalid global memory opcode {}", inst.GetOpcode());
    }
}

std::optional<StorageBufferAddr> CbufWord(const IR::Value& value) {
    if (value.IsImmediate()) {
        return std::nullopt;
    }
    const IR::Inst* source{value.InstRecursive()};
    u32 word{};
    if (source->GetOpcode() == IR::Opcode::CompositeExtractU32x2) {
        if (!source->Arg(1).IsImmediate() || source->Arg(1).U32() > 1 ||
            source->Arg(0).IsImmediate()) {
            return std::nullopt;
        }
        word = source->Arg(1).U32();
        source = source->Arg(0).InstRecursive();
        if (source->GetOpcode() != IR::Opcode::GetCbufU32x2) {
            return std::nullopt;
        }
    } else if (source->GetOpcode() != IR::Opcode::GetCbufU32) {
        return std::nullopt;
    }
    if (!source->Arg(0).IsImmediate() || !source->Arg(1).IsImmediate()) {
        return std::nullopt;
    }
    return StorageBufferAddr{source->Arg(0).U32(), source->Arg(1).U32() + word * 4};
}

void FoldNvnComputeUniformLoads(IR::Program& program) {
    if (program.stage != Stage::Compute) {
        return;
    }
    for (IR::Block* block : program.post_order_blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            u32 bits{};
            bool is_signed{};
            switch (inst.GetOpcode()) {
            case IR::Opcode::LoadGlobalS8: bits = 8; is_signed = true; break;
            case IR::Opcode::LoadGlobalU8: bits = 8; break;
            case IR::Opcode::LoadGlobalS16: bits = 16; is_signed = true; break;
            case IR::Opcode::LoadGlobalU16: bits = 16; break;
            case IR::Opcode::LoadGlobal32: bits = 32; break;
            case IR::Opcode::LoadGlobal64: bits = 64; break;
            case IR::Opcode::LoadGlobal128: bits = 128; break;
            default: continue;
            }
            const auto low{TrackLowAddress(&inst)};
            if (!low) {
                continue;
            }
            const auto base{CbufWord(low->value)};
            if (!base || base->index != 0 || base->offset < 0x210 ||
                base->offset >= 0x2f0 || (base->offset - 0x210) % 16 != 0) {
                continue;
            }
            const IR::Inst* address{inst.Arg(0).InstRecursive()};
            if (address->GetOpcode() == IR::Opcode::IAdd64) {
                address = address->Arg(0).InstRecursive();
            }
            if (address->GetOpcode() == IR::Opcode::PackUint2x32) {
                address = address->Arg(0).InstRecursive();
            }
            const auto high{CbufWord(address->Arg(1))};
            if (!high || high->index != 0 || high->offset != base->offset + 4) {
                continue;
            }
            if (low->imm_offset < 0 || static_cast<u32>(low->imm_offset) > 0x10000 - bits / 8) {
                throw NotImplementedException("NVN compute uniform read exceeds constant-buffer address space");
            }
            // Compute uniforms use c0's address table, not graphics' banked LDC.
            // Recover that bank before tracking pointers loaded from the uniform.
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            const auto bank{ir.Imm32(3 + (base->offset - 0x210) / 16)};
            const u32 offset{static_cast<u32>(low->imm_offset)};
            IR::Value value;
            if (bits == 128) {
                value = ir.CompositeConstruct(ir.GetCbuf(bank, ir.Imm32(offset)),
                    ir.GetCbuf(bank, ir.Imm32(offset + 4)),
                    ir.GetCbuf(bank, ir.Imm32(offset + 8)),
                    ir.GetCbuf(bank, ir.Imm32(offset + 12)));
            } else {
                value = ir.GetCbuf(bank, ir.Imm32(offset), bits, is_signed);
            }
            inst.ReplaceUsesWith(value);
        }
    }
}
} // Anonymous namespace

void GlobalMemoryToStorageBufferPass(IR::Program& program, const HostTranslateInfo& host_info) {
    FoldNvnComputeUniformLoads(program);
    if (host_info.dump_ir) {
        LOG_INFO(Shader, "Yuzu IR after compute uniform lowering:\n{}", IR::DumpProgram(program));
    }
    StorageInfo info;
    MemoryPointerProvenance memory{program, host_info.dump_ir};
    for (IR::Block* const block : program.post_order_blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (!IsGlobalMemory(inst)) {
                continue;
            }
            CollectStorageBuffers(*block, inst, info, host_info.dump_ir, memory);
        }
    }
    for (const StorageBufferAddr& storage_buffer : info.set) {
        program.info.storage_buffers_descriptors.push_back({
            .cbuf_index = storage_buffer.index,
            .cbuf_offset = storage_buffer.offset,
            .count = 1,
            .is_written = info.writes.contains(storage_buffer),
        });
    }
    for (const StorageInst& storage_inst : info.to_replace) {
        const StorageBufferAddr storage_buffer{storage_inst.storage_buffer};
        const auto it{info.set.find(storage_inst.storage_buffer)};
        const IR::U32 index{IR::Value{static_cast<u32>(info.set.index_of(it))}};
        IR::Block* const block{storage_inst.block};
        IR::Inst* const inst{storage_inst.inst};
        const IR::U32 offset{
            StorageOffset(*block, *inst, storage_buffer, host_info.min_ssbo_alignment)};
        Replace(*block, *inst, index, offset);
    }
}

template <typename Descriptors, typename Descriptor, typename Func>
static u32 Add(Descriptors& descriptors, const Descriptor& desc, Func&& pred) {
    // TODO: Handle arrays
    const auto it{std::ranges::find_if(descriptors, pred)};
    if (it != descriptors.end()) {
        return static_cast<u32>(std::distance(descriptors.begin(), it));
    }
    descriptors.push_back(desc);
    return static_cast<u32>(descriptors.size()) - 1;
}

void JoinStorageInfo(Info& base, Info& source) {
    auto& descriptors = base.storage_buffers_descriptors;
    for (auto& desc : source.storage_buffers_descriptors) {
        auto it{std::ranges::find_if(descriptors, [&desc](const auto& existing) {
            return desc.cbuf_index == existing.cbuf_index &&
                   desc.cbuf_offset == existing.cbuf_offset && desc.count == existing.count;
        })};
        if (it != descriptors.end()) {
            it->is_written |= desc.is_written;
            continue;
        }
        descriptors.push_back(desc);
    }
}

} // namespace Shader::Optimization
