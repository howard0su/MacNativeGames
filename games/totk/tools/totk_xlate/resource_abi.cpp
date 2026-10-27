#include "resource_abi.h"

#include <cstring>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "nvn_msl.h"

namespace Totk {
namespace {

namespace json = boost::json;

std::string Hex(u32 value) {
    std::ostringstream text;
    text << std::uppercase << std::hex << value;
    return text.str();
}

void One(u32 count) {
    if (count != 1) throw std::runtime_error("resource arrays need explicit argument-buffer array support");
}

void DebugName(std::vector<u32>& output, u32 opcode, std::vector<u32> args, const std::string& name) {
    const size_t start = args.size();
    args.resize(start + (name.size() + 4) / 4);
    std::memcpy(args.data() + start, name.c_str(), name.size() + 1);
    output.push_back((static_cast<u32>(args.size() + 1) << 16) | opcode);
    output.insert(output.end(), args.begin(), args.end());
}

void Rename(std::vector<u32>& words, const std::map<u32, std::string>& names,
            const std::set<u32>& buffers) {
    std::vector<u32> output(words.begin(), words.begin() + 5);
    bool inserted = false;
    for (size_t i = 5; i < words.size();) {
        const u32 count = words[i] >> 16, opcode = words[i] & 0xffff;
        if (!count || count > words.size() - i) throw std::runtime_error("invalid generated SPIR-V");
        if (!inserted && opcode == spv::OpDecorate) {
            for (const auto& [id, name] : names) DebugName(output, spv::OpName, {id}, name);
            for (u32 id : buffers) DebugName(output, spv::OpMemberName, {id, 0}, "_m0");
            inserted = true;
        }
        const bool replacing_name = opcode == spv::OpName && names.contains(words.at(i + 1));
        const bool replacing_member = opcode == spv::OpMemberName &&
                                      buffers.contains(words.at(i + 1)) && words.at(i + 2) == 0;
        if (!replacing_name && !replacing_member) {
            output.insert(output.end(), words.begin() + i, words.begin() + i + count);
        }
        i += count;
    }
    if ((!names.empty() || !buffers.empty()) && !inserted) {
        throw std::runtime_error("generated SPIR-V has no resource annotation section");
    }
    words = std::move(output);
}

template <typename Descriptor>
u32 SamplerRegister(const Descriptor& desc) {
    const auto slot = [](u32 bank, u32 offset) -> u32 {
        if (bank != 2 || offset < 0x568 || offset >= 0x968 || (offset - 0x568) % 8) {
            throw std::runtime_error("separate sampler handle has no supported NVN slot");
        }
        return (offset - 0x568) / 8;
    };
    if (desc.cbuf_index == 2 && desc.cbuf_offset >= 0x568) {
        return slot(desc.cbuf_index, desc.cbuf_offset);
    }
    return slot(desc.secondary_cbuf_index, desc.secondary_cbuf_offset);
}

template <typename Descriptor>
std::string TextureName(const std::string& prefix, const Descriptor& desc) {
    if (desc.cbuf_offset % 4 || desc.cbuf_offset / 4 > 0x3fff) {
        throw std::runtime_error("unaligned or oversized texture handle offset");
    }
    u32 handle = desc.cbuf_offset / 4;
    if (!desc.has_secondary) {
        return prefix + "_t_cb" + std::to_string(desc.cbuf_index) + "_" + Hex(handle);
    }
    if (desc.secondary_cbuf_offset % 4 || desc.secondary_cbuf_offset / 4 > 0x3fff ||
        desc.cbuf_index > 0xffff || desc.secondary_cbuf_index >= 0xffff) {
        throw std::runtime_error("unsupported secondary texture handle offset");
    }
    handle |= (desc.secondary_cbuf_offset / 4) << 14;
    handle |= 1u << 28;
    const u32 banks = desc.cbuf_index | ((desc.secondary_cbuf_index + 1) << 16);
    return prefix + "_t_cb" + std::to_string(banks) + "_" + Hex(handle);
}

bool Fillable(u32 bank, u32 offset, const std::string& stage) {
    if (bank == 2) {
        return (offset >= 0x20 && offset < 0x120) || (offset >= 0x168 && offset < 0x968);
    }
    if (stage == "cs") return offset >= 0x200 && offset < 0x410;
    if (stage == "ps") return offset >= 0x510 && offset < 0x610;
    return (offset >= 0x110 && offset < 0x210) || (offset >= 0x610 && offset < 0x620);
}

struct MetalField {
    std::string name;
    std::string type;
};

std::map<u32, MetalField> MetalFields(const std::string& source) {
    static const std::regex declaration{R"(\s*([^\n;{}]+?)\s+(\w+)\s*\[\[id\((\d+)\)\]\];)"};
    std::map<u32, MetalField> fields;
    size_t position = 0;
    while ((position = source.find("[[id(", position)) != std::string::npos) {
        const auto newline = source.rfind('\n', position);
        const auto start = newline == std::string::npos ? 0 : newline + 1;
        const auto end = source.find('\n', position);
        const auto line = source.substr(start, end - start);
        std::smatch match;
        if (!std::regex_match(line, match, declaration)) {
            throw std::runtime_error("cannot parse SPIRV-Cross resource declaration");
        }
        const auto argument = std::stoul(match[3].str());
        if (argument > UINT32_MAX ||
            !fields.emplace(static_cast<u32>(argument),
                            MetalField{match[2].str(), match[1].str()}).second) {
            throw std::runtime_error("invalid or duplicate SPIRV-Cross resource argument ID");
        }
        position = end == std::string::npos ? source.size() : end;
    }
    return fields;
}

const std::string& FieldType(const std::map<u32, MetalField>& fields,
                             const std::string& name, u32 argument) {
    const auto found = fields.find(argument);
    if (found == fields.end() || found->second.name != name) {
        throw std::runtime_error("SPIRV-Cross resource binding is absent from final MSL: " + name);
    }
    return found->second.type;
}

} // namespace

ResourceAbi PrepareResourceAbi(const CompileRequest& request, CompileResult& compiled) {
    ResourceAbi abi;
    const auto& info = compiled.info;
    if (info.uses_render_area || info.uses_rescaling_uniform) {
        throw std::runtime_error("shader requires unsupported emulator support data");
    }
    const std::string prefix = request.stage == "vs" ? "vp" : request.stage == "ps" ? "fp" : "cp";
    std::map<u32, json::object> logical;
    std::map<u32, json::object> samplers;
    u32 binding = 0;
    for (const auto& desc : info.constant_buffer_descriptors) {
        One(desc.count);
        json::object item{{"msl_name", prefix + "_c" + std::to_string(desc.index)},
                          {"required_bytes", info.constant_buffer_used_sizes.at(desc.index)}};
        if (desc.index >= 3 && desc.index <= 16) {
            item["kind"] = "cbuf";
            item["reg"] = desc.index - 3;
        } else if (desc.index == 1) {
            item["kind"] = "c1";
            item["bank"] = 1;
        } else if (desc.index == 0 || desc.index == 2) {
            item["kind"] = "drvcb";
            item["bank"] = desc.index;
        } else {
            throw std::runtime_error("constant buffer has no supported NVN binding");
        }
        logical[binding++] = item;
    }
    size_t storage_index = 0;
    for (const auto& desc : info.storage_buffers_descriptors) {
        One(desc.count);
        json::object item{{"msl_name", prefix + "_s" + std::to_string(storage_index++)},
                          {"rw", desc.is_written}};
        const u32 base = request.stage == "vs" ? 0x110 : request.stage == "ps" ? 0x510 : 0x310;
        if (desc.cbuf_index >= 3 && desc.cbuf_index <= 16) {
            item["kind"] = "sbuf_ubo";
            item["reg"] = desc.cbuf_index - 3;
            item["off"] = desc.cbuf_offset;
        } else if (desc.cbuf_index == 0 && request.stage == "cs" &&
                   desc.cbuf_offset >= 0x210 && desc.cbuf_offset < 0x310 &&
                   (desc.cbuf_offset - 0x210) % 16 == 0) {
            item["kind"] = "ubo_ptr";
            item["reg"] = (desc.cbuf_offset - 0x210) / 16;
        } else if (desc.cbuf_index == 0 && desc.cbuf_offset >= base &&
                   desc.cbuf_offset - base < 0x100 && (desc.cbuf_offset - base) % 16 == 0) {
            item["kind"] = "sbuf";
            item["reg"] = (desc.cbuf_offset - base) / 16;
        } else {
            throw std::runtime_error("storage buffer pointer has no supported NVN slot");
        }
        logical[binding++] = item;
    }
    const auto texture_dep = [&](u32 slot) {
        const auto found = request.textures.find(slot);
        const std::string answer = found == request.textures.end() ? "2d" : found->second;
        if (compiled.texture_queries.contains(slot)) {
            abi.texture_deps[std::to_string(slot)] = answer;
        } else {
            abi.texture_deps[std::to_string(slot)] = answer.ends_with("_depth") ? "*_depth" : "*";
        }
    };
    const auto image_dep = [&](u32 slot) {
        const auto found = request.images.find(slot);
        const auto state = found == request.images.end() ? ImageState{} : found->second;
        abi.image_deps[std::to_string(slot)] = json::object{{"type", state.type}, {"fmt", state.format}};
    };
    for (const auto& desc : info.texture_buffer_descriptors) {
        One(desc.count);
        const u32 slot = TextureRegister(desc);
        logical[binding++] = {{"kind", "tbuf"}, {"reg", slot}, {"msl_name", TextureName(prefix, desc)}};
        texture_dep(slot);
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        One(desc.count);
        const auto slot = ImageSlot(desc.cbuf_index, desc.cbuf_offset);
        logical[binding++] = {{"kind", "img"}, {"reg", slot}, {"rw", desc.is_written},
                              {"msl_name", prefix + "_i_cb" + std::to_string(desc.cbuf_index) +
                                           "_" + Hex(desc.cbuf_offset / 4)}};
        image_dep(slot);
    }
    std::set<u32> compare_slots;
    for (const auto& desc : info.texture_descriptors) {
        One(desc.count);
        const u32 slot = TextureRegister(desc);
        const std::string name = TextureName(prefix, desc);
        logical[binding] = {{"kind", "tex"}, {"reg", slot}, {"msl_name", name}};
        samplers[binding] = {{"kind", desc.has_secondary ? "samp" : "samp_tex"},
                            {"reg", desc.has_secondary ? SamplerRegister(desc) : slot},
                            {"msl_name", name + "Smplr"}};
        if (desc.is_depth) compare_slots.insert(slot);
        ++binding;
        texture_dep(slot);
    }
    for (const auto& desc : info.image_descriptors) {
        One(desc.count);
        const auto slot = ImageSlot(desc.cbuf_index, desc.cbuf_offset);
        logical[binding++] = {{"kind", "img"}, {"reg", slot}, {"rw", desc.is_written},
                              {"msl_name", prefix + "_i_cb" + std::to_string(desc.cbuf_index) +
                                           "_" + Hex(desc.cbuf_offset / 4)}};
        image_dep(slot);
    }
    for (u32 slot : compare_slots) abi.compare_slots.push_back(slot);
    for (u32 slot : compiled.texture_queries) texture_dep(slot);
    for (u32 slot : compiled.image_queries) image_dep(slot);
    for (const auto& [bank, reads] : compiled.constant_reads) {
        if (bank != 0 && bank != 2) continue;
        json::array offsets;
        for (u32 offset : reads.vec4_offsets) {
            offsets.push_back(offset);
        }
        for (const auto& [offset, width] : reads.byte_ranges) {
            for (u32 byte = offset; byte - offset < width; ++byte) {
                if (!Fillable(bank, byte, request.stage)) {
                    abi.unfillable.emplace_back("c" + std::to_string(bank) + "+0x" + Hex(byte));
                    break;
                }
            }
        }
        if (reads.dynamic) abi.unfillable.emplace_back("c" + std::to_string(bank) + " dynamic index");
        abi.driver_reads[std::to_string(bank)] = json::object{
            {"vec4_offsets", offsets}, {"dynamic", reads.dynamic}};
    }
    if (!abi.unfillable.empty()) {
        throw std::runtime_error("shader reads unfillable driver data: " + json::serialize(abi.unfillable));
    }
    spirv_cross::Compiler compiler{compiled.words};
    const auto resources = compiler.get_shader_resources();
    std::map<u32, std::string> names;
    std::set<u32> buffers;
    const auto collect = [&](const auto& list, bool buffer) {
        for (const auto& resource : list) {
            if (compiler.get_decoration(resource.id, spv::DecorationDescriptorSet) != 0) {
                throw std::runtime_error("generated resource is outside descriptor set zero");
            }
            const u32 slot = compiler.get_decoration(resource.id, spv::DecorationBinding);
            const auto found = logical.find(slot);
            if (found == logical.end()) throw std::runtime_error("generated resource has no logical descriptor");
            abi.resources[resource.id] = found->second;
            names[resource.id] = std::string(found->second.at("msl_name").as_string());
            if (buffer) buffers.insert(resource.base_type_id);
            if (samplers.contains(slot)) abi.samplers[resource.id] = samplers.at(slot);
        }
    };
    collect(resources.uniform_buffers, true);
    collect(resources.storage_buffers, true);
    collect(resources.sampled_images, false);
    collect(resources.separate_images, false);
    collect(resources.storage_images, false);
    if (abi.resources.size() != logical.size() || !resources.push_constant_buffers.empty() ||
        !resources.separate_samplers.empty() || !resources.subpass_inputs.empty() ||
        !resources.atomic_counters.empty()) {
        throw std::runtime_error("generated SPIR-V resources do not match compiler descriptors");
    }
    Rename(compiled.words, names, buffers);
    return abi;
}

MetalResult CompileMetal(const CompileResult& compiled, const ResourceAbi& abi,
                         bool verify_bindings) {
    spirv_cross::NvnCompilerMSL compiler{compiled.words};
    auto options = compiler.get_msl_options();
    options.msl_version = 20300;
    options.argument_buffers = true;
    options.texture_buffer_native = true;
    options.pad_fragment_output_components = true;
    compiler.set_msl_options(options);
    MetalResult result{compiler.compile(), {}};
    const auto fields = MetalFields(result.source);
    const auto add = [&](u32 id, const json::object& resource, bool sampler) {
        const u32 argument = sampler ? compiler.get_automatic_msl_resource_binding_secondary(id) :
                                       compiler.get_automatic_msl_resource_binding(id);
        if (argument == UINT32_MAX) return;
        auto item = resource;
        const auto name = compiler.get_name(id) + (sampler ? "Smplr" : "");
        item["msl_name"] = name;
        item["arg_id"] = argument;
        item["msl_type"] = FieldType(fields, name, argument);
        result.bindings.push_back(std::move(item));
    };
    for (const auto& [id, resource] : abi.resources) add(id, resource, false);
    for (const auto& [id, resource] : abi.samplers) add(id, resource, true);
    if (result.bindings.size() > 64) throw std::runtime_error("shader exceeds native reflection binding capacity");
    if (verify_bindings) CheckMetalBindings(result.source, result.bindings);
    return result;
}

void CheckMetalBindings(const std::string& source, const json::array& bindings) {
    const auto fields = MetalFields(source);
    for (const auto& value : bindings) {
        const auto& item = value.as_object();
        const auto type = FieldType(fields, std::string(item.at("msl_name").as_string()),
                                    static_cast<u32>(item.at("arg_id").as_uint64()));
        if (type != item.at("msl_type").as_string()) {
            throw std::runtime_error("SPIRV-Cross resource type disagrees with native reflection");
        }
    }
}

} // namespace Totk
