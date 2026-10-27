#include <boost/json/src.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <cstdlib>

#include "pipeline.h"

namespace Totk {
namespace {
namespace json = boost::json;
namespace fs = std::filesystem;

void Write(const fs::path& path, const char* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    file.write(data, static_cast<std::streamsize>(size));
    file.close();
    if (!file) throw std::runtime_error("cannot write " + path.string());
}

void Write(const fs::path& path, const std::string& text) {
    Write(path, text.data(), text.size());
}

struct Transaction {
    fs::path directory;
    explicit Transaction(const fs::path& output) {
        std::string pattern = (output / ".yuzu-request-XXXXXX").string();
        if (!mkdtemp(pattern.data())) throw std::runtime_error("cannot create request staging directory");
        directory = pattern;
    }
    ~Transaction() {
        std::error_code error;
        fs::remove_all(directory, error);
        if (error) std::cerr << "cannot remove request staging directory: " << error.message() << '\n';
    }
};
} // namespace

void ValidateVariant(const std::string& variant) {
    static const std::regex safe{"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}"};
    if (!std::regex_match(variant, safe)) {
        throw std::runtime_error("variant must be a safe filename component");
    }
}

void InvalidateShader(const fs::path& output, const std::string& variant) {
    ValidateVariant(variant);
    fs::remove(output / (variant + "_b.json"));
}

ShaderArtifacts TranslateShader(const CompileRequest& input, const std::string& variant,
                                const std::string& chain, const json::object& attributes,
                                const json::object& colors, const std::string& c1,
                                const MetalCompiler& metal_compiler) {
    ValidateVariant(variant);
    if (chain.empty()) throw std::runtime_error("shader chain must not be empty");
    if (input.mode != CompileMode::Full && input.mode != CompileMode::OnlineFast)
        throw std::runtime_error("invalid compile mode");
    if (input.stage != "vs" && input.stage != "ps" && input.stage != "cs")
        throw std::runtime_error("stage must be vs, ps, or cs");
    if (input.code.empty() || input.code.size() > 16 * 1024 * 1024)
        throw std::runtime_error("invalid shader size");
    if (input.c1.size() > 4096) throw std::runtime_error("c1 must be at most 4096 bytes");
    if (input.stage == "cs") {
        if (!input.workgroup) throw std::runtime_error("compute shader without thread group size");
        for (const auto size : *input.workgroup)
            if (!size) throw std::runtime_error("invalid compute thread group size");
    }
    for (const auto& [slot, type] : input.textures) {
        if (slot >= 128) throw std::runtime_error("invalid texture slot");
        TextureType(type);
    }
    for (const auto& [slot, image] : input.images) {
        if (slot >= 9) throw std::runtime_error("invalid image slot");
        TextureType(image.type);
    }
    auto compiled = Compile(input);
    const auto& info = compiled.info;
    if (info.uses_global_memory)
        throw std::runtime_error("untracked global memory pointers are not supported");
    if (info.uses_subgroup_invocation_id || info.uses_subgroup_shuffles ||
        info.uses_subgroup_vote || info.uses_subgroup_mask || info.uses_fswzadd)
        throw std::runtime_error("subgroup compatibility is not validated");
    const auto abi = PrepareResourceAbi(input, compiled);
    const auto metal = metal_compiler ? metal_compiler(compiled, abi) :
        CompileMetal(compiled, abi, input.mode == CompileMode::Full);
    const auto& source = metal.source;
    size_t entry_position = std::string::npos;
    for (const auto* prefix : {"vertex ", "fragment ", "kernel "}) {
        const auto position = source.find(prefix);
        if (position < entry_position) entry_position = position;
    }
    if (entry_position == std::string::npos) throw std::runtime_error("cannot identify MSL entry point");
    const auto entry_line = source.substr(entry_position, source.find('\n', entry_position) - entry_position);
    std::smatch entry;
    static const std::regex entry_pattern{R"(\b(vertex|fragment|kernel)\s+\w+\s+(\w+)\s*\()"};
    if (!std::regex_search(entry_line, entry, entry_pattern))
        throw std::runtime_error("cannot identify MSL entry point");
    const std::string kind = input.stage == "vs" ? "vertex" : input.stage == "ps" ? "fragment" : "kernel";
    if (entry[1].str() != kind) throw std::runtime_error("MSL stage mismatch");
    json::array vertex_inputs;
    json::object attribute_deps;
    static const std::regex vertex_input{R"((\w+)\s+\w+\s*\[\[attribute\((\d+)\)\]\])"};
    if (input.stage == "vs") {
        size_t position = 0;
        while ((position = source.find("[[attribute(", position)) < entry_position) {
            const auto newline = source.rfind('\n', position);
            const auto start = newline == std::string::npos ? 0 : newline + 1;
            const auto end = source.find('\n', position);
            const auto line = source.substr(start, end - start);
            std::smatch match;
            if (!std::regex_search(line, match, vertex_input))
                throw std::runtime_error("cannot identify MSL vertex input");
            const std::string slot = match[2].str();
            const auto* type = attributes.if_contains(slot);
            vertex_inputs.push_back(json::object{
                {"location", std::stoul(slot)}, {"msl_type", match[1].str()}});
            attribute_deps[slot] = type ? *type : json::value("float");
            position = end == std::string::npos ? source.size() : end;
        }
    }
    json::object color_deps;
    if (input.stage == "ps") {
        for (size_t i = 0; i < compiled.info.stores_frag_color.size(); ++i) {
            if (!compiled.info.stores_frag_color[i]) continue;
            const auto slot = std::to_string(i);
            const auto* type = colors.if_contains(slot);
            color_deps[slot] = type ? *type : json::value("float");
        }
    }
    json::value workgroup;
    if (input.workgroup) {
        json::array sizes;
        for (const auto size : *input.workgroup) sizes.push_back(size);
        workgroup = std::move(sizes);
    }
    json::object reflection{
        {"key", variant + "_b"}, {"route", "B"}, {"chain", chain}, {"stage", input.stage},
        {"backend", "yuzu-cpp"}, {"compile_mode", CompileModeName(input.mode)},
        {"entry", entry[2].str()}, {"msl_entry_kind", kind},
        {"threadgroup_size", workgroup}, {"c1", c1}, {"notes", json::array{}},
        {"bindings", metal.bindings}, {"vertex_inputs", vertex_inputs}, {"raster", json::object{}},
        {"drv_reads", abi.driver_reads}, {"drv_unfillable", abi.unfillable},
        {"deps", json::object{{"tex", abi.texture_deps}, {"img", abi.image_deps},
                              {"attr", attribute_deps}, {"fout", color_deps}}},
        {"compare_slots", abi.compare_slots}};
    return {variant, std::move(compiled.words), source, json::serialize(reflection),
            json::serialize(json::object{{"backend", "yuzu"}, {"notes", json::array{}},
                                        {"bindings", metal.bindings}, {"drv_reads", abi.driver_reads}})};
}

void PublishShader(const fs::path& output, const ShaderArtifacts& artifacts) {
    ValidateVariant(artifacts.variant);
    fs::create_directories(output);
    InvalidateShader(output, artifacts.variant);
    Transaction transaction{output};
    const auto& variant = artifacts.variant;
    const auto base = variant + "_b";
    Write(transaction.directory / (variant + ".spv"),
          reinterpret_cast<const char*>(artifacts.spirv.data()), artifacts.spirv.size() * sizeof(u32));
    Write(transaction.directory / (variant + ".res.json"), artifacts.resources);
    Write(transaction.directory / (base + ".metal"), artifacts.metal);
    Write(transaction.directory / (base + ".json"), artifacts.reflection);
    for (const auto& file : {variant + ".spv", variant + ".res.json", base + ".metal", base + ".json"})
        fs::rename(transaction.directory / file, output / file);
}
} // namespace Totk
