#pragma once

#include <filesystem>
#include <functional>
#include "resource_abi.h"

namespace Totk {

struct ShaderArtifacts {
    std::string variant;
    std::vector<u32> spirv;
    std::string metal;
    std::string reflection;
    std::string resources;
};

using MetalCompiler = std::function<MetalResult(const CompileResult&, const ResourceAbi&)>;

void ValidateVariant(const std::string& variant);
void InvalidateShader(const std::filesystem::path& output, const std::string& variant);
ShaderArtifacts TranslateShader(const CompileRequest& input, const std::string& variant,
                                const std::string& chain, const boost::json::object& attributes,
                                const boost::json::object& colors, const std::string& c1,
                                const MetalCompiler& metal_compiler = {});
void PublishShader(const std::filesystem::path& output, const ShaderArtifacts& artifacts);

} // namespace Totk
