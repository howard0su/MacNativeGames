#pragma once

#include <boost/json.hpp>
#include <map>

#include "compiler.h"

namespace Totk {

struct ResourceAbi {
    std::map<u32, boost::json::object> resources;
    std::map<u32, boost::json::object> samplers;
    boost::json::object driver_reads;
    boost::json::array unfillable;
    boost::json::object texture_deps;
    boost::json::object image_deps;
    boost::json::array compare_slots;
};

struct MetalResult {
    std::string source;
    boost::json::array bindings;
};

ResourceAbi PrepareResourceAbi(const CompileRequest& request, CompileResult& compiled);
MetalResult CompileMetal(const CompileResult& compiled, const ResourceAbi& abi,
                         bool verify_bindings = true);
void CheckMetalBindings(const std::string& source, const boost::json::array& bindings);

} // namespace Totk
