#include <boost/json.hpp>

#include <charconv>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#include "compiler.h"
#include "resource_abi.h"
#include "pipeline.h"

extern char** environ;

namespace {

namespace json = boost::json;
namespace fs = std::filesystem;

struct Options {
    std::string cross;
    fs::path output;
    std::string chain;
    Totk::CompileMode mode{Totk::CompileMode::Full};
};

std::string String(const json::value& value) {
    if (!value.is_string()) {
        throw std::runtime_error("expected a JSON string");
    }
    return std::string(value.as_string());
}

bool Flag(const json::object& request, const char* name) {
    const auto* value = request.if_contains(name);
    if (!value || value->is_null()) {
        return false;
    }
    if (value->is_bool()) {
        return value->as_bool();
    }
    if (value->is_int64() && (value->as_int64() == 0 || value->as_int64() == 1)) {
        return value->as_int64() == 1;
    }
    throw std::runtime_error(std::string(name) + " must be a boolean or 0/1");
}

json::object Object(const json::object& request, const char* name) {
    const auto* value = request.if_contains(name);
    if (!value || value->is_null()) {
        return {};
    }
    if (!value->is_object()) {
        throw std::runtime_error(std::string(name) + " must be an object");
    }
    return value->as_object();
}

template <size_t N>
void Types(const json::object& source, std::array<Shader::AttributeType, N>& output) {
    for (const auto& [key, value] : source) {
        unsigned slot{};
        const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), slot);
        if (error != std::errc{} || end != key.data() + key.size() || slot >= N ||
            key != std::to_string(slot)) {
            throw std::runtime_error("invalid attribute or fragment output slot");
        }
        const auto type = String(value);
        if (type == "float") {
            output[slot] = Shader::AttributeType::Float;
        } else if (type == "sint") {
            output[slot] = Shader::AttributeType::SignedInt;
        } else if (type == "uint") {
            output[slot] = Shader::AttributeType::UnsignedInt;
        } else {
            throw std::runtime_error("type must be float, sint, or uint");
        }
    }
}

u32 Slot(std::string_view key, u32 count) {
    u32 slot{};
    const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), slot);
    if (error != std::errc{} || end != key.data() + key.size() || slot >= count ||
        key != std::to_string(slot)) {
        throw std::runtime_error("invalid resource slot");
    }
    return slot;
}

u32 Size(const json::value& value) {
    if (!value.is_int64() || value.as_int64() < 0 || value.as_int64() > UINT32_MAX) {
        throw std::runtime_error("memory size must be an unsigned 32-bit byte count");
    }
    return static_cast<u32>(value.as_int64());
}

std::vector<unsigned char> Hex(const std::string& text) {
    if (text.size() % 2 || text.size() > 8192) {
        throw std::runtime_error("c1 must be paired hex bytes, at most 4096 bytes");
    }
    const auto nibble = [](char c) -> unsigned {
        if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return static_cast<unsigned>(c - 'A' + 10);
        throw std::runtime_error("c1 contains a non-hex digit");
    };
    std::vector<unsigned char> bytes;
    for (size_t i = 0; i < text.size(); i += 2) {
        bytes.push_back(static_cast<unsigned char>(nibble(text[i]) * 16 + nibble(text[i + 1])));
    }
    return bytes;
}

void Write(const fs::path& path, const char* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    file.write(data, static_cast<std::streamsize>(size));
    file.close();
    if (!file) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

void Write(const fs::path& path, const std::string& text) {
    Write(path, text.data(), text.size());
}

struct Transaction {
    fs::path directory;

    explicit Transaction(const fs::path& output) {
        std::string pattern = (output / ".yuzu-request-XXXXXX").string();
        if (!mkdtemp(pattern.data())) {
            throw std::runtime_error("cannot create request staging directory");
        }
        directory = pattern;
    }
    ~Transaction() {
        std::error_code error;
        fs::remove_all(directory, error);
        if (error) {
            std::cerr << "cannot remove request staging directory: " << error.message() << '\n';
        }
    }
};

Totk::MetalResult Msl(const Options& options, const Totk::CompileResult& result,
                     const Totk::ResourceAbi& abi, const fs::path& spirv, const fs::path& metal) {
    auto compiled = Totk::CompileMetal(result, abi, options.mode == Totk::CompileMode::Full);
    const char* mode = std::getenv("TOTK_XLATE_SPIRV_CROSS");
    if (mode && std::string_view(mode) == "process") {
        std::vector<std::string> args{
            options.cross, spirv.string(), "--msl", "--msl-version", "20300",
            "--msl-argument-buffers", "--msl-texture-buffer-native",
            "--msl-pad-fragment-output", "--output", metal.string()};
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        posix_spawn_file_actions_t actions;
        int error = posix_spawn_file_actions_init(&actions);
        if (error) throw std::runtime_error("cannot initialize SPIRV-Cross process");
        error = posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDOUT_FILENO);
        pid_t child{};
        if (!error) {
            error = posix_spawnp(&child, argv[0], &actions, nullptr, argv.data(), environ);
        }
        const int destroyed = posix_spawn_file_actions_destroy(&actions);
        if (error) throw std::runtime_error("cannot launch SPIRV-Cross: " + std::to_string(error));
        int status{};
        pid_t waited;
        do {
            waited = waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (destroyed || waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status)) {
            throw std::runtime_error("SPIRV-Cross process failed");
        }
        std::ifstream file(metal, std::ios::binary);
        if (!file) throw std::runtime_error("SPIRV-Cross produced no MSL");
        std::string source{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        if (file.bad() || source.empty()) throw std::runtime_error("cannot read SPIRV-Cross MSL");
        Totk::CheckMetalBindings(source, compiled.bindings);
        compiled.source = std::move(source);
        return compiled;
    }
    Write(metal, compiled.source);
    return compiled;
}

void Handle(const Options& options, const json::object& request) {
    const auto variant = String(request.at("variant"));
    Totk::ValidateVariant(variant);
    const std::string base = variant + "_b";
    // Reflection is the commit marker: a failed retry must not expose stale success.
    Totk::InvalidateShader(options.output, variant);
    Totk::CompileRequest input;
    input.mode = options.mode;
    input.stage = String(request.at("stage"));
    if (input.stage != "vs" && input.stage != "ps" && input.stage != "cs") {
        throw std::runtime_error("stage must be vs, ps, or cs");
    }
    const auto* c1_value = request.if_contains("c1");
    const std::string c1 = c1_value ? String(*c1_value) : "";
    input.c1 = Hex(c1);
    if (const auto* size = request.if_contains("c1_buffer_size")) input.c1_buffer_size = Size(*size);
    const auto attributes = Object(request, "attr");
    const auto colors = Object(request, "fout");
    Types(attributes, input.attributes);
    Types(colors, input.colors);
    for (const auto& [key, value] : Object(request, "tex")) {
        const auto type = String(value);
        Totk::TextureType(type);
        input.textures[Slot(key, 128)] = type;
    }
    for (const auto& [key, value] : Object(request, "img")) {
        const auto& image = value.as_object();
        Totk::ImageState state;
        if (const auto* type = image.if_contains("type")) state.type = String(*type);
        if (const auto* format = image.if_contains("fmt")) state.format = String(*format);
        Totk::TextureType(state.type);
        input.images[Slot(key, 9)] = state;
    }
    input.depth_minus_w = Flag(request, "depth_minus_w");
    input.early_fragment_tests = Flag(request, "early_fragment_tests");
    if (request.if_contains("entry_point") && input.stage != "cs") {
        throw std::runtime_error("compute entry point is only valid for cs");
    }
    if (input.stage == "cs") {
        const auto* local = request.if_contains("local");
        if (!local || !local->is_array() || local->as_array().size() != 3) {
            throw std::runtime_error("compute shader without thread group size");
        }
        std::array<u32, 3> workgroup;
        size_t component = 0;
        for (const auto& size : local->as_array()) {
            if (!size.is_int64() || size.as_int64() <= 0 || size.as_int64() > UINT32_MAX) {
                throw std::runtime_error("invalid compute thread group size");
            }
            workgroup[component++] = static_cast<u32>(size.as_int64());
        }
        input.workgroup = workgroup;
        if (const auto* entry = request.if_contains("entry_point")) input.entry_point = Size(*entry);
        if (const auto* size = request.if_contains("local_memory")) input.local_memory = Size(*size);
        if (const auto* size = request.if_contains("shared_memory")) input.shared_memory = Size(*size);
    }
    input.code = Totk::ReadShader(String(request.at("code")));
    Totk::MetalCompiler override;
    const char* cross_mode = std::getenv("TOTK_XLATE_SPIRV_CROSS");
    std::unique_ptr<Transaction> comparison;
    if (cross_mode && std::string_view(cross_mode) == "process") {
        comparison = std::make_unique<Transaction>(options.output);
        override = [&](const Totk::CompileResult& compiled, const Totk::ResourceAbi& abi) {
            const auto spirv = comparison->directory / (variant + ".spv");
            const auto metal = comparison->directory / (base + ".metal");
            Write(spirv, reinterpret_cast<const char*>(compiled.words.data()),
                  compiled.words.size() * sizeof(u32));
            return Msl(options, compiled, abi, spirv, metal);
        };
    }
    Totk::PublishShader(options.output, Totk::TranslateShader(
        input, variant, options.chain, attributes, colors, c1, override));
}

Options Parse(int argc, char** argv) {
    if (argc < 2 || std::string_view(argv[1]) != "--service") {
        throw std::runtime_error("expected --service");
    }
    Options options;
    std::string debug = std::getenv("TOTK_SHADER_DEBUGMODE") ?
                        std::getenv("TOTK_SHADER_DEBUGMODE") : "1";
    std::string mode = std::getenv("TOTK_SHADER_COMPILE_MODE") ?
                       std::getenv("TOTK_SHADER_COMPILE_MODE") : "full";
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 == argc) throw std::runtime_error("missing CLI option value");
        const std::string_view key{argv[i]};
        if (key == "--spirv-cross") options.cross = argv[i + 1];
        else if (key == "--out") options.output = argv[i + 1];
        else if (key == "--chain") options.chain = argv[i + 1];
        else if (key == "--debug-mode") debug = argv[i + 1];
        else if (key == "--compile-mode") mode = argv[i + 1];
        else throw std::runtime_error("unknown CLI option");
    }
    if (options.cross.empty() || options.output.empty() || options.chain.empty() ||
        (debug != "0" && debug != "1") || (mode != "full" && mode != "online-fast")) {
        throw std::runtime_error("require --spirv-cross PATH --out DIR --chain NAME "
                                 "[--debug-mode 0|1] [--compile-mode full|online-fast]");
    }
    options.mode = mode == "full" ? Totk::CompileMode::Full : Totk::CompileMode::OnlineFast;
    return options;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--identity") {
        std::cout << TOTK_COMPILER_IDENTITY << '\n';
        return 0;
    }
    Options options;
    try {
        options = Parse(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    try {
        fs::create_directories(options.output);
        std::cerr << "Yuzu shader service: compute requires explicit memory sizes; "
                     "debug-mode does not select Ryujinx optimizer behavior\n";
        std::cerr << "compile-mode=" << Totk::CompileModeName(options.mode)
                  << "; IR verification=" << (options.mode == Totk::CompileMode::Full ? "on" : "off")
                  << "; GPU loop watchdog=on\n";
        std::string line;
        size_t sequence = 0;
        while (std::getline(std::cin, line)) {
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            ++sequence;
            std::cerr << "[request " << sequence << "] begin\n";
            json::object reply;
            try {
                const auto value = json::parse(line);
                if (value.is_object()) {
                    if (const auto* variant = value.as_object().if_contains("variant")) {
                        std::cerr << "[request " << sequence << "] variant="
                                  << json::serialize(*variant).substr(0, 500) << '\n';
                    }
                }
                Handle(options, value.as_object());
                reply["ok"] = true;
                std::cerr << "[request " << sequence << "] ok\n";
            } catch (const std::exception& error) {
                std::cerr << "[request " << sequence << "] failed: " << error.what() << '\n';
                reply = {{"ok", false}, {"error", error.what()}};
            }
            std::cout << json::serialize(reply) << '\n' << std::flush;
            if (!std::cout) throw std::runtime_error("cannot write service reply");
        }
        if (std::cin.bad()) throw std::runtime_error("cannot read service request");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
