#include <stdexcept>

#include "common/logging/log.h"

namespace Common::Log {

void FmtLogMessageImpl(Class, Level level, const char* filename, unsigned int line,
                       const char*, fmt::string_view format, const fmt::format_args& args) {
    const std::string message = fmt::vformat(format, args);
    fmt::print(stderr, "{}:{}: {}\n", filename, line, message);
    if (level >= Level::Warning) {
        throw std::runtime_error(message);
    }
}

} // namespace Common::Log

void assert_fail_impl() {
    throw std::runtime_error("Yuzu compiler assertion failed");
}

[[noreturn]] void unreachable_impl() {
    throw std::runtime_error("Yuzu compiler reached an unreachable path");
}
