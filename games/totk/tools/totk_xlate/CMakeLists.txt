cmake_minimum_required(VERSION 3.20)
project(totk_xlate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(fmt 9 REQUIRED)
find_package(Boost 1.75 REQUIRED CONFIG)

get_filename_component(DEFAULT_REPOSITORY "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(REPOSITORY "${DEFAULT_REPOSITORY}" CACHE PATH "MacNativeGames repository (also for copied work trees)")
set(YUZU "${REPOSITORY}/vendor/yuzu")
set(SIRIT "${YUZU}/externals/sirit")

file(GLOB_RECURSE SIRIT_SOURCES CONFIGURE_DEPENDS "${SIRIT}/src/*.cpp")
add_library(totk_sirit OBJECT ${SIRIT_SOURCES})
target_include_directories(totk_sirit PUBLIC
    "${SIRIT}/include"
    "${SIRIT}/externals/SPIRV-Headers/include"
    PRIVATE "${SIRIT}/src")

file(GLOB_RECURSE COMPILER_SOURCES CONFIGURE_DEPENDS
    "${YUZU}/src/shader_recompiler/frontend/*.cpp"
    "${YUZU}/src/shader_recompiler/ir_opt/*.cpp"
    "${YUZU}/src/shader_recompiler/backend/spirv/*.cpp")
add_library(totk_yuzu_compiler OBJECT ${COMPILER_SOURCES} diagnostics.cpp)
target_include_directories(totk_yuzu_compiler PUBLIC "${YUZU}/src")
target_link_libraries(totk_yuzu_compiler PUBLIC totk_sirit fmt::fmt-header-only Boost::headers)
target_compile_options(totk_yuzu_compiler PRIVATE
    "$<$<CXX_COMPILER_ID:Clang,AppleClang>:-fbracket-depth=1024>")

add_library(totk_yuzu_frontend OBJECT compiler.cpp)
target_link_libraries(totk_yuzu_frontend PUBLIC totk_yuzu_compiler)
target_include_directories(totk_yuzu_frontend PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(totk_yuzu_frontend PRIVATE -Wall -Wextra -Werror)

add_executable(totk_yuzu_probe compiler_probe.cpp)
target_link_libraries(totk_yuzu_probe PRIVATE totk_yuzu_frontend totk_yuzu_compiler totk_sirit)
target_compile_options(totk_yuzu_probe PRIVATE -Wall -Wextra -Werror)

set(SPIRV_CROSS_SRC "" CACHE PATH "Checkout of the SPIRV-Cross commit pinned in build_app.py")
if(SPIRV_CROSS_SRC)
    find_package(Git REQUIRED)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${SPIRV_CROSS_SRC}" rev-parse HEAD
        OUTPUT_VARIABLE CROSS_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    file(STRINGS "${REPOSITORY}/build_app.py" CROSS_PIN_LINE REGEX "^SPIRV_CROSS = ")
    string(REGEX REPLACE ".*'([0-9a-f]+)'\\)$" "\\1" CROSS_PIN "${CROSS_PIN_LINE}")
    if(NOT CROSS_REVISION STREQUAL CROSS_PIN)
        message(FATAL_ERROR "SPIRV_CROSS_SRC must be the commit pinned in build_app.py")
    endif()
    set(CROSS_SOURCES)
    foreach(NAME spirv_cfg spirv_cpp spirv_cross spirv_cross_parsed_ir spirv_cross_util
                 spirv_glsl spirv_hlsl spirv_msl spirv_parser spirv_reflect)
        list(APPEND CROSS_SOURCES "${SPIRV_CROSS_SRC}/${NAME}.cpp")
    endforeach()
    add_library(totk_spirv_cross_core OBJECT ${CROSS_SOURCES})
    target_include_directories(totk_spirv_cross_core PUBLIC "${SPIRV_CROSS_SRC}")
    file(GLOB_RECURSE IDENTITY_SOURCES CONFIGURE_DEPENDS
        "${YUZU}/src/*" "${SIRIT}/src/*" "${SIRIT}/include/*"
        "${SIRIT}/externals/SPIRV-Headers/include/*"
        "${SPIRV_CROSS_SRC}/*.cpp" "${SPIRV_CROSS_SRC}/*.hpp" "${SPIRV_CROSS_SRC}/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
    list(APPEND IDENTITY_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
    list(SORT IDENTITY_SOURCES)
    set(IDENTITY "${CROSS_REVISION};${CMAKE_SYSTEM_NAME};${CMAKE_OSX_SYSROOT};${CMAKE_OSX_ARCHITECTURES};${CMAKE_OSX_DEPLOYMENT_TARGET};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_BUILD_TYPE};${CMAKE_CXX_FLAGS};${CMAKE_CXX_FLAGS_RELEASE};${fmt_VERSION};${Boost_VERSION}")
    foreach(SOURCE IN LISTS IDENTITY_SOURCES)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SOURCE}")
        file(SHA256 "${SOURCE}" DIGEST)
        string(APPEND IDENTITY ";${DIGEST}")
    endforeach()
    string(SHA256 COMPILER_IDENTITY "${IDENTITY}")
    add_library(totk_shader_pipeline STATIC pipeline.cpp resource_abi.cpp
        $<TARGET_OBJECTS:totk_yuzu_frontend> $<TARGET_OBJECTS:totk_yuzu_compiler>
        $<TARGET_OBJECTS:totk_sirit> $<TARGET_OBJECTS:totk_spirv_cross_core>)
    target_include_directories(totk_shader_pipeline PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(totk_shader_pipeline PRIVATE totk_yuzu_frontend totk_yuzu_compiler
        totk_sirit totk_spirv_cross_core)
    target_compile_options(totk_shader_pipeline PRIVATE -Wall -Wextra -Werror)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/compiler-identity.txt" "${COMPILER_IDENTITY}\n")
endif()

if(SPIRV_CROSS_SRC)
    add_library(totk_probe_spirv_cross STATIC spirv_cross_inproc.cpp)
    target_link_libraries(totk_probe_spirv_cross PUBLIC totk_spirv_cross_core)
    target_link_libraries(totk_yuzu_probe PRIVATE totk_probe_spirv_cross)
    target_compile_definitions(totk_yuzu_probe PRIVATE TOTK_PROBE_MSL)
    add_executable(totk_xlate service.cpp)
    target_link_libraries(totk_xlate PRIVATE totk_shader_pipeline totk_yuzu_frontend)
    target_compile_definitions(totk_xlate PRIVATE TOTK_COMPILER_IDENTITY="${COMPILER_IDENTITY}")
    target_compile_options(totk_xlate PRIVATE -Wall -Wextra -Werror)
    add_executable(totk_spirv_cross_driver tests/spirv_cross_driver.cpp)
    target_link_libraries(totk_spirv_cross_driver PRIVATE totk_probe_spirv_cross)
    add_executable(totk_instruction_fixture tests/instruction_fixture.cpp)
    target_link_libraries(totk_instruction_fixture PRIVATE totk_yuzu_compiler totk_sirit totk_probe_spirv_cross)
    target_compile_options(totk_instruction_fixture PRIVATE -Wall -Wextra -Werror)
    add_executable(totk_scalar_resource_fixtures tests/scalar_resource_fixtures.cpp)
    target_link_libraries(totk_scalar_resource_fixtures PRIVATE totk_yuzu_compiler totk_sirit)
    target_compile_options(totk_scalar_resource_fixtures PRIVATE -Wall -Wextra -Werror)
endif()

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(SPIRV_CROSS_SRC)
    set(SERVICE_GPU_PROBE)
    if(APPLE)
        set(SERVICE_GPU_PROBE "$<TARGET_FILE:totk_msl_compile_probe>")
    endif()
    add_test(NAME service COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_service.py"
        "$<TARGET_FILE:totk_xlate>" "$<TARGET_FILE:totk_spirv_cross_driver>"
        ${SERVICE_GPU_PROBE})
    add_test(NAME service_fast COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_service.py"
        "$<TARGET_FILE:totk_xlate>" "$<TARGET_FILE:totk_spirv_cross_driver>"
        ${SERVICE_GPU_PROBE})
    set_tests_properties(service_fast PROPERTIES
        ENVIRONMENT "TOTK_TEST_COMPILE_MODE=online-fast")
    set(STORAGE_IMAGE_GPU_PROBE)
    if(APPLE)
        set(STORAGE_IMAGE_GPU_PROBE "$<TARGET_FILE:totk_storage_image_gpu>")
    endif()
    add_test(NAME storage_images COMMAND "${Python3_EXECUTABLE}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_storage_images.py"
        "$<TARGET_FILE:totk_xlate>" "$<TARGET_FILE:totk_spirv_cross_driver>"
        ${STORAGE_IMAGE_GPU_PROBE})
    add_test(NAME storage_images_fast COMMAND "${Python3_EXECUTABLE}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_storage_images.py"
        "$<TARGET_FILE:totk_xlate>" "$<TARGET_FILE:totk_spirv_cross_driver>"
        ${STORAGE_IMAGE_GPU_PROBE})
    set_tests_properties(storage_images_fast PROPERTIES
        ENVIRONMENT "TOTK_TEST_COMPILE_MODE=online-fast")
    add_test(NAME scalar_resources COMMAND "${Python3_EXECUTABLE}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scalar_resources.py"
        "$<TARGET_FILE:totk_scalar_resource_fixtures>" "$<TARGET_FILE:totk_spirv_cross_driver>")
endif()
add_test(NAME compiler_probe COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_compiler_probe.py"
    "$<TARGET_FILE:totk_yuzu_probe>" "$<BOOL:${SPIRV_CROSS_SRC}>")

if(APPLE AND SPIRV_CROSS_SRC)
    enable_language(OBJC)
    set(NVN "${REPOSITORY}/games/totk/third_party/d2r_gfx/gfx/nvn")
    add_executable(totk_msl_compile_probe tests/compile_msl.m "${NVN}/nvn_refl.m")
    target_include_directories(totk_msl_compile_probe PRIVATE "${NVN}")
    target_link_libraries(totk_msl_compile_probe PRIVATE "-framework Metal" "-framework Foundation")
    target_compile_options(totk_msl_compile_probe PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    add_executable(totk_storage_image_gpu tests/storage_image_gpu.m)
    target_link_libraries(totk_storage_image_gpu PRIVATE "-framework Metal" "-framework Foundation")
    target_compile_options(totk_storage_image_gpu PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    add_test(NAME metal_compile COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_compiler_probe.py"
        "$<TARGET_FILE:totk_yuzu_probe>" 1 "$<TARGET_FILE:totk_msl_compile_probe>")
    add_test(NAME fp64_ports COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fp64_ports.py"
        "$<TARGET_FILE:totk_yuzu_probe>" 1 "$<TARGET_FILE:totk_msl_compile_probe>")
    add_test(NAME additional_instruction_ports COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_additional_instruction_ports.py"
        "$<TARGET_FILE:totk_xlate>" "$<TARGET_FILE:totk_msl_compile_probe>"
        "$<TARGET_FILE:totk_instruction_fixture>")
    add_test(NAME additional_instruction_ports_fast COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_additional_instruction_ports.py"
        "$<TARGET_FILE:totk_xlate>" "$<TARGET_FILE:totk_msl_compile_probe>"
        "$<TARGET_FILE:totk_instruction_fixture>")
    set_tests_properties(additional_instruction_ports_fast PROPERTIES
        ENVIRONMENT "TOTK_SHADER_COMPILE_MODE=online-fast")
    add_test(NAME instruction_ports COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_instruction_ports.py"
        "$<TARGET_FILE:totk_xlate>" "$<TARGET_FILE:totk_spirv_cross_driver>"
        "$<TARGET_FILE:totk_msl_compile_probe>")
endif()
