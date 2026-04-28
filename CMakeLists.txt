cmake_minimum_required(VERSION 3.16)
project(turboquant_nif C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── llama.cpp path ──────────────────────────────────────────────────────────────
if(NOT DEFINED LLAMA_CPP_DIR)
  get_filename_component(_parent "${CMAKE_SOURCE_DIR}" DIRECTORY)
  get_filename_component(_desktop "${_parent}" DIRECTORY)
  set(LLAMA_CPP_DIR "${_desktop}/turboquant-godot/thirdparty/llama_cpp")
endif()

message(STATUS "LLAMA_CPP_DIR = ${LLAMA_CPP_DIR}")

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "llama.cpp not found at ${LLAMA_CPP_DIR}.\n"
    "Set -DLLAMA_CPP_DIR=<path> or LLAMA_CPP_DIR env var to the turboquant fork.")
endif()

# ── llama.cpp build options ─────────────────────────────────────────────────────
set(LLAMA_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER      OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS       OFF)

add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_BINARY_DIR}/llama_build EXCLUDE_FROM_ALL)

# mtmd (multimodal) library — only if the llama_cpp tree didn't already add it
if(NOT TARGET mtmd)
  add_subdirectory(${LLAMA_CPP_DIR}/tools/mtmd ${CMAKE_BINARY_DIR}/mtmd_build EXCLUDE_FROM_ALL)
endif()

# ── ERTS include dir ────────────────────────────────────────────────────────────
if(NOT DEFINED ERTS_INCLUDE_DIR)
  execute_process(
    COMMAND erl -eval "io:format(code:root_dir()), halt()" -noshell
    OUTPUT_VARIABLE _erl_root
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(ERTS_INCLUDE_DIR "${_erl_root}/usr/include")
endif()
message(STATUS "ERTS_INCLUDE_DIR = ${ERTS_INCLUDE_DIR}")

# ── NIF shared library ──────────────────────────────────────────────────────────
add_library(turboquant_nif SHARED c_src/llama_nif.cpp)

target_include_directories(turboquant_nif PRIVATE
  standalone
  ${ERTS_INCLUDE_DIR}
  ${LLAMA_CPP_DIR}/include
  ${LLAMA_CPP_DIR}/vendor          # nlohmann/json.hpp lives here
  ${LLAMA_CPP_DIR}/tools/mtmd      # mtmd.h, mtmd-helper.h
)

target_link_libraries(turboquant_nif PRIVATE llama mtmd)

# Platform-specific link settings
if(APPLE)
  target_link_libraries(turboquant_nif PRIVATE
    "-framework Metal"
    "-framework Foundation"
    "-framework MetalKit"
    "-framework Accelerate"
  )
  set_target_properties(turboquant_nif PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup"
    PREFIX     "lib"
    SUFFIX     ".so"
    OUTPUT_NAME "turboquant_nif"
  )
elseif(WIN32)
  set_target_properties(turboquant_nif PROPERTIES
    PREFIX      "lib"
    SUFFIX      ".dll"
    OUTPUT_NAME "turboquant_nif"
  )
else()
  set_target_properties(turboquant_nif PROPERTIES
    PREFIX      "lib"
    SUFFIX      ".so"
    OUTPUT_NAME "turboquant_nif"
  )
endif()
