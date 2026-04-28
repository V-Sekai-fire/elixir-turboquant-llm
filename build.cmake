# build.cmake — cross-platform NIF builder (no MSYS2 required on Windows)
# Invoked by elixir_make as: cmake -P build.cmake
# Works on macOS, Linux, and Windows via cmake's built-in scripting mode.

set(PROJECT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(BUILD_DIR   "${PROJECT_DIR}/_build/nif_cmake")

# ── Paths from elixir_make env ────────────────────────────────────────────────
if(DEFINED ENV{MIX_APP_PATH})
  set(PRIV_DIR "$ENV{MIX_APP_PATH}/priv")
else()
  set(PRIV_DIR "${PROJECT_DIR}/priv")
endif()

if(DEFINED ENV{LLAMA_CPP_DIR})
  set(LLAMA_CPP_DIR "$ENV{LLAMA_CPP_DIR}")
else()
  set(LLAMA_CPP_DIR "${PROJECT_DIR}/thirdparty/llama_cpp")
endif()

# ── Platform detection ────────────────────────────────────────────────────────
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  set(NIF_NAME    "libturboquant_nif.so")
  set(GPU_FLAGS   -DGGML_METAL=ON -DGGML_VULKAN=OFF)
  set(GEN_FLAG    "")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
  set(NIF_NAME    "libturboquant_nif.dll")
  set(GPU_FLAGS   -DGGML_METAL=OFF -DGGML_VULKAN=OFF)
  set(GEN_FLAG    -GNinja)
else()
  set(NIF_NAME    "libturboquant_nif.so")
  set(GPU_FLAGS   -DGGML_METAL=OFF)
  set(GEN_FLAG    "")
endif()

# ── Parallel jobs ─────────────────────────────────────────────────────────────
cmake_host_system_information(RESULT NCPU QUERY NUMBER_OF_PHYSICAL_CORES)
if(NCPU LESS 1)
  set(NCPU 4)
endif()

# ── ERTS include dir ──────────────────────────────────────────────────────────
if(DEFINED ENV{ERTS_INCLUDE_DIR})
  set(ERTS_FLAG -DERTS_INCLUDE_DIR=$ENV{ERTS_INCLUDE_DIR})
else()
  set(ERTS_FLAG "")
endif()

# ── Create directories ────────────────────────────────────────────────────────
file(MAKE_DIRECTORY "${PRIV_DIR}")
file(MAKE_DIRECTORY "${BUILD_DIR}")

# ── Configure ─────────────────────────────────────────────────────────────────
execute_process(
  COMMAND ${CMAKE_COMMAND}
    ${GEN_FLAG}
    -S "${PROJECT_DIR}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DLLAMA_CPP_DIR=${LLAMA_CPP_DIR}
    ${ERTS_FLAG}
    ${GPU_FLAGS}
  RESULT_VARIABLE result
)
if(result)
  message(FATAL_ERROR "cmake configure failed (exit ${result})")
endif()

# ── Build ─────────────────────────────────────────────────────────────────────
execute_process(
  COMMAND ${CMAKE_COMMAND} --build "${BUILD_DIR}" --target turboquant_nif -j ${NCPU}
  RESULT_VARIABLE result
)
if(result)
  message(FATAL_ERROR "cmake build failed (exit ${result})")
endif()

# ── Install NIF to priv/ ──────────────────────────────────────────────────────
# MSVC multi-config generators put output in Release/ subdirectory.
if(EXISTS "${BUILD_DIR}/Release/${NIF_NAME}")
  file(COPY "${BUILD_DIR}/Release/${NIF_NAME}" DESTINATION "${PRIV_DIR}")
elseif(EXISTS "${BUILD_DIR}/${NIF_NAME}")
  file(COPY "${BUILD_DIR}/${NIF_NAME}" DESTINATION "${PRIV_DIR}")
else()
  message(FATAL_ERROR "${NIF_NAME} not found in ${BUILD_DIR} after build")
endif()

message(STATUS "Installed ${NIF_NAME} → ${PRIV_DIR}")
