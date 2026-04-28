# clean.cmake — cross-platform NIF cleaner
# Invoked by elixir_make as: cmake -P clean.cmake

set(PROJECT_DIR "${CMAKE_CURRENT_LIST_DIR}")

if(DEFINED ENV{MIX_APP_PATH})
  set(PRIV_DIR "$ENV{MIX_APP_PATH}/priv")
else()
  set(PRIV_DIR "${PROJECT_DIR}/priv")
endif()

file(REMOVE_RECURSE "${PROJECT_DIR}/_build/nif_cmake")
file(REMOVE "${PRIV_DIR}/libturboquant_nif.so")
file(REMOVE "${PRIV_DIR}/libturboquant_nif.dll")
message(STATUS "Cleaned NIF build artifacts")
