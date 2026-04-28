PRIV_DIR   = $(MIX_APP_PATH)/priv
BUILD_DIR  = _build/nif_cmake

# Default: look for turboquant-godot as a sibling of the rpg repo
LLAMA_CPP_DIR ?= $(abspath $(dir $(firstword $(MAKEFILE_LIST)))/../../turboquant-godot/thirdparty/llama_cpp)

UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
  NIF_NAME       = libturboquant_nif.so
  CMAKE_GPU      = -DGGML_METAL=ON -DGGML_VULKAN=OFF
  CMAKE_GENERATOR =
else ifeq ($(UNAME), Linux)
  NIF_NAME       = libturboquant_nif.so
  CMAKE_GPU      = -DGGML_METAL=OFF
  CMAKE_GENERATOR =
else
  # Windows (MSYS2 / MinGW) — Ninja avoids MSVC multi-config subdirs.
  # lib prefix matches what :erlang.load_nif/2 looks for on Windows.
  NIF_NAME       = libturboquant_nif.dll
  CMAKE_GPU      = -DGGML_METAL=OFF -DGGML_VULKAN=OFF
  CMAKE_GENERATOR = -G Ninja
endif

NIF_SO  = $(PRIV_DIR)/$(NIF_NAME)
NCPU   := $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

all: $(NIF_SO)

$(NIF_SO): c_src/llama_nif.cpp CMakeLists.txt | $(PRIV_DIR)
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) \
	  $(CMAKE_GENERATOR) \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DERTS_INCLUDE_DIR=$(ERTS_INCLUDE_DIR) \
	  -DLLAMA_CPP_DIR=$(LLAMA_CPP_DIR) \
	  $(CMAKE_GPU)
	cmake --build $(BUILD_DIR) --target turboquant_nif -j$(NCPU)
	@if [ -f "$(BUILD_DIR)/Release/$(NIF_NAME)" ]; then \
	  cp "$(BUILD_DIR)/Release/$(NIF_NAME)" $(NIF_SO); \
	elif [ -f "$(BUILD_DIR)/$(NIF_NAME)" ]; then \
	  cp "$(BUILD_DIR)/$(NIF_NAME)" $(NIF_SO); \
	else \
	  echo "ERROR: $(NIF_NAME) not found after cmake build" && exit 1; \
	fi

$(PRIV_DIR):
	mkdir -p $(PRIV_DIR)

clean:
	rm -rf $(BUILD_DIR) $(NIF_SO)

.PHONY: all clean
