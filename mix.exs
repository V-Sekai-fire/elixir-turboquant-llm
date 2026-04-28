defmodule TurboquantLlm.MixProject do
  use Mix.Project

  def project do
    [
      app: :elixir_turboquant_llm,
      version: "0.1.0",
      elixir: "~> 1.17",
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_executable: "cmake",
      make_targets: ["-P", "build.cmake"],
      make_clean: ["-P", "clean.cmake"],
      make_env: make_env(),
      make_error_message: """

      Failed to compile the turboquant NIF.

      Make sure the turboquant-godot llama.cpp fork is present and point to it:

        LLAMA_CPP_DIR=/path/to/turboquant-godot/thirdparty/llama_cpp mix compile

      By default the Makefile looks for it at:
        ../../turboquant-godot/thirdparty/llama_cpp
      (i.e. turboquant-godot must be a sibling of this repository on disk)

      macOS: Xcode command-line tools and cmake are required.
      Linux: cmake, a C++17 compiler, and optionally the Vulkan SDK.
      Windows: MSYS2 with cmake and Ninja installed.
      """,
      test_coverage: [tool: ExCoveralls],
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps()
    ]
  end

  def cli do
    [preferred_envs: [propcheck: :test, coveralls: :test, "coveralls.detail": :test, "coveralls.html": :test]]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {TurboquantLlm.Application, []}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_),     do: ["lib"]

  defp make_env do
    case System.get_env("LLAMA_CPP_DIR") do
      nil -> %{}
      dir -> %{"LLAMA_CPP_DIR" => dir}
    end
  end

  defp deps do
    [
      {:elixir_make, "~> 0.9"},
      {:fine, "0.1.6"},
      {:jason, "~> 1.4"},
      {:propcheck, "~> 1.4", only: [:test, :dev], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev], runtime: false},
      {:excoveralls, "~> 0.18", only: [:test], runtime: false},
      {:mox, "~> 1.2", only: [:test]}
    ]
  end
end
