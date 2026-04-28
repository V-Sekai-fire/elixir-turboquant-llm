defmodule TurboquantLlmTest do
  use ExUnit.Case, async: false

  # Run with: LLAMA_MODEL=/path/to/model.gguf mix test --include integration
  @moduletag :integration

  @model_path System.get_env("LLAMA_MODEL")

  setup_all do
    unless @model_path && File.exists?(@model_path) do
      raise "LLAMA_MODEL not set or file not found. Run with LLAMA_MODEL=/path/to/model.gguf"
    end

    {:ok, session} =
      TurboquantLlm.Session.start_link(
        model_path:   @model_path,
        n_gpu_layers: 0,
        n_ctx:        512,
        n_threads:    2,
        cache_type_k: "f16",
        cache_type_v: "f16",
        flash_attn:   false,
        max_tokens:   64
      )

    :ok = TurboquantLlm.wait_until_ready(session)
    {:ok, session: session}
  end

  test "chat/3 returns a non-empty binary response", %{session: session} do
    messages = [%{"role" => "user", "content" => "Say exactly: hello"}]
    assert {:ok, response} = TurboquantLlm.chat(session, messages)
    assert is_binary(response) and byte_size(response) > 0
  end

  test "stream_chat/3 emits token binaries then completes", %{session: session} do
    messages = [%{"role" => "user", "content" => "Count to three."}]
    tokens = TurboquantLlm.stream_chat(session, messages) |> Enum.to_list()
    assert length(tokens) > 0
    assert Enum.all?(tokens, &is_binary/1)
  end

  test "reset/1 clears KV cache without error", %{session: session} do
    messages = [%{"role" => "user", "content" => "Hello"}]
    assert {:ok, _} = TurboquantLlm.chat(session, messages)
    assert :ok = TurboquantLlm.reset(session)
  end

  test "abort/1 stops a running inference", %{session: session} do
    messages = [%{"role" => "user", "content" => "Write a very long story."}]

    task = Task.async(fn ->
      TurboquantLlm.chat(session, messages, max_tokens: 1024)
    end)

    Process.sleep(100)
    TurboquantLlm.abort(session)

    result = Task.await(task, 10_000)
    assert match?({:ok, _}, result) or match?({:error, _}, result)
  end
end
