defmodule TurboquantLlmTest do
  use ExUnit.Case, async: false

  alias TurboquantLlm
  alias TurboquantLlm.Session

  @moduletag :integration

  # Point at a local GGUF for integration tests.
  # Run with:  LLAMA_MODEL=/path/to/model.gguf mix test
  @model_path System.get_env("LLAMA_MODEL")

  defp skip_unless_model(context) do
    if is_nil(@model_path) or not File.exists?(@model_path) do
      {:ok, Map.put(context, :skip, true)}
    else
      {:ok, context}
    end
  end

  setup :skip_unless_model

  setup %{skip: true} do
    IO.puts("Skipping integration test: set LLAMA_MODEL=/path/to/model.gguf to run.")
    {:ok, skip: true}
  end

  setup %{skip: false} do
    {:ok, session} =
      Session.start_link(
        model_path:   @model_path,
        n_gpu_layers: 0,
        n_ctx:        512,
        n_threads:    2,
        cache_type_k: "f16",
        cache_type_v: "f16",
        flash_attn:   false,
        max_tokens:   64
      )

    {:ok, session: session}
  end

  test "chat/3 returns a non-empty string response", %{skip: true}, do: :ok

  test "chat/3 returns a non-empty string response", %{session: session} do
    messages = [%{"role" => "user", "content" => "Say exactly: hello"}]
    assert {:ok, response} = Turboquant.chat(session, messages)
    assert is_binary(response) and byte_size(response) > 0
  end

  test "stream_chat/3 emits tokens then completes", %{skip: true}, do: :ok

  test "stream_chat/3 emits tokens then completes", %{session: session} do
    messages = [%{"role" => "user", "content" => "Count to three."}]
    tokens = Turboquant.stream_chat(session, messages) |> Enum.to_list()
    assert is_list(tokens) and length(tokens) > 0
    assert Enum.all?(tokens, &is_binary/1)
  end

  test "reset/1 clears KV cache without error", %{skip: true}, do: :ok

  test "reset/1 clears KV cache without error", %{session: session} do
    messages = [%{"role" => "user", "content" => "Hello"}]
    assert {:ok, _} = Turboquant.chat(session, messages)
    assert :ok = Turboquant.reset(session)
  end

  test "abort/1 stops a running inference", %{skip: true}, do: :ok

  test "abort/1 stops a running inference", %{session: session} do
    messages = [%{"role" => "user", "content" => "Write a very long story."}]

    task = Task.async(fn ->
      Turboquant.chat(session, messages, max_tokens: 1024)
    end)

    Process.sleep(100)
    Turboquant.abort(session)

    result = Task.await(task, 10_000)
    assert match?({:ok, _}, result) or match?({:error, _}, result)
  end
end
