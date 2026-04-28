defmodule TurboquantLlm do
  @moduledoc """
  llama.cpp bound to Elixir via FINE NIFs.

  Supports TurboQuant KV-cache quantisation (turbo4/turbo3/turbo2) from the
  turboquant-godot fork, with Metal on macOS, CPU on Linux/Windows.

  ## Quick start

      {:ok, session} =
        TurboquantLlm.Session.start_link(
          model_path:   "/models/my-model.gguf",
          n_gpu_layers: -1,
          n_ctx:        4096,
          cache_type_k: "turbo4",
          cache_type_v: "turbo4"
        )

      {:ok, reply} =
        TurboquantLlm.chat(session, [
          %{role: "system",  content: "You are a helpful NPC."},
          %{role: "user",    content: "What quests are available?"}
        ])

      # Streaming
      TurboquantLlm.stream_chat(session, messages)
      |> Stream.each(&IO.write/1)
      |> Stream.run()
  """

  alias TurboquantLlm.Session

  @doc "Start a new inference session. See `Session.start_link/1` for options."
  defdelegate start_session(opts), to: Session, as: :start_link

  @doc "Registry via-tuple for naming a session. See `Session.via/2`."
  defdelegate via(registry, key), to: Session

  @doc "Blocking chat completion."
  defdelegate chat(session, messages, opts \\ []), to: Session

  @doc "Streaming chat — returns a lazy `Stream` of token binaries."
  defdelegate stream_chat(session, messages, opts \\ []), to: Session

  @doc "Abort the running inference."
  defdelegate abort(session), to: Session

  @doc "Reset context (clears KV cache and conversation history)."
  defdelegate reset(session), to: Session

  @doc "Returns `true` while the model/context are still loading after `start_session/1`."
  defdelegate loading?(session), to: Session

  @doc "Returns `:loading`, `:ready`, or `:error`."
  defdelegate status(session), to: Session

  @doc "Blocks until ready or timeout. Returns `:ok`, `{:error, :timeout}`, or `{:error, :load_failed}`."
  defdelegate wait_until_ready(session, timeout_ms \\ 120_000), to: Session

  @doc "Returns `true` if `text` fits within the session's context window."
  defdelegate fits_in_context?(session, text), to: Session

  @doc "Model description string, e.g. `\"llama 7B Q4_0\"`."
  defdelegate model_desc(session), to: Session

  @doc "Number of tokens `text` tokenises to. Fast, non-blocking."
  defdelegate tokenize_count(session, text), to: Session

  @doc "Context length the model was trained on. Useful for detecting n_ctx overshoot."
  defdelegate model_n_ctx_train(session), to: Session
end
