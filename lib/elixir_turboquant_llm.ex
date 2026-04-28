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

  alias TurboquantLlm.{Downloader, Session}

  # Default model: Qwen3.5-27B reasoning-distilled, Q4_K_M quant (~15 GB).
  # Same file the turboquant-project Godot app downloads by default.
  @default_model_url "https://huggingface.co/mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF/resolve/main/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.i1-Q4_K_M.gguf"

  @doc "The default model URL (Qwen3.5-27B Q4_K_M reasoning distil)."
  def default_model_url, do: @default_model_url

  @doc """
  Local cache directory for downloaded models.
  Overridable via `TURBOQUANT_CACHE_DIR` env var.
  """
  def cache_dir do
    System.get_env("TURBOQUANT_CACHE_DIR") ||
      Path.join([System.user_home!(), ".cache", "turboquant-llm"])
  end

  @doc """
  Downloads `url` (default: `default_model_url/0`) to `cache_dir/0`.
  Skips the download if the file already exists.

  Returns `{:ok, local_path}` or `{:error, reason}`.

  ## Options
    * `:on_progress` — `fun(bytes_received, total_bytes)` for progress reporting.
    * `:url` — override the download URL.

  ## Example

      {:ok, path} = TurboquantLlm.download_model(
        on_progress: fn recv, total ->
          IO.write("\\r\#{Float.round(recv / total * 100, 1)}%")
        end
      )
      {:ok, session} = TurboquantLlm.start_session(model_path: path, n_gpu_layers: -1)
  """
  def download_model(opts \\ []) do
    url  = Keyword.get(opts, :url, @default_model_url)
    name = url |> URI.parse() |> Map.fetch!(:path) |> Path.basename()
    dest = Path.join(cache_dir(), name)
    Downloader.download(url, dest, opts)
  end

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
