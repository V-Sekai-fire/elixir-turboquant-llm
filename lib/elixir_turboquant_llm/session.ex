defmodule TurboquantLlm.Session do
  @moduledoc """
  GenServer that owns a single LLM model + context pair and serialises
  inference requests for one NPC / character session.

  ## Options (start_link)

    * `:model_path`     - (required) path to a GGUF model file
    * `:n_gpu_layers`   - layers offloaded to GPU; -1 = all (default: -1)
    * `:use_mmap`       - use memory-mapped file I/O (default: true)
    * `:use_mlock`      - lock model in RAM (default: false)
    * `:n_ctx`          - context window size in tokens (default: 4096)
    * `:n_threads`      - CPU threads for inference (default: logical CPUs - 1)
    * `:flash_attn`     - enable Flash Attention (default: true)
    * `:cache_type_k`   - KV-cache K type: "turbo4" | "turbo3" | "turbo2" | "q8_0" | "q4_0" | "f16" (default: "turbo4")
    * `:cache_type_v`   - KV-cache V type (same options, default: "turbo4")
    * `:temperature`    - default sampling temperature (default: 0.7)
    * `:max_tokens`     - default max output tokens; 0 = no limit (default: 0)
    * `:top_k`          - default top-k sampling (default: 40)
    * `:top_p`          - default top-p sampling (default: 0.95)
    * `:repeat_penalty` - default repetition penalty (default: 1.1)
    * `:mmproj_path`    - path to the vision-encoder (mmproj) GGUF for VL models; omit or
                          set to `""` for text-only models. When set, messages may include
                          an `"images"` key with a list of file paths:
                          `[%{"role" => "user", "content" => "...", "images" => ["/img.jpg"]}]`
  """

  use GenServer
  require Logger

  @default_opts [
    n_gpu_layers:   -1,
    use_mmap:       true,
    use_mlock:      false,
    n_ctx:          4096,
    flash_attn:     true,
    cache_type_k:   "turbo4",
    cache_type_v:   "turbo4",
    temperature:    0.7,
    max_tokens:     0,
    top_k:          40,
    top_p:          0.95,
    repeat_penalty: 1.1,
    mmproj_path:    ""
  ]

  # ── Client API ──────────────────────────────────────────────────────────────

  @doc """
  Returns a `{:via, Registry, ...}` tuple for naming sessions through a Registry.

      # In your supervision tree
      children = [
        {Registry, keys: :unique, name: NpcRegistry},
        {Session, model_path: "...", name: Session.via(NpcRegistry, :guard_captain)}
      ]

      # Later
      Session.chat(Session.via(NpcRegistry, :guard_captain), messages)
  """
  def via(registry, key), do: {:via, Registry, {registry, key}}

  # GenServer options (name:, timeout:, debug:, spawn_opt:, hibernate_after:)
  # must be passed to start_link, not init/1.
  @genserver_keys [:name, :timeout, :debug, :spawn_opt, :hibernate_after]

  def start_link(opts) do
    {server_opts, init_opts} = Keyword.split(opts, @genserver_keys)
    GenServer.start_link(__MODULE__, init_opts, server_opts)
  end

  @doc "Blocking chat completion. Returns `{:ok, response}` or `{:error, reason}`."
  def chat(session, messages, opts \\ []) do
    GenServer.call(session, {:chat, messages, opts}, :infinity)
  end

  @doc """
  Streaming chat. Returns a `Stream` that emits binary tokens.

  The stream collects `{:turboquant_token, bin}` messages from the NIF's
  dirty-scheduler thread, finishing when `{:turboquant_done, _}` arrives.
  Raises on `{:turboquant_error, reason}`.
  """
  def stream_chat(session, messages, opts \\ []) do
    caller = self()
    :ok = GenServer.call(session, {:start_stream, messages, opts, caller}, :infinity)

    Stream.resource(
      fn -> :streaming end,
      fn :streaming ->
        receive do
          {:turboquant_token, token} ->
            {[token], :streaming}

          {:turboquant_done, _full} ->
            {:halt, :done}

          {:turboquant_error, reason} ->
            raise "TurboquantLLM inference error: #{reason}"
        after
          120_000 ->
            raise "TurboquantLLM stream timeout (120 s)"
        end
      end,
      fn _ -> :ok end
    )
  end

  @doc "Returns the model description string, e.g. `\"llama 7B Q4_0\"`."
  def model_desc(session), do: GenServer.call(session, :model_desc)

  @doc "Returns how many tokens `text` tokenises to (fast, non-blocking)."
  def tokenize_count(session, text), do: GenServer.call(session, {:tokenize_count, text})

  @doc "Returns the context length the model was trained on."
  def model_n_ctx_train(session), do: GenServer.call(session, :model_n_ctx_train)

  @doc "Signal the running inference to stop at the next token boundary."
  def abort(session), do: GenServer.cast(session, :abort)

  @doc "Clear KV cache and token history (resets conversation context)."
  def reset(session), do: GenServer.call(session, :reset)

  @doc "Returns `true` while the model and context are still loading."
  def loading?(session), do: GenServer.call(session, :loading?)

  @doc "Returns `:loading`, `:ready`, or `:error`."
  def status(session), do: GenServer.call(session, :status)

  @doc """
  Blocks until the session is ready or `timeout_ms` elapses.
  Returns `:ok`, `{:error, :timeout}`, or `{:error, :load_failed}`.
  """
  def wait_until_ready(session, timeout_ms \\ 120_000) do
    deadline = System.monotonic_time(:millisecond) + timeout_ms
    do_wait(session, deadline)
  end

  defp do_wait(session, deadline) do
    try do
      case status(session) do
        :ready   -> :ok
        :error   -> {:error, :load_failed}
        :loading ->
          remaining = deadline - System.monotonic_time(:millisecond)
          if remaining <= 0 do
            {:error, :timeout}
          else
            Process.sleep(min(100, remaining))
            do_wait(session, deadline)
          end
      end
    catch
      :exit, _ -> {:error, :load_failed}
    end
  end

  @doc """
  Returns `true` if `text` tokenises to fewer tokens than the session's
  context window. Fast, non-blocking.
  """
  def fits_in_context?(session, text), do: GenServer.call(session, {:fits_in_context, text})

  # ── GenServer callbacks ─────────────────────────────────────────────────────

  @impl true
  def init(opts) do
    opts = Keyword.merge(@default_opts, opts)
    # Return immediately so start_link/supervisor don't time out on large models.
    # Actual NIF calls happen in handle_continue/2.
    {:ok, %{opts: opts, status: :loading}, {:continue, :load}}
  end

  @impl true
  def handle_continue(:load, %{opts: opts} = _state) do
    n_threads = Keyword.get(opts, :n_threads, max(1, System.schedulers_online() - 1))

    try do
      model = nif().model_load(
        opts[:model_path],
        opts[:n_gpu_layers],
        opts[:use_mmap],
        opts[:use_mlock],
        opts[:mmproj_path]
      )

      n_ctx_train = nif().model_n_ctx_train(model)
      if opts[:n_ctx] > n_ctx_train do
        Logger.warning(
          "TurboquantSession: n_ctx (#{opts[:n_ctx]}) exceeds model training context " <>
          "(#{n_ctx_train}). Inference quality may degrade beyond #{n_ctx_train} tokens."
        )
      end

      ctx = nif().context_create(
        model,
        opts[:n_ctx],
        n_threads,
        opts[:flash_attn],
        opts[:cache_type_k],
        opts[:cache_type_v]
      )

      {:noreply, %{
        model:  model,
        ctx:    ctx,
        status: :ready,
        n_ctx:  opts[:n_ctx],
        infer:  %{
          temperature:    opts[:temperature],
          max_tokens:     opts[:max_tokens],
          top_k:          opts[:top_k],
          top_p:          opts[:top_p],
          repeat_penalty: opts[:repeat_penalty]
        }
      }}
    rescue
      e ->
        Logger.error("TurboquantSession load failed: #{Exception.message(e)}")
        {:noreply, %{status: :error}}
    end
  end

  @impl true
  def handle_call(:status, _from, state) do
    {:reply, state.status, state}
  end

  def handle_call({:fits_in_context, text}, _from, %{status: :ready} = state) do
    count = nif().tokenize_count(state.model, text)
    {:reply, count < state.n_ctx, state}
  end

  def handle_call(:model_n_ctx_train, _from, %{status: :ready} = state) do
    {:reply, nif().model_n_ctx_train(state.model), state}
  end

  def handle_call(:model_desc, _from, %{status: :ready} = state) do
    {:reply, nif().model_desc(state.model), state}
  end

  def handle_call({:tokenize_count, text}, _from, %{status: :ready} = state) do
    {:reply, nif().tokenize_count(state.model, text), state}
  end

  def handle_call(:loading?, _from, state) do
    {:reply, state.status == :loading, state}
  end

  def handle_call(_msg, _from, %{status: :loading} = state) do
    {:reply, {:error, :loading}, state}
  end

  def handle_call(_msg, _from, %{status: :error} = state) do
    {:reply, {:error, :load_failed}, state}
  end

  def handle_call({:chat, messages, call_opts}, from, state) do
    p        = merge_infer(state.infer, call_opts)
    ctx      = state.ctx
    msgs_json = Jason.encode!(messages)

    # Spawn task so GenServer mailbox stays responsive during dirty NIF
    Task.start(fn ->
      result =
        try do
          response = nif().chat_complete(
            ctx, msgs_json,
            p.temperature, p.max_tokens, p.top_k, p.top_p, p.repeat_penalty
          )
          {:ok, response}
        rescue
          e -> {:error, Exception.message(e)}
        end

      GenServer.reply(from, result)
    end)

    {:noreply, state}
  end

  def handle_call({:start_stream, messages, call_opts, subscriber}, _from, state) do
    p        = merge_infer(state.infer, call_opts)
    ctx      = state.ctx
    msgs_json = Jason.encode!(messages)

    # Fire-and-forget: dirty NIF sends tokens directly to subscriber.
    # Elixir rescue ensures the subscriber always receives a terminal message
    # even if the BEAM-level NIF call itself raises before C++ error handling runs.
    Task.start(fn ->
      try do
        nif().chat_stream(
          ctx, subscriber, msgs_json,
          p.temperature, p.max_tokens, p.top_k, p.top_p, p.repeat_penalty
        )
      rescue
        e -> send(subscriber, {:turboquant_error, Exception.message(e)})
      end
    end)

    {:reply, :ok, state}
  end

  def handle_call(:reset, _from, state) do
    nif().context_reset(state.ctx)
    {:reply, :ok, state}
  end

  @impl true
  def handle_cast(:abort, %{status: :ready} = state) do
    nif().context_abort(state.ctx)
    {:noreply, state}
  end

  def handle_cast(:abort, state) do
    # Ignore abort during loading or error — no context exists yet.
    {:noreply, state}
  end

  # ── Helpers ─────────────────────────────────────────────────────────────────

  defp merge_infer(defaults, overrides) do
    overrides
    |> Enum.into(%{})
    |> then(&Map.merge(defaults, &1))
  end

  defp nif do
    Application.get_env(:elixir_turboquant_llm, :nif_module, TurboquantLlm.NIF)
  end
end
