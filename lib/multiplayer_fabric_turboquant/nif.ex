defmodule MultiplayerFabric.Turboquant.NIF do
  @on_load :__on_load__

  def __on_load__ do
    path =
      :code.priv_dir(:multiplayer_fabric_turboquant)
      |> to_string()
      |> Kernel.<>("/libturboquant_nif")

    :erlang.load_nif(path, 0)
  end

  # Backend lifecycle
  def backend_init(), do: :erlang.nif_error(:not_loaded)
  def backend_free(), do: :erlang.nif_error(:not_loaded)

  # model_load(path, n_gpu_layers, use_mmap, use_mlock, mmproj_path) -> opaque model ref
  # mmproj_path: path to vision-encoder GGUF, or "" for text-only models.
  # Dirty IO-bound: loads GGUF file(s) from disk.
  def model_load(_path, _n_gpu_layers, _use_mmap, _use_mlock, _mmproj_path),
    do: :erlang.nif_error(:not_loaded)

  # context_create(model, n_ctx, n_threads, flash_attn, cache_type_k, cache_type_v)
  # -> opaque context ref
  def context_create(_model, _n_ctx, _n_threads, _flash_attn, _cache_type_k, _cache_type_v),
    do: :erlang.nif_error(:not_loaded)

  # model_desc(model) -> binary  (e.g. "llama 7B Q4_0")
  def model_desc(_model), do: :erlang.nif_error(:not_loaded)

  # model_n_ctx_train(model) -> integer  (context length the model was trained on)
  def model_n_ctx_train(_model), do: :erlang.nif_error(:not_loaded)

  # tokenize_count(model, text) -> integer
  # Returns the number of tokens `text` tokenises to. Non-dirty: fast.
  def tokenize_count(_model, _text), do: :erlang.nif_error(:not_loaded)

  # context_reset(ctx) -> :ok   [clears KV cache + cached token list]
  def context_reset(_ctx), do: :erlang.nif_error(:not_loaded)

  # context_abort(ctx) -> :ok   [sets abort flag; inference stops at next token]
  def context_abort(_ctx), do: :erlang.nif_error(:not_loaded)

  # chat_complete(ctx, messages_json, temperature, max_tokens, top_k, top_p, repeat_penalty)
  # -> binary  (full response)
  # Dirty CPU-bound.
  def chat_complete(
        _ctx,
        _messages_json,
        _temperature,
        _max_tokens,
        _top_k,
        _top_p,
        _repeat_penalty
      ),
      do: :erlang.nif_error(:not_loaded)

  # chat_stream(ctx, subscriber_pid, messages_json, temperature, max_tokens, top_k, top_p, repeat_penalty)
  # -> :ok
  # Sends {:turboquant_token, binary} for each piece, then {:turboquant_done, binary}
  # or {:turboquant_error, binary} to subscriber_pid.
  # Dirty CPU-bound.
  def chat_stream(
        _ctx,
        _subscriber,
        _messages_json,
        _temperature,
        _max_tokens,
        _top_k,
        _top_p,
        _repeat_penalty
      ),
      do: :erlang.nif_error(:not_loaded)
end
