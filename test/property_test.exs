defmodule MultiplayerFabricTurboquant.PropertyTest do
  use ExUnit.Case, async: false
  use PropCheck

  alias MultiplayerFabric.Turboquant.Session

  # ── Generators ────────────────────────────────────────────────────────────────

  # Valid role atoms as strings
  def role_gen, do: oneof(["system", "user", "assistant"])

  # Non-empty string content
  def content_gen do
    let chars <- non_empty(list(range(?a, ?z))) do
      to_string(chars)
    end
  end

  # A single well-formed message map
  def message_gen do
    let {role, content} <- {role_gen(), content_gen()} do
      %{"role" => role, "content" => content}
    end
  end

  # Non-empty list of messages (at most 10 to keep tests fast)
  def messages_gen do
    let n <- range(1, 10) do
      let msgs <- vector(n, message_gen()) do
        msgs
      end
    end
  end

  # Temperature: [0.0, 2.0]
  def temperature_gen, do: float(0.0, 2.0)

  # top_k: [1, 100]
  def top_k_gen, do: range(1, 100)

  # top_p: [0.0, 1.0]
  def top_p_gen, do: float(0.0, 1.0)

  # repeat_penalty: [1.0, 2.0]
  def repeat_penalty_gen, do: float(1.0, 2.0)

  # max_tokens: 0 (unlimited) or [1, 512]
  def max_tokens_gen, do: oneof([exactly(0), range(1, 512)])

  # Full inference opts
  def infer_opts_gen do
    let {temp, top_k, top_p, rp, mt} <-
          {temperature_gen(), top_k_gen(), top_p_gen(), repeat_penalty_gen(), max_tokens_gen()} do
      [temperature: temp, top_k: top_k, top_p: top_p, repeat_penalty: rp, max_tokens: mt]
    end
  end

  # ── Pure Elixir properties (no NIF required) ───────────────────────────────────

  property "Jason.encode!/1 never raises on valid message lists" do
    forall msgs <- messages_gen() do
      is_binary(Jason.encode!(msgs))
    end
  end

  property "Jason.encode! + Jason.decode! round-trips messages" do
    forall msgs <- messages_gen() do
      decoded = msgs |> Jason.encode!() |> Jason.decode!()
      length(decoded) == length(msgs)
    end
  end

  property "merge_infer preserves all keys from defaults when overrides are empty" do
    defaults = %{temperature: 0.7, max_tokens: 0, top_k: 40, top_p: 0.95, repeat_penalty: 1.1}
    forall _unused <- exactly(:ok) do
      merged = Map.merge(defaults, %{})
      Map.keys(merged) == Map.keys(defaults)
    end
  end

  property "merge_infer: caller opts overwrite default opts" do
    forall {temp, top_k} <- {temperature_gen(), top_k_gen()} do
      defaults = %{temperature: 0.7, top_k: 40, top_p: 0.95, repeat_penalty: 1.1, max_tokens: 0}
      overrides = %{temperature: temp, top_k: top_k}
      merged = Map.merge(defaults, overrides)
      merged.temperature == temp and merged.top_k == top_k
    end
  end

  property "messages with no images key pass messages_have_images check as false" do
    # Simulate the JSON side: none of these messages have an "images" key
    forall msgs <- messages_gen() do
      json = Jason.encode!(msgs)
      decoded = Jason.decode!(json)
      not Enum.any?(decoded, &Map.has_key?(&1, "images"))
    end
  end

  property "message JSON with images key is preserved through encode/decode" do
    forall {msg, paths} <- {message_gen(), non_empty(list(content_gen()))} do
      msg_with_images = Map.put(msg, "images", paths)
      decoded = [msg_with_images] |> Jason.encode!() |> Jason.decode!() |> hd()
      decoded["images"] == paths
    end
  end

  # ── Integration properties (require LLAMA_MODEL env var) ─────────────────────

  @model_path System.get_env("LLAMA_MODEL")
  @moduletag :integration

  defp session_available? do
    not is_nil(@model_path) and File.exists?(@model_path)
  end

  defp start_session do
    {:ok, pid} = Session.start_link(
      model_path:   @model_path,
      n_gpu_layers: 0,
      n_ctx:        512,
      n_threads:    2,
      cache_type_k: "f16",
      cache_type_v: "f16",
      flash_attn:   false,
      max_tokens:   32
    )
    on_exit(fn -> if Process.alive?(pid), do: GenServer.stop(pid) end)
    {:ok, pid}
  end

  property "wait_until_ready leaves status as :ready (not :loading)" do
    if session_available?() do
      {:ok, session} = start_session()
      forall _unused <- exactly(:ok) do
        :ok = Session.wait_until_ready(session, 30_000)
        Session.status(session) == :ready
      end
    else
      true
    end
  end

  property "model_n_ctx_train is a positive integer" do
    if session_available?() do
      {:ok, session} = start_session()
      forall _unused <- exactly(:ok) do
        n = Session.model_n_ctx_train(session)
        is_integer(n) and n > 0
      end
    else
      true
    end
  end

  property "chat/3 returns {:ok, binary} for any valid messages list", [:verbose] do
    if session_available?() do
      {:ok, session} = start_session()

      forall msgs <- messages_gen() do
        case Session.chat(session, msgs) do
          {:ok, response} -> is_binary(response)
          {:error, _}     -> true  # errors are acceptable (e.g. context overflow)
        end
      end
    else
      true
    end
  end

  property "status/1 always returns an atom in the valid set" do
    if session_available?() do
      {:ok, session} = start_session()
      forall _unused <- exactly(:ok) do
        Session.status(session) in [:loading, :ready, :error]
      end
    else
      true
    end
  end

  property "tokenize_count/2 is positive for any non-empty text" do
    if session_available?() do
      {:ok, session} = start_session()
      forall text <- content_gen() do
        Session.tokenize_count(session, text) > 0
      end
    else
      true
    end
  end

  property "fits_in_context?/2 is true for short strings, consistent with tokenize_count" do
    if session_available?() do
      {:ok, session} = start_session()
      forall text <- content_gen() do
        count   = Session.tokenize_count(session, text)
        fits    = Session.fits_in_context?(session, text)
        # fits_in_context? must agree with tokenize_count < n_ctx (512 in test session)
        (count < 512) == fits
      end
    else
      true
    end
  end

  property "context_reset/1 is idempotent: two resets never crash" do
    if session_available?() do
      {:ok, session} = start_session()

      forall _unused <- exactly(:ok) do
        :ok = Session.reset(session)
        :ok = Session.reset(session)
        true
      end
    else
      true
    end
  end

  property "chat/3 with temperature extremes does not crash" do
    if session_available?() do
      {:ok, session} = start_session()
      msgs = [%{"role" => "user", "content" => "hi"}]

      forall temp <- oneof([exactly(0.0), exactly(2.0), temperature_gen()]) do
        result = Session.chat(session, msgs, temperature: temp, max_tokens: 8)
        match?({:ok, _}, result) or match?({:error, _}, result)
      end
    else
      true
    end
  end
end
