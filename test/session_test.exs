defmodule TurboquantLlm.SessionTest do
  use ExUnit.Case, async: false
  import Mox

  alias TurboquantLlm.{NIF.Mock, Session}

  setup :set_mox_global
  setup :verify_on_exit!

  # ── Shared stub helpers ───────────────────────────────────────────────────

  defp model_ref, do: make_ref()
  defp ctx_ref,   do: make_ref()

  defp stub_load(model \\ model_ref(), ctx \\ ctx_ref()) do
    stub(Mock, :model_load,        fn _, _, _, _, _ -> model end)
    stub(Mock, :model_n_ctx_train, fn _ -> 4096 end)
    stub(Mock, :context_create,    fn _, _, _, _, _, _ -> ctx end)
    {model, ctx}
  end

  defp start_ready(opts \\ []) do
    {model, ctx} = stub_load()
    {:ok, session} = Session.start_link(Keyword.merge([model_path: "/fake.gguf"], opts))
    :ok = Session.wait_until_ready(session)
    {session, model, ctx}
  end

  # ── Loading lifecycle ─────────────────────────────────────────────────────

  test "status is :loading immediately after start, :ready after handle_continue" do
    {model, ctx} = stub_load()
    {:ok, session} = Session.start_link(model_path: "/fake.gguf")
    # allow a tiny window to observe :loading before handle_continue fires
    assert Session.status(session) in [:loading, :ready]
    :ok = Session.wait_until_ready(session)
    assert Session.status(session) == :ready
    _ = {model, ctx}
  end

  test "loading? returns false once ready" do
    stub_load()
    {:ok, session} = Session.start_link(model_path: "/fake.gguf")
    :ok = Session.wait_until_ready(session)
    refute Session.loading?(session)
  end

  test "status is :error when model_load raises" do
    stub(Mock, :model_load, fn _, _, _, _, _ -> raise "bad path" end)
    {:ok, session} = Session.start_link(model_path: "/bad.gguf")
    # wait_until_ready should return :load_failed
    assert {:error, :load_failed} = Session.wait_until_ready(session, 2_000)
    assert Session.status(session) == :error
  end

  test "all calls return {:error, :load_failed} when session is in error state" do
    stub(Mock, :model_load, fn _, _, _, _, _ -> raise "bad path" end)
    {:ok, session} = Session.start_link(model_path: "/bad.gguf")
    assert {:error, :load_failed} = Session.wait_until_ready(session, 2_000)
    assert {:error, :load_failed} = Session.chat(session, [%{"role" => "user", "content" => "hi"}])
    assert {:error, :load_failed} = Session.reset(session)
  end

  # ── via/2 ─────────────────────────────────────────────────────────────────

  test "via/2 returns a Registry via-tuple" do
    assert {:via, Registry, {MyReg, :npc}} = Session.via(MyReg, :npc)
  end

  # ── Utility calls ─────────────────────────────────────────────────────────

  test "model_desc/1 delegates to nif" do
    {session, model, _ctx} = start_ready()
    expect(Mock, :model_desc, fn ^model -> "llama 27B Q4_K_M" end)
    assert Session.model_desc(session) == "llama 27B Q4_K_M"
  end

  test "model_n_ctx_train/1 returns the training context length" do
    {session, model, _ctx} = start_ready()
    stub(Mock, :model_n_ctx_train, fn ^model -> 32_768 end)
    assert Session.model_n_ctx_train(session) == 32_768
  end

  test "tokenize_count/2 delegates to nif" do
    {session, model, _ctx} = start_ready()
    expect(Mock, :tokenize_count, fn ^model, "hello" -> 3 end)
    assert Session.tokenize_count(session, "hello") == 3
  end

  test "fits_in_context?/2 true when token count < n_ctx" do
    {session, model, _ctx} = start_ready(n_ctx: 10)
    stub(Mock, :tokenize_count, fn ^model, _ -> 5 end)
    assert Session.fits_in_context?(session, "short text")
  end

  test "fits_in_context?/2 false when token count >= n_ctx" do
    {session, model, _ctx} = start_ready(n_ctx: 4)
    stub(Mock, :tokenize_count, fn ^model, _ -> 5 end)
    refute Session.fits_in_context?(session, "too long")
  end

  # ── chat/3 ────────────────────────────────────────────────────────────────

  test "chat/3 returns {:ok, binary} on success" do
    {session, _model, ctx} = start_ready()
    expect(Mock, :chat_complete, fn ^ctx, _json, 0.7, 0, 40, 0.95, 1.1 -> "hi there" end)
    assert {:ok, "hi there"} =
      Session.chat(session, [%{"role" => "user", "content" => "hi"}])
  end

  test "chat/3 returns {:error, msg} when NIF raises" do
    {session, _model, _ctx} = start_ready()
    stub(Mock, :chat_complete, fn _, _, _, _, _, _, _ -> raise "inference failed" end)
    assert {:error, "inference failed"} =
      Session.chat(session, [%{"role" => "user", "content" => "hi"}])
  end

  test "chat/3 merges per-call opts over session defaults" do
    {session, _model, ctx} = start_ready()
    # Expect temperature 0.1 (override) not 0.7 (default)
    expect(Mock, :chat_complete, fn ^ctx, _json, 0.1, 0, 40, 0.95, 1.1 -> "cool" end)
    assert {:ok, "cool"} =
      Session.chat(session, [%{"role" => "user", "content" => "hi"}], temperature: 0.1)
  end

  # ── stream_chat/3 ─────────────────────────────────────────────────────────

  test "stream_chat/3 collects tokens sent to subscriber" do
    {session, _model, ctx} = start_ready()
    caller = self()

    stub(Mock, :chat_stream, fn ^ctx, subscriber, _json, _, _, _, _, _ ->
      send(subscriber, {:turboquant_token, "hello"})
      send(subscriber, {:turboquant_token, " world"})
      send(subscriber, {:turboquant_done, "hello world"})
      :ok
    end)

    tokens =
      Session.stream_chat(session, [%{"role" => "user", "content" => "hi"}])
      |> Enum.to_list()

    assert tokens == ["hello", " world"]
    _ = caller
  end

  # ── reset/1 and abort/1 ───────────────────────────────────────────────────

  test "reset/1 calls context_reset on the ctx" do
    {session, _model, ctx} = start_ready()
    expect(Mock, :context_reset, fn ^ctx -> :ok end)
    assert :ok = Session.reset(session)
  end

  test "abort/1 calls context_abort on the ctx" do
    {session, _model, ctx} = start_ready()
    expect(Mock, :context_abort, fn ^ctx -> :ok end)
    Session.abort(session)
    # Give the cast time to be processed
    :sys.get_state(session)
  end

  test "abort/1 is ignored when status is :error" do
    stub(Mock, :model_load, fn _, _, _, _, _ -> raise "bad" end)
    {:ok, session} = Session.start_link(model_path: "/bad.gguf")
    Session.wait_until_ready(session, 2_000)
    # Should not call context_abort — no ctx exists
    Session.abort(session)
    :sys.get_state(session)
  end
end
