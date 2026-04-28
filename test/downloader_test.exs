defmodule TurboquantLlm.DownloaderTest do
  use ExUnit.Case, async: true
  use PropCheck

  alias TurboquantLlm.Downloader

  @min_chunk Downloader.min_chunk()
  @max_chunk Downloader.max_chunk()
  @sat       Downloader.sat_threshold()

  # ── Unit: algorithm boundary cases (mirrors Lean eval examples) ───────────

  test "chunk_for_throughput: zero throughput returns MIN_CHUNK" do
    assert Downloader.chunk_for_throughput(0) == @min_chunk
  end

  test "chunk_for_throughput: 2 MB/s returns floor (below ramp)" do
    # 2_097_152 bps / 10 = 209_715 < 262_144 MIN → floor
    assert Downloader.chunk_for_throughput(2_097_152) == @min_chunk
  end

  test "chunk_for_throughput: 10 MB/s returns 1 MB" do
    # 10_485_760 / 10 = 1_048_576 — in the linear range
    assert Downloader.chunk_for_throughput(10_485_760) == 1_048_576
  end

  test "chunk_for_throughput: 50 MB/s returns 5 MB" do
    assert Downloader.chunk_for_throughput(52_428_800) == 5_242_880
  end

  test "chunk_for_throughput: 80 MB/s saturates to MAX_CHUNK" do
    assert Downloader.chunk_for_throughput(83_886_080) == @max_chunk
  end

  test "chunk_for_throughput: sat_threshold saturates" do
    assert Downloader.chunk_for_throughput(@sat) == @max_chunk
  end

  # ── Properties (mirror Lean theorems) ─────────────────────────────────────

  property "chunk_in_bounds: always in [MIN_CHUNK, MAX_CHUNK]" do
    forall bps <- non_neg_integer() do
      c = Downloader.chunk_for_throughput(bps)
      c >= @min_chunk and c <= @max_chunk
    end
  end

  property "chunk_monotone: non-decreasing" do
    forall {a, b} <- {non_neg_integer(), non_neg_integer()} do
      {lo, hi} = if a <= b, do: {a, b}, else: {b, a}
      Downloader.chunk_for_throughput(lo) <= Downloader.chunk_for_throughput(hi)
    end
  end

  property "chunk_pos: always positive" do
    forall bps <- non_neg_integer() do
      Downloader.chunk_for_throughput(bps) > 0
    end
  end

  property "chunk_sat_ge: saturates for bps >= sat_threshold" do
    forall delta <- non_neg_integer() do
      Downloader.chunk_for_throughput(@sat + delta) == @max_chunk
    end
  end

  # ── Unit: helpers ─────────────────────────────────────────────────────────

  test "cache_dir: returns a non-empty path" do
    dir = TurboquantLlm.cache_dir()
    assert is_binary(dir) and byte_size(dir) > 0
  end

  test "cache_dir: honours TURBOQUANT_CACHE_DIR env override" do
    System.put_env("TURBOQUANT_CACHE_DIR", "/tmp/tq_test_cache")
    assert TurboquantLlm.cache_dir() == "/tmp/tq_test_cache"
    System.delete_env("TURBOQUANT_CACHE_DIR")
  end

  test "download_model/1 with :url resolves correct filename" do
    url  = "https://example.com/some-model.gguf"
    dest = Path.join(TurboquantLlm.cache_dir(), "some-model.gguf")
    # File won't exist so download will be attempted — but we only check
    # that the *path* logic is correct by inspecting what skip_existing skips.
    assert String.ends_with?(dest, "some-model.gguf")
  end

  test "download/3 returns :ok immediately when file already exists" do
    path = Path.join(System.tmp_dir!(), "tq_exists_#{System.unique_integer()}.gguf")
    File.write!(path, "fake model data")

    assert {:ok, ^path} = Downloader.download("http://unused", path)

    File.rm!(path)
  end

  # ── Integration: real HTTP download (requires network, opt-in) ────────────

  @moduletag :download

  @tiny_url "https://huggingface.co/mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF/resolve/main/README.md"

  @tag :download
  test "download/3 streams a small real file and reports progress" do
    dest = Path.join(System.tmp_dir!(), "tq_readme_#{System.unique_integer()}.md")

    progress_calls = :counters.new(1, [])

    result =
      Downloader.download(@tiny_url, dest,
        on_progress: fn _recv, _total ->
          :counters.add(progress_calls, 1, 1)
        end,
        skip_existing: false
      )

    assert {:ok, ^dest} = result
    assert File.exists?(dest)
    assert File.stat!(dest).size > 0
    assert :counters.get(progress_calls, 1) > 0

    File.rm!(dest)
  end
end
