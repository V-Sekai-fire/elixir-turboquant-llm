defmodule TurboquantLlm.Downloader do
  @moduledoc """
  Streaming GGUF downloader with adaptive chunk-size reporting.

  The chunk sizing algorithm is ported from turboquant-project's main.gd and
  formally verified in lean/DownloadChunk.lean:

      chunk = clamp(throughput_bps / 10, MIN_CHUNK, MAX_CHUNK)

  One chunk targets ~100 ms of data at the measured transfer rate.
  """

  @min_chunk 256 * 1024        # 256 KB  — floor
  @max_chunk 8 * 1024 * 1024   # 8 MB   — ceiling
  @sat_threshold @max_chunk * 10

  # ── Public API ──────────────────────────────────────────────────────────────

  @doc "Minimum chunk size in bytes (256 KB)."
  def min_chunk, do: @min_chunk

  @doc "Maximum chunk size in bytes (8 MB)."
  def max_chunk, do: @max_chunk

  @doc "Throughput (bps) at which chunk size saturates to MAX_CHUNK."
  def sat_threshold, do: @sat_threshold

  @doc """
  Returns the recommended chunk size for a measured `throughput_bps`.
  Mirrors the Lean-proved `chunkForThroughput` from turboquant-project.
  """
  def chunk_for_throughput(bps) when is_integer(bps) and bps >= 0 do
    t = div(bps, 10)
    cond do
      t <= @min_chunk -> @min_chunk
      t >= @max_chunk -> @max_chunk
      true            -> t
    end
  end

  @doc """
  Downloads `url` to `dest_path` with streaming.

  Options:
    * `:on_progress` — `fun(bytes_received, total_bytes)` called after each chunk.
      `total_bytes` is 0 if the server omits Content-Length.
    * `:timeout_ms`  — per-chunk receive timeout in ms (default: 600_000).
    * `:skip_existing` — if `true` (default) and file already exists, return immediately.

  Returns `{:ok, dest_path}` or `{:error, reason}`.
  """
  def download(url, dest_path, opts \\ []) do
    on_progress  = Keyword.get(opts, :on_progress, fn _, _ -> :ok end)
    timeout_ms   = Keyword.get(opts, :timeout_ms, 600_000)
    skip_existing = Keyword.get(opts, :skip_existing, true)

    if skip_existing and File.exists?(dest_path) do
      {:ok, dest_path}
    else
      Application.ensure_all_started(:inets)
      Application.ensure_all_started(:ssl)
      File.mkdir_p!(Path.dirname(dest_path))
      do_download(String.to_charlist(url), dest_path, on_progress, timeout_ms)
    end
  end

  # ── Internals ───────────────────────────────────────────────────────────────

  defp do_download(url, dest_path, on_progress, timeout_ms) do
    total = get_content_length(url)

    {:ok, ref} =
      :httpc.request(
        :get,
        {url, []},
        [{:timeout, :infinity}, {:connect_timeout, 30_000}, {:autoredirect, true}],
        [{:stream_to, self()}, {:sync, false}]
      )

    case File.open(dest_path, [:write, :binary]) do
      {:ok, io} ->
        result = recv_stream(io, ref, 0, total, on_progress, timeout_ms)
        File.close(io)

        case result do
          :ok ->
            {:ok, dest_path}

          {:error, _} = err ->
            File.rm(dest_path)
            err
        end

      {:error, reason} ->
        {:error, {:open_failed, reason}}
    end
  end

  defp recv_stream(io, ref, received, total, on_progress, timeout_ms) do
    receive do
      {:http, {^ref, :stream_start, _headers}} ->
        recv_stream(io, ref, received, total, on_progress, timeout_ms)

      {:http, {^ref, :stream, chunk}} ->
        IO.binwrite(io, chunk)
        new_received = received + byte_size(chunk)
        on_progress.(new_received, total)
        recv_stream(io, ref, new_received, total, on_progress, timeout_ms)

      {:http, {^ref, :stream_end, _trailers}} ->
        :ok

      {:http, {^ref, {:error, reason}}} ->
        {:error, reason}
    after
      timeout_ms -> {:error, :timeout}
    end
  end

  defp get_content_length(url) do
    case :httpc.request(
           :head,
           {url, []},
           [{:timeout, 15_000}, {:connect_timeout, 10_000}, {:autoredirect, true}],
           []
         ) do
      {:ok, {{_, 200, _}, headers, _}} ->
        case List.keyfind(headers, ~c"content-length", 0) do
          {_, len} -> len |> to_string() |> String.to_integer()
          nil -> 0
        end

      _ ->
        0
    end
  end
end
