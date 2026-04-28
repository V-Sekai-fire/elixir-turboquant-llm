defmodule TurboquantLlm.Application do
  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    case load_nif_backend() do
      :ok ->
        children = []
        opts = [strategy: :one_for_one, name: TurboquantLlm.Supervisor]
        Supervisor.start_link(children, opts)

      {:error, reason} ->
        {:error, reason}
    end
  end

  @impl true
  def stop(_state) do
    try do
      TurboquantLlm.NIF.backend_free()
    rescue
      _ -> :ok
    end

    :ok
  end

  defp load_nif_backend do
    try do
      TurboquantLlm.NIF.backend_init()
      :ok
    rescue
      e in ErlangError ->
        msg = Exception.message(e)
        if String.contains?(msg, "not_loaded") do
          Logger.error("""
          TurboquantLLM: NIF not loaded. Run `mix compile` first.
          If the build failed, set LLAMA_CPP_DIR and recompile:
            LLAMA_CPP_DIR=/path/to/turboquant-godot/thirdparty/llama_cpp mix compile
          """)
          {:error, :nif_not_loaded}
        else
          {:error, {:nif_init_failed, msg}}
        end
    end
  end
end
