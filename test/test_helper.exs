Application.ensure_all_started(:propcheck)
ExUnit.start(exclude: [:integration, :download])

Application.put_env(:elixir_turboquant_llm, :nif_module, TurboquantLlm.NIF.Mock)
