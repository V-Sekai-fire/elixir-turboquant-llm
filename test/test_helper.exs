Application.ensure_all_started(:propcheck)
ExUnit.start(exclude: [:integration, :download])
