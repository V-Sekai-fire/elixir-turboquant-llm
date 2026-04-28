defmodule TurboquantLlm.NIF.Behaviour do
  @type resource :: term()

  @callback backend_init() :: :ok
  @callback backend_free() :: :ok

  @callback model_load(
              path         :: String.t(),
              n_gpu_layers :: integer(),
              use_mmap     :: boolean(),
              use_mlock    :: boolean(),
              mmproj_path  :: String.t()
            ) :: resource()

  @callback model_desc(model :: resource()) :: binary()
  @callback model_n_ctx_train(model :: resource()) :: non_neg_integer()

  @callback context_create(
              model        :: resource(),
              n_ctx        :: integer(),
              n_threads    :: integer(),
              flash_attn   :: boolean(),
              cache_type_k :: String.t(),
              cache_type_v :: String.t()
            ) :: resource()

  @callback context_reset(ctx :: resource()) :: :ok
  @callback context_abort(ctx :: resource()) :: :ok

  @callback tokenize_count(model :: resource(), text :: String.t()) :: non_neg_integer()

  @callback chat_complete(
              ctx            :: resource(),
              messages_json  :: String.t(),
              temperature    :: float(),
              max_tokens     :: integer(),
              top_k          :: integer(),
              top_p          :: float(),
              repeat_penalty :: float()
            ) :: binary()

  @callback chat_stream(
              ctx            :: resource(),
              subscriber     :: pid(),
              messages_json  :: String.t(),
              temperature    :: float(),
              max_tokens     :: integer(),
              top_k          :: integer(),
              top_p          :: float(),
              repeat_penalty :: float()
            ) :: :ok
end
