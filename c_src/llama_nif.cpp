#include <fine.hpp>
#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

// ── Resource types ──────────────────────────────────────────────────────────────

struct LlamaModel {
    llama_model *model       = nullptr;
    std::string  mmproj_path;               // empty → text-only model

    ~LlamaModel() {
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
    }
};
FINE_RESOURCE(LlamaModel);

struct LlamaContext {
    llama_context            *ctx         = nullptr;
    llama_model              *model_ptr   = nullptr; // borrowed; owned by LlamaModel resource
    mtmd_context             *mmproj_ctx  = nullptr; // null when text-only
    std::vector<llama_token>  cached_tokens;
    std::atomic<bool>         abort_flag{false};
    std::mutex                inference_mutex;

    ~LlamaContext() {
        if (mmproj_ctx) { mtmd_free(mmproj_ctx); mmproj_ctx = nullptr; }
        if (ctx)        { llama_free(ctx);        ctx        = nullptr; }
    }
};
FINE_RESOURCE(LlamaContext);

// ── Cache-type helper ───────────────────────────────────────────────────────────

static ggml_type parse_cache_type(const std::string &name) {
    if (name == "turbo4") return GGML_TYPE_TURBO4_0;
    if (name == "turbo3") return GGML_TYPE_TURBO3_0;
    if (name == "turbo2") return GGML_TYPE_TURBO2_0;
    if (name == "q8_0")   return GGML_TYPE_Q8_0;
    if (name == "q4_0")   return GGML_TYPE_Q4_0;
    return GGML_TYPE_F16;
}

// ── Backend lifecycle ───────────────────────────────────────────────────────────

fine::Atom backend_init(ErlNifEnv *env) { llama_backend_init(); return fine::Atom("ok"); }
FINE_NIF(backend_init, 0);

fine::Atom backend_free(ErlNifEnv *env) { llama_backend_free(); return fine::Atom("ok"); }
FINE_NIF(backend_free, 0);

// ── Model loading (IO-bound dirty NIF) ─────────────────────────────────────────
// mmproj_path: empty string → text-only; non-empty → VL model, mtmd context
// created lazily at context_create time (needs the loaded llama_model).

fine::ResourcePtr<LlamaModel> model_load(ErlNifEnv *env,
    std::string path,
    int64_t     n_gpu_layers,
    bool        use_mmap,
    bool        use_mlock,
    std::string mmproj_path) {

    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = (n_gpu_layers < 0) ? INT_MAX : static_cast<int>(n_gpu_layers);
    params.use_mmap     = use_mmap;
    params.use_mlock    = use_mlock;

    llama_model *m = llama_model_load_from_file(path.c_str(), params);
    if (!m) {
        throw std::runtime_error("failed to load model: " + path);
    }

    auto resource          = fine::make_resource<LlamaModel>();
    resource->model        = m;
    resource->mmproj_path  = std::move(mmproj_path);
    return resource;
}
FINE_NIF(model_load, ERL_NIF_DIRTY_JOB_IO_BOUND);

// ── Context creation ────────────────────────────────────────────────────────────
// If model has an mmproj_path, initialise the mtmd vision encoder here.

fine::ResourcePtr<LlamaContext> context_create(ErlNifEnv *env,
    fine::ResourcePtr<LlamaModel> model,
    int64_t     n_ctx,
    int64_t     n_threads,
    bool        flash_attn,
    std::string cache_type_k,
    std::string cache_type_v) {

    if (!model->model) {
        throw std::runtime_error("model not loaded");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = static_cast<uint32_t>(n_ctx);
    cparams.n_threads       = static_cast<int32_t>(n_threads);
    cparams.flash_attn_type = flash_attn
                                ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k          = parse_cache_type(cache_type_k);
    cparams.type_v          = parse_cache_type(cache_type_v);

    llama_context *c = llama_init_from_model(model->model, cparams);
    if (!c) {
        throw std::runtime_error("llama_init_from_model failed");
    }

    auto resource       = fine::make_resource<LlamaContext>();
    resource->ctx       = c;
    resource->model_ptr = model->model;

    if (!model->mmproj_path.empty()) {
        mtmd_context_params mparams  = mtmd_context_params_default();
        mparams.use_gpu              = true;
        mparams.n_threads            = static_cast<int>(n_threads);
        mparams.flash_attn_type      = cparams.flash_attn_type;
        mparams.warmup               = false;

        resource->mmproj_ctx = mtmd_init_from_file(
            model->mmproj_path.c_str(), model->model, mparams);

        if (!resource->mmproj_ctx) {
            // ctx destructor will free llama_context
            throw std::runtime_error(
                "mtmd_init_from_file failed for: " + model->mmproj_path);
        }
    }

    return resource;
}
FINE_NIF(context_create, ERL_NIF_DIRTY_JOB_IO_BOUND);

// ── Context control ─────────────────────────────────────────────────────────────

fine::Atom context_reset(ErlNifEnv *env, fine::ResourcePtr<LlamaContext> ctx) {
    if (ctx->ctx) {
        llama_memory_clear(llama_get_memory(ctx->ctx), true);
    }
    ctx->cached_tokens.clear();
    ctx->abort_flag.store(false, std::memory_order_relaxed);
    return fine::Atom("ok");
}
FINE_NIF(context_reset, 0);

fine::Atom context_abort(ErlNifEnv *env, fine::ResourcePtr<LlamaContext> ctx) {
    ctx->abort_flag.store(true, std::memory_order_relaxed);
    return fine::Atom("ok");
}
FINE_NIF(context_abort, 0);

// ── Inference parameters ────────────────────────────────────────────────────────

struct InferParams {
    double  temperature     = 0.7;
    int64_t max_tokens      = 0;
    int64_t top_k           = 40;
    double  top_p           = 0.95;
    double  repeat_penalty  = 1.1;
    bool    enable_thinking = false;
};

// ── UTF-8 flush helper ──────────────────────────────────────────────────────────

static std::string flush_utf8(std::vector<uint8_t> &buf, bool force) {
    std::string out;
    int emit_up_to = 0;
    for (int j = 0; j < static_cast<int>(buf.size()); ) {
        uint8_t b = buf[j];
        int seq;
        if      (b < 0x80)            seq = 1;
        else if ((b & 0xE0) == 0xC0)  seq = 2;
        else if ((b & 0xF0) == 0xE0)  seq = 3;
        else if ((b & 0xF8) == 0xF0)  seq = 4;
        else { ++j; emit_up_to = j; continue; }

        if (j + seq <= static_cast<int>(buf.size())) {
            j += seq; emit_up_to = j;
        } else if (force) {
            j = static_cast<int>(buf.size()); emit_up_to = j;
        } else {
            break;
        }
    }
    if (emit_up_to > 0) {
        out.assign(reinterpret_cast<char *>(buf.data()), emit_up_to);
        buf.erase(buf.begin(), buf.begin() + emit_up_to);
    }
    return out;
}

// ── Shared decode loop (sampler + token generation) ────────────────────────────
// Starts from n_past (after prefill). Appends generated token_ids to cached_tokens.

static std::string run_decode_loop(
    LlamaContext                             &rctx,
    const InferParams                        &ip,
    llama_pos                                 n_past,
    std::function<void(const std::string &)>  on_token) {

    llama_context     *lctx  = rctx.ctx;
    llama_model       *lmodel = rctx.model_ptr;
    const llama_vocab *vocab  = llama_model_get_vocab(lmodel);
    const uint32_t     n_ctx  = llama_n_ctx(lctx);

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler,
        llama_sampler_init_penalties(64, static_cast<float>(ip.repeat_penalty), 0.0f, 0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(static_cast<int32_t>(ip.top_k)));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(static_cast<float>(ip.top_p), 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(static_cast<float>(ip.temperature)));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string          full_response;
    const llama_token    eos = llama_vocab_eos(vocab);
    std::vector<uint8_t> utf8_buf;

    const int token_budget = (ip.max_tokens > 0)
        ? static_cast<int>(ip.max_tokens)
        : static_cast<int>(n_ctx) - static_cast<int>(n_past);

    for (int i = 0; i < token_budget; ++i) {
        if (rctx.abort_flag.load(std::memory_order_relaxed)) break;

        llama_token token_id = llama_sampler_sample(sampler, lctx, -1);
        llama_sampler_accept(sampler, token_id);
        if (token_id == eos) break;

        char piece_buf[256];
        int  piece_len = llama_token_to_piece(
            vocab, token_id, piece_buf, sizeof(piece_buf), 0, true);
        if (piece_len < 0) continue;

        for (int j = 0; j < piece_len; ++j) {
            utf8_buf.push_back(static_cast<uint8_t>(piece_buf[j]));
        }

        std::string piece = flush_utf8(utf8_buf, false);
        if (!piece.empty()) { full_response += piece; on_token(piece); }

        rctx.cached_tokens.push_back(token_id);
        llama_batch next = llama_batch_get_one(&token_id, 1);
        if (llama_decode(lctx, next) != 0) break;
    }

    if (!utf8_buf.empty()) {
        std::string tail = flush_utf8(utf8_buf, true);
        if (!tail.empty()) { full_response += tail; on_token(tail); }
    }

    llama_sampler_free(sampler);
    return full_response;
}

// ── Text-only prefill (with KV-cache prefix reuse) ─────────────────────────────

static std::string apply_text_template(const std::string &messages_json, bool enable_thinking) {
    json msgs = json::parse(messages_json);
    std::string prompt;
    for (const auto &msg : msgs) {
        std::string role    = msg.value("role",    "user");
        std::string content = msg.value("content", "");
        prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }
    if (!enable_thinking) {
        prompt += "<|im_start|>assistant\n<think>\n</think>\n";
    } else {
        prompt += "<|im_start|>assistant\n";
    }
    return prompt;
}

static std::string run_inference_text(
    LlamaContext                             &rctx,
    const std::string                        &messages_json,
    const InferParams                        &ip,
    std::function<void(const std::string &)>  on_token) {

    llama_context     *lctx  = rctx.ctx;
    llama_model       *lmodel = rctx.model_ptr;
    const llama_vocab *vocab  = llama_model_get_vocab(lmodel);
    const uint32_t     n_ctx  = llama_n_ctx(lctx);

    std::string prompt = apply_text_template(messages_json, ip.enable_thinking);

    std::vector<llama_token> prompt_tokens(prompt.size() + 16);
    int n_prompt = llama_tokenize(vocab,
        prompt.c_str(), static_cast<int32_t>(prompt.size()),
        prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()),
        true, true);
    if (n_prompt < 0) throw std::runtime_error("tokenization failed");
    prompt_tokens.resize(n_prompt);

    if (static_cast<uint32_t>(n_prompt) >= n_ctx) {
        llama_memory_clear(llama_get_memory(lctx), true);
        rctx.cached_tokens.clear();
        throw std::runtime_error("prompt exceeds context window");
    }

    // KV-cache prefix reuse
    int n_match  = 0;
    int n_cached = static_cast<int>(rctx.cached_tokens.size());
    while (n_match < n_cached && n_match < n_prompt &&
           rctx.cached_tokens[n_match] == prompt_tokens[n_match]) {
        ++n_match;
    }
    if (n_match < n_cached) {
        llama_memory_seq_rm(llama_get_memory(lctx), 0, n_match, -1);
        rctx.cached_tokens.resize(n_match);
    }

    if (n_match < n_prompt) {
        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data() + n_match, n_prompt - n_match);
        if (llama_decode(lctx, batch) != 0) {
            throw std::runtime_error("prefill (llama_decode) failed");
        }
    }
    rctx.cached_tokens = prompt_tokens;

    return run_decode_loop(rctx, ip, static_cast<llama_pos>(n_prompt), on_token);
}

// ── Multimodal prefill (mtmd) ───────────────────────────────────────────────────
// Only called when rctx.mmproj_ctx != nullptr and at least one message has "images".
// Inserts one <__media__> marker per image into the ChatML prompt, then uses
// mtmd_tokenize + mtmd_helper_eval_chunks for prefill.
// KV-cache prefix reuse is skipped for multimodal turns.

static bool messages_have_images(const std::string &messages_json) {
    json msgs = json::parse(messages_json);
    for (const auto &msg : msgs) {
        if (msg.contains("images") && !msg["images"].empty()) return true;
    }
    return false;
}

struct MmInput {
    std::string              prompt;       // text with <__media__> markers inline
    std::vector<std::string> image_paths;
};

static MmInput build_mm_prompt(const std::string &messages_json, bool enable_thinking) {
    json msgs = json::parse(messages_json);
    MmInput result;
    std::string prompt;
    const char *marker = mtmd_default_marker();   // "<__media__>"

    for (const auto &msg : msgs) {
        std::string role    = msg.value("role",    "user");
        std::string content = msg.value("content", "");

        // One marker per image, prepended before the text content
        std::string decorated;
        if (msg.contains("images")) {
            for (const auto &img : msg["images"]) {
                result.image_paths.push_back(img.get<std::string>());
                decorated += std::string(marker) + "\n";
            }
        }
        decorated += content;

        prompt += "<|im_start|>" + role + "\n" + decorated + "<|im_end|>\n";
    }

    if (!enable_thinking) {
        prompt += "<|im_start|>assistant\n<think>\n</think>\n";
    } else {
        prompt += "<|im_start|>assistant\n";
    }

    result.prompt = std::move(prompt);
    return result;
}

static std::string run_inference_mm(
    LlamaContext                             &rctx,
    const std::string                        &messages_json,
    const InferParams                        &ip,
    std::function<void(const std::string &)>  on_token) {

    llama_context *lctx = rctx.ctx;
    mtmd_context  *mctx = rctx.mmproj_ctx;

    MmInput mm = build_mm_prompt(messages_json, ip.enable_thinking);

    // Load bitmaps (thread-safe per mtmd-helper.h docs)
    std::vector<mtmd_bitmap *> bitmaps;
    bitmaps.reserve(mm.image_paths.size());
    for (const auto &path : mm.image_paths) {
        mtmd_bitmap *bm = mtmd_helper_bitmap_init_from_file(mctx, path.c_str());
        if (!bm) {
            for (auto *b : bitmaps) mtmd_bitmap_free(b);
            throw std::runtime_error("failed to load image: " + path);
        }
        bitmaps.push_back(bm);
    }

    // Tokenize text + images together
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    mtmd_input_text    text_input;
    text_input.text          = mm.prompt.c_str();
    text_input.add_special   = true;
    text_input.parse_special = true;

    int32_t ret = mtmd_tokenize(mctx, chunks, &text_input,
        const_cast<const mtmd_bitmap **>(bitmaps.data()),
        bitmaps.size());

    for (auto *b : bitmaps) mtmd_bitmap_free(b);

    if (ret != 0) {
        mtmd_input_chunks_free(chunks);
        throw std::runtime_error("mtmd_tokenize failed (code " + std::to_string(ret) + ")");
    }

    // No prefix reuse when images are present — clear the KV cache entirely
    llama_memory_clear(llama_get_memory(lctx), true);
    rctx.cached_tokens.clear();

    // Prefill all chunks (text interleaved with image embeddings)
    llama_pos new_n_past = 0;
    ret = mtmd_helper_eval_chunks(
        mctx, lctx, chunks,
        /*n_past=*/0, /*seq_id=*/0, /*n_batch=*/512,
        /*logits_last=*/true, &new_n_past);

    mtmd_input_chunks_free(chunks);

    if (ret != 0) {
        throw std::runtime_error("mtmd_helper_eval_chunks failed (code " + std::to_string(ret) + ")");
    }

    return run_decode_loop(rctx, ip, new_n_past, on_token);
}

// ── Inference dispatcher ────────────────────────────────────────────────────────

static std::string run_inference(
    LlamaContext                             &rctx,
    const std::string                        &messages_json,
    const InferParams                        &ip,
    std::function<void(const std::string &)>  on_token) {

    if (rctx.mmproj_ctx && messages_have_images(messages_json)) {
        return run_inference_mm(rctx, messages_json, ip, on_token);
    }
    return run_inference_text(rctx, messages_json, ip, on_token);
}

// ── Model utilities ─────────────────────────────────────────────────────────────

// Returns the model description string (architecture + size).
std::string model_desc(ErlNifEnv *env, fine::ResourcePtr<LlamaModel> model) {
    char buf[256];
    llama_model_desc(model->model, buf, sizeof(buf));
    return std::string(buf);
}
FINE_NIF(model_desc, 0);

// Returns the context length the model was trained on.
int64_t model_n_ctx_train(ErlNifEnv *env, fine::ResourcePtr<LlamaModel> model) {
    return static_cast<int64_t>(llama_model_n_ctx_train(model->model));
}
FINE_NIF(model_n_ctx_train, 0);

// Returns the number of tokens the given text tokenises to.
// Useful for checking whether a prompt fits in the context window before
// paying the cost of a dirty-scheduler inference call.
int64_t tokenize_count(ErlNifEnv *env,
    fine::ResourcePtr<LlamaModel> model,
    std::string                   text) {

    const llama_vocab *vocab = llama_model_get_vocab(model->model);
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(vocab,
        text.c_str(), static_cast<int32_t>(text.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()),
        /*add_special=*/true, /*parse_special=*/true);
    if (n < 0) throw std::runtime_error("tokenization failed");
    return static_cast<int64_t>(n);
}
FINE_NIF(tokenize_count, 0);

// ── chat_complete – blocking, returns full response ─────────────────────────────

std::string chat_complete(ErlNifEnv *env,
    fine::ResourcePtr<LlamaContext> ctx,
    std::string messages_json,
    double      temperature,
    int64_t     max_tokens,
    int64_t     top_k,
    double      top_p,
    double      repeat_penalty) {

    std::lock_guard<std::mutex> lock(ctx->inference_mutex);
    ctx->abort_flag.store(false, std::memory_order_relaxed);

    InferParams ip;
    ip.temperature    = temperature;
    ip.max_tokens     = max_tokens;
    ip.top_k          = top_k;
    ip.top_p          = top_p;
    ip.repeat_penalty = repeat_penalty;

    return run_inference(*ctx, messages_json, ip, [](const std::string &) {});
}
FINE_NIF(chat_complete, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// ── chat_stream – sends {:turboquant_token, bin} + {:turboquant_done, bin} ──────

fine::Atom chat_stream(ErlNifEnv *env,
    fine::ResourcePtr<LlamaContext> ctx,
    ErlNifPid   subscriber,
    std::string messages_json,
    double      temperature,
    int64_t     max_tokens,
    int64_t     top_k,
    double      top_p,
    double      repeat_penalty) {

    std::lock_guard<std::mutex> lock(ctx->inference_mutex);
    ctx->abort_flag.store(false, std::memory_order_relaxed);

    InferParams ip;
    ip.temperature    = temperature;
    ip.max_tokens     = max_tokens;
    ip.top_k          = top_k;
    ip.top_p          = top_p;
    ip.repeat_penalty = repeat_penalty;

    auto send_binary = [&](const char *atom, const std::string &data) {
        ErlNifEnv   *menv = enif_alloc_env();
        ERL_NIF_TERM bin;
        unsigned char *buf = enif_make_new_binary(menv, data.size(), &bin);
        std::memcpy(buf, data.data(), data.size());
        ERL_NIF_TERM msg = enif_make_tuple2(menv,
            enif_make_atom(menv, atom), bin);
        enif_send(nullptr, &subscriber, menv, msg);
        enif_free_env(menv);
    };

    try {
        std::string full = run_inference(*ctx, messages_json, ip,
            [&](const std::string &piece) { send_binary("turboquant_token", piece); });
        send_binary("turboquant_done", full);
    } catch (const std::exception &e) {
        send_binary("turboquant_error", std::string(e.what()));
    }
    return fine::Atom("ok");
}
FINE_NIF(chat_stream, ERL_NIF_DIRTY_JOB_CPU_BOUND);

FINE_INIT("Elixir.TurboquantLlm.NIF");
