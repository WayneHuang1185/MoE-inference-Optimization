#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <inttypes.h>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct trace_args {
    std::string model;
    std::string prompt = "Hello";
    std::string prompt_file;
    std::string output = "expert_trace.jsonl";
    int32_t n_predict = 0;
    int32_t n_ctx = 4096;
    int32_t n_threads = (int32_t) std::max(1u, std::thread::hardware_concurrency());
    int32_t n_gpu_layers = 0;
    bool no_mmap = false;
    bool quiet = false;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s -m model.gguf [-p prompt | -f prompt.txt] [-o trace.jsonl] [options]\n"
        "\n"
        "Options:\n"
        "  -m, --model PATH       GGUF model path\n"
        "  -p, --prompt TEXT      prompt text\n"
        "  -f, --file PATH        prompt file\n"
        "  -o, --output PATH      JSONL output path, or '-' for stdout. Default: expert_trace.jsonl\n"
        "  -n, --predict N        generated tokens to trace after the prompt. Default: 0\n"
        "  -c, --ctx-size N       context size. Default: 4096\n"
        "  -t, --threads N        CPU threads. Default: hardware concurrency\n"
        "  -ngl N                 GPU layers to offload. Default: 0\n"
        "  --no-mmap              disable llama.cpp mmap model loading\n"
        "  --quiet                do not print generated text\n"
        "  -h, --help             show this help\n"
        "\n"
        "Output JSONL fields:\n"
        "  decode, phase, layer, batch_index, pos, token, piece, n_experts, prob_sum, expert_probs\n"
        "\n"
        "Note:\n"
        "  expert_probs is the ffn_moe_probs softmax output. Array index == expert id.\n",
        argv0);
}

static void quiet_log_callback(enum ggml_log_level, const char *, void *) {
}

static bool parse_i32(const char * s, int32_t & out) {
    char * end = nullptr;
    errno = 0;
    long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return false;
    }
    out = (int32_t) v;
    return true;
}

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open prompt file: " + path);
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static bool parse_args(int argc, char ** argv, trace_args & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need_value = [&](const char * opt) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires a value\n", opt);
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (a == "-m" || a == "--model") {
            const char * v = need_value(a.c_str());
            if (!v) return false;
            args.model = v;
        } else if (a == "-p" || a == "--prompt") {
            const char * v = need_value(a.c_str());
            if (!v) return false;
            args.prompt = v;
        } else if (a == "-f" || a == "--file") {
            const char * v = need_value(a.c_str());
            if (!v) return false;
            args.prompt_file = v;
        } else if (a == "-o" || a == "--output") {
            const char * v = need_value(a.c_str());
            if (!v) return false;
            args.output = v;
        } else if (a == "-n" || a == "--predict" || a == "--n-predict") {
            const char * v = need_value(a.c_str());
            if (!v || !parse_i32(v, args.n_predict)) return false;
        } else if (a == "-c" || a == "--ctx-size") {
            const char * v = need_value(a.c_str());
            if (!v || !parse_i32(v, args.n_ctx)) return false;
        } else if (a == "-t" || a == "--threads") {
            const char * v = need_value(a.c_str());
            if (!v || !parse_i32(v, args.n_threads)) return false;
        } else if (a == "-ngl" || a == "--gpu-layers" || a == "--n-gpu-layers") {
            const char * v = need_value(a.c_str());
            if (!v || !parse_i32(v, args.n_gpu_layers)) return false;
        } else if (a == "--no-mmap") {
            args.no_mmap = true;
        } else if (a == "--quiet") {
            args.quiet = true;
        } else {
            std::fprintf(stderr, "error: unknown argument: %s\n", a.c_str());
            return false;
        }
    }

    if (args.model.empty()) {
        std::fprintf(stderr, "error: --model is required\n");
        return false;
    }
    if (args.n_predict < 0 || args.n_ctx <= 0 || args.n_threads <= 0) {
        std::fprintf(stderr, "error: invalid numeric argument\n");
        return false;
    }
    return true;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char) c);
                }
        }
    }
    return out;
}

static std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buf(64);
    int n = llama_token_to_piece(vocab, token, buf.data(), (int32_t) buf.size(), 0, true);
    if (n < 0) {
        buf.resize((size_t) -n);
        n = llama_token_to_piece(vocab, token, buf.data(), (int32_t) buf.size(), 0, true);
    }
    if (n < 0) {
        return "";
    }
    return std::string(buf.data(), (size_t) n);
}

static bool parse_probs_layer(const char * name, int & layer) {
    const char prefix[] = "ffn_moe_probs-";
    const size_t n = sizeof(prefix) - 1;
    if (std::strncmp(name, prefix, n) != 0) {
        return false;
    }
    char * end = nullptr;
    long v = std::strtol(name + n, &end, 10);
    if (end == name + n || *end != '\0') {
        return false;
    }
    layer = (int) v;
    return true;
}

static float tensor_f32_at(const uint8_t * data, ggml_type type, const size_t * nb, int64_t i0, int64_t i1) {
    const size_t off = (size_t) i1 * nb[1] + (size_t) i0 * nb[0];
    switch (type) {
        case GGML_TYPE_F32:  return *(const float *) (data + off);
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) (data + off));
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) (data + off));
        case GGML_TYPE_I32:  return (float) *(const int32_t *) (data + off);
        case GGML_TYPE_I64:  return (float) *(const int64_t *) (data + off);
        case GGML_TYPE_I16:  return (float) *(const int16_t *) (data + off);
        case GGML_TYPE_I8:   return (float) *(const int8_t  *) (data + off);
        default:             return 0.0f;
    }
}

class expert_tracer {
public:
    expert_tracer(const llama_vocab * vocab, const std::string & path) : vocab(vocab) {
        if (path == "-") {
            out = stdout;
            owns = false;
        } else {
            out = std::fopen(path.c_str(), "wb");
            owns = true;
        }
        if (!out) {
            throw std::runtime_error("failed to open output: " + path);
        }
    }

    ~expert_tracer() {
        if (out) {
            std::fflush(out);
            if (owns) {
                std::fclose(out);
            }
        }
    }

    expert_tracer(const expert_tracer &) = delete;
    expert_tracer & operator=(const expert_tracer &) = delete;

    void begin_decode(int decode_idx, std::string phase, const std::vector<llama_token> & toks, llama_pos start_pos) {
        decode = decode_idx;
        current_phase = std::move(phase);
        tokens = toks;
        positions.resize(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            positions[i] = start_pos + (llama_pos) i;
        }
    }

    bool cb_eval(ggml_tensor * t, bool ask) {
        int layer = -1;
        const bool want = parse_probs_layer(t->name, layer);
        if (ask) {
            return want;
        }
        if (!want) {
            return true;
        }
        write_tensor(t, layer);
        return true;
    }

private:
    const llama_vocab * vocab = nullptr;
    FILE * out = nullptr;
    bool owns = false;

    int decode = 0;
    std::string current_phase;
    std::vector<llama_token> tokens;
    std::vector<llama_pos> positions;
    std::vector<uint8_t> staging;

    void write_tensor(ggml_tensor * t, int layer) {
        if (t->ne[0] <= 0 || t->ne[1] <= 0) {
            return;
        }

        uint8_t * data = nullptr;
        if (ggml_backend_buffer_is_host(t->buffer)) {
            data = (uint8_t *) t->data;
        } else {
            const size_t n_bytes = ggml_nbytes(t);
            staging.resize(n_bytes);
            ggml_backend_tensor_get(t, staging.data(), 0, n_bytes);
            data = staging.data();
        }

        const int64_t n_experts = t->ne[0];
        const int64_t n_tok_tensor = t->ne[1];
        const int64_t n_tok = std::min<int64_t>(n_tok_tensor, (int64_t) tokens.size());

        for (int64_t it = 0; it < n_tok; ++it) {
            const llama_token tok = tokens[(size_t) it];
            const std::string piece = token_piece(vocab, tok);
            double prob_sum = 0.0;
            for (int64_t expert = 0; expert < n_experts; ++expert) {
                prob_sum += tensor_f32_at(data, t->type, t->nb, expert, it);
            }

            std::fprintf(out,
                "{\"decode\":%d,\"phase\":\"%s\",\"layer\":%d,"
                "\"batch_index\":%" PRId64 ",\"pos\":%d,\"token\":%d,"
                "\"piece\":\"%s\",\"n_experts\":%" PRId64 ",\"prob_sum\":%.9g,\"expert_probs\":[",
                decode,
                json_escape(current_phase).c_str(),
                layer,
                it,
                (int) positions[(size_t) it],
                tok,
                json_escape(piece).c_str(),
                n_experts,
                prob_sum);

            for (int64_t expert = 0; expert < n_experts; ++expert) {
                if (expert > 0) {
                    std::fputc(',', out);
                }
                std::fprintf(out, "%.9g", tensor_f32_at(data, t->type, t->nb, expert, it));
            }
            std::fputs("]}\n", out);
        }
    }
};

static bool trace_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    return ((expert_tracer *) user_data)->cb_eval(t, ask);
}

static llama_batch make_batch(std::vector<llama_token> & tokens, std::vector<llama_pos> & pos, std::vector<int32_t> & n_seq_id,
        std::vector<llama_seq_id> & seq_storage, std::vector<llama_seq_id *> & seq_id, std::vector<int8_t> & logits,
        llama_pos start_pos, bool output_all) {
    const int32_t n = (int32_t) tokens.size();
    pos.resize((size_t) n);
    n_seq_id.assign((size_t) n, 1);
    seq_storage.assign((size_t) n, 0);
    seq_id.resize((size_t) n);
    logits.assign((size_t) n, 0);

    for (int32_t i = 0; i < n; ++i) {
        pos[(size_t) i] = start_pos + i;
        seq_id[(size_t) i] = &seq_storage[(size_t) i];
        logits[(size_t) i] = output_all || i == n - 1 ? 1 : 0;
    }

    llama_batch batch;
    batch.n_tokens = n;
    batch.token = tokens.data();
    batch.embd = nullptr;
    batch.pos = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id = seq_id.data();
    batch.logits = logits.data();
    return batch;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    trace_args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 1;
    }

    if (!args.prompt_file.empty()) {
        try {
            args.prompt = read_file(args.prompt_file);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
    }

    if (args.quiet) {
        llama_log_set(quiet_log_callback, nullptr);
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = args.n_gpu_layers;
    model_params.use_mmap = !args.no_mmap;

    llama_model * model = llama_model_load_from_file(args.model.c_str(), model_params);
    if (!model) {
        std::fprintf(stderr, "error: unable to load model: %s\n", args.model.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, args.prompt.c_str(), (int32_t) args.prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        std::fprintf(stderr, "error: failed to tokenize prompt\n");
        llama_model_free(model);
        return 1;
    }

    std::vector<llama_token> prompt_tokens((size_t) n_prompt);
    if (llama_tokenize(vocab, args.prompt.c_str(), (int32_t) args.prompt.size(), prompt_tokens.data(), n_prompt, true, true) < 0) {
        std::fprintf(stderr, "error: failed to tokenize prompt\n");
        llama_model_free(model);
        return 1;
    }

    const int32_t needed_ctx = n_prompt + std::max<int32_t>(1, args.n_predict);
    if (args.n_ctx < needed_ctx) {
        args.n_ctx = needed_ctx;
    }

    expert_tracer tracer(vocab, args.output);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = args.n_ctx;
    ctx_params.n_batch = std::max<int32_t>(n_prompt, 1);
    ctx_params.n_ubatch = ctx_params.n_batch;
    ctx_params.n_threads = args.n_threads;
    ctx_params.n_threads_batch = args.n_threads;
    ctx_params.no_perf = true;
    ctx_params.cb_eval = trace_cb_eval;
    ctx_params.cb_eval_user_data = &tracer;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::fprintf(stderr, "error: failed to create llama_context\n");
        llama_model_free(model);
        return 1;
    }

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_storage;
    std::vector<llama_seq_id *> seq_id;
    std::vector<int8_t> logits;

    tracer.begin_decode(0, "prompt", prompt_tokens, 0);
    llama_batch batch = make_batch(prompt_tokens, pos, n_seq_id, seq_storage, seq_id, logits, 0, true);
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "error: prompt decode failed\n");
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_token next = LLAMA_TOKEN_NULL;
    if (args.n_predict > 0) {
        next = llama_sampler_sample(smpl, ctx, -1);
    }
    llama_pos cur_pos = (llama_pos) n_prompt;

    if (!args.quiet) {
        std::fprintf(stderr, "%s", args.prompt.c_str());
    }

    for (int32_t i = 0; i < args.n_predict; ++i) {
        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }

        if (!args.quiet) {
            const std::string piece = token_piece(vocab, next);
            std::fprintf(stderr, "%s", piece.c_str());
            std::fflush(stderr);
        }

        std::vector<llama_token> one = { next };
        tracer.begin_decode(i + 1, "generation", one, cur_pos);
        batch = make_batch(one, pos, n_seq_id, seq_storage, seq_id, logits, cur_pos, true);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "\nerror: generation decode failed at token %d\n", i);
            break;
        }

        ++cur_pos;
        next = llama_sampler_sample(smpl, ctx, -1);
    }

    if (!args.quiet) {
        std::fprintf(stderr, "\n");
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
