#include "cuinfer/cuda_utils.h"
#include "cuinfer/model.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

static void print_usage(const char* prog) {
    printf("Usage: %s <model.gguf> [prompt] [max_tokens] [--dump-logits [file]]\n\n", prog);
    printf("  model.gguf      Path to a GGUF model file\n");
    printf("  prompt          Text prompt (default: \"Once upon a time\")\n");
    printf("  max_tokens      Max tokens to generate (default: 50)\n");
    printf("  --dump-logits   Dump top-20 logits after prompt prefill to JSON\n");
    printf("                  Optional: specify output file (default: cuinfer_logits.json)\n");
}

// ── Logit Dump Helpers ──────────────────────────────────────────────────
//
// When --dump-logits is passed, we run only the prefill forward pass
// (no autoregressive generation), extract the top-K logit values and
// their token indices from the last position, and write them to a JSON
// file. This allows comparison against a PyTorch reference
// (scripts/verify_logits.py).

struct LogitEntry {
    int rank;
    int token_id;
    float logit_value;
    std::string token_string;
};

/// Copy the last-position logits from GPU to CPU, find the top-K,
/// and return them sorted by descending logit value.
static std::vector<LogitEntry> extract_top_logits(
    const cuinfer::Tensor& logits_tensor,
    int seq_len,
    int vocab_size,
    const cuinfer::Tokenizer& tokenizer,
    int top_k)
{
    // logits_tensor shape: [seq_len, vocab_size], on GPU.
    // We want the last row: logits_tensor[seq_len - 1, :].
    const int last_pos = seq_len - 1;

    // Copy the entire last row to host
    std::vector<float> host_logits(vocab_size);
    const float* device_ptr = logits_tensor.data_as<float>()
                              + static_cast<size_t>(last_pos) * vocab_size;
    CUINFER_CHECK_CUDA(cudaMemcpy(
        host_logits.data(), device_ptr,
        vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Build index array and partial sort to get top-K
    std::vector<int> indices(vocab_size);
    for (int i = 0; i < vocab_size; ++i) indices[i] = i;

    int k = std::min(top_k, vocab_size);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) {
                          return host_logits[a] > host_logits[b];
                      });

    // Build result entries
    std::vector<LogitEntry> entries;
    entries.reserve(k);
    for (int r = 0; r < k; ++r) {
        int idx = indices[r];
        LogitEntry e;
        e.rank = r;
        e.token_id = idx;
        e.logit_value = host_logits[idx];
        e.token_string = tokenizer.decode({static_cast<int32_t>(idx)});
        entries.push_back(std::move(e));
    }

    return entries;
}

/// Escape a string for JSON output (handle quotes, backslashes, control chars).
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20 || c >= 0x80) {
                    // Escape control chars and non-ASCII bytes as \uXXXX
                    // to guarantee valid JSON (some tokens contain incomplete
                    // UTF-8 sequences from the BPE vocabulary).
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

/// Write the logit dump to a JSON file matching the format expected
/// by scripts/verify_logits.py.
static void write_logits_json(
    const std::string& path,
    const std::string& prompt,
    const std::vector<int32_t>& input_token_ids,
    const std::vector<std::string>& input_token_strings,
    const std::vector<LogitEntry>& entries)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    f << "{\n";
    f << "  \"prompt\": \"" << json_escape(prompt) << "\",\n";

    // Input token IDs
    f << "  \"input_token_ids\": [";
    for (size_t i = 0; i < input_token_ids.size(); ++i) {
        if (i > 0) f << ", ";
        f << input_token_ids[i];
    }
    f << "],\n";

    // Input token strings
    f << "  \"input_token_strings\": [";
    for (size_t i = 0; i < input_token_strings.size(); ++i) {
        if (i > 0) f << ", ";
        f << "\"" << json_escape(input_token_strings[i]) << "\"";
    }
    f << "],\n";

    f << "  \"num_input_tokens\": " << input_token_ids.size() << ",\n";
    f << "  \"logits_position\": " << (input_token_ids.size() - 1) << ",\n";
    f << "  \"top_k\": " << entries.size() << ",\n";

    // Top logits array
    f << "  \"top_logits\": [\n";
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        f << "    {\n";
        f << "      \"rank\": " << e.rank << ",\n";
        f << "      \"token_id\": " << e.token_id << ",\n";
        f << "      \"token_string\": \"" << json_escape(e.token_string) << "\",\n";

        // Write logit value with enough precision for comparison.
        // Use %g to avoid trailing zeros, but ensure at least 6 sig figs.
        char buf[64];
        snprintf(buf, sizeof(buf), "%.6f", e.logit_value);
        f << "      \"logit_value\": " << buf << "\n";

        f << "    }";
        if (i + 1 < entries.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";

    f.close();
    printf("Logits written to: %s\n", path.c_str());
}


// ── Main ────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    printf("CuInfer — Lightweight LLM Inference Engine\n\n");

    try {
        cuinfer::print_device_info();
        printf("\n");

        if (argc < 2) {
            print_usage(argv[0]);
            return 0;
        }

        // ── Parse arguments ──────────────────────────────────────
        // Positional: model_path [prompt] [max_tokens]
        // Flags: --dump-logits [output_file]
        //
        // We scan argv for --dump-logits first, then treat the
        // remaining positional args in order.

        bool dump_logits = false;
        std::string logits_output_path = "cuinfer_logits.json";
        std::vector<std::string> positional_args;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--dump-logits") == 0) {
                dump_logits = true;
                // Check if next arg is a filename (not another flag, not a .gguf)
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    // Peek: if it looks like a .json file, use it as the output path
                    std::string next = argv[i + 1];
                    if (next.size() > 5 &&
                        next.substr(next.size() - 5) == ".json") {
                        logits_output_path = next;
                        ++i;  // consume this arg
                    }
                }
            } else {
                positional_args.push_back(argv[i]);
            }
        }

        if (positional_args.empty()) {
            print_usage(argv[0]);
            return 0;
        }

        const std::string model_path = positional_args[0];
        const std::string prompt = (positional_args.size() >= 2)
                                       ? positional_args[1]
                                       : "Once upon a time";
        const int max_tokens = (positional_args.size() >= 3)
                                   ? std::stoi(positional_args[2])
                                   : 50;

        // Load model
        cuinfer::TransformerModel model(model_path);
        printf("\n");

        if (dump_logits) {
            // ── Dump-logits mode ─────────────────────────────────
            // Run only the prefill forward pass and extract logits
            // from the last position for comparison with PyTorch.

            printf("Mode: --dump-logits\n");
            printf("Prompt: \"%s\"\n", prompt.c_str());

            // Tokenize the prompt
            std::vector<int32_t> token_ids =
                model.tokenizer().encode(prompt, true);

            printf("Input tokens (%zu): [", token_ids.size());
            for (size_t i = 0; i < token_ids.size(); ++i) {
                if (i > 0) printf(", ");
                printf("%d", token_ids[i]);
            }
            printf("]\n");

            // Decode each token individually for the dump
            std::vector<std::string> token_strings;
            token_strings.reserve(token_ids.size());
            for (int32_t tid : token_ids) {
                token_strings.push_back(
                    model.tokenizer().decode({tid}));
            }

            printf("Token strings: [");
            for (size_t i = 0; i < token_strings.size(); ++i) {
                if (i > 0) printf(", ");
                printf("\"%s\"", token_strings[i].c_str());
            }
            printf("]\n\n");

            // Forward pass (prefill)
            model.reset();
            cuinfer::Tensor logits = model.forward(token_ids);

            // Extract top-20 logits from the last position
            constexpr int TOP_K = 20;
            auto entries = extract_top_logits(
                logits,
                static_cast<int>(token_ids.size()),
                model.config().vocab_size,
                model.tokenizer(),
                TOP_K);

            // Print to console
            printf("Top-%d logits at last position (next-token prediction):\n",
                   TOP_K);
            for (const auto& e : entries) {
                printf("  #%2d: [%6d] %10.4f  \"%s\"\n",
                       e.rank + 1, e.token_id, e.logit_value,
                       e.token_string.c_str());
            }
            printf("\n");

            // Write JSON
            write_logits_json(logits_output_path, prompt,
                              token_ids, token_strings, entries);

        } else {
            // ── Normal generation mode ───────────────────────────
            printf("Prompt: \"%s\"\n", prompt.c_str());
            printf("Generating %d tokens...\n\n", max_tokens);

            cuinfer::CudaTimer timer;
            timer.start();

            std::string output = model.generate(prompt, max_tokens);

            timer.stop();

            printf("─── Output ───\n%s\n──────────────\n\n", output.c_str());

            float elapsed = timer.elapsed_ms();
            int total_tokens = model.current_position();
            float tok_per_sec = (total_tokens > 0)
                ? (total_tokens / (elapsed / 1000.0f))
                : 0.0f;

            printf("Generated %d tokens in %.1f ms (%.1f tok/s)\n",
                   total_tokens, elapsed, tok_per_sec);
            cuinfer::print_vram_usage("after generation");
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
