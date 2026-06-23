#include "sbm/experiment.hpp"

#include "sbm/machine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace sbm {
namespace {

struct TokenAccumulator {
    double cross_entropy{};
    double target_probability{};
    std::uint64_t top1{};
    std::uint64_t top5{};
    std::uint64_t examples{};
};

void add_distribution(TokenAccumulator& accumulator,
                      std::span<const float> probabilities,
                      std::uint32_t target) {
    const float target_probability = std::max(probabilities[target], 1e-12F);
    accumulator.cross_entropy -= std::log(target_probability);
    accumulator.target_probability += target_probability;
    const auto maximum = static_cast<std::uint32_t>(std::distance(
        probabilities.begin(),
        std::max_element(probabilities.begin(), probabilities.end())));
    accumulator.top1 += maximum == target ? 1U : 0U;

    std::size_t better = 0U;
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
        if (index != target && probabilities[index] > target_probability) ++better;
    }
    accumulator.top5 += better < std::min<std::size_t>(5U, probabilities.size()) ? 1U : 0U;
    ++accumulator.examples;
}

TokenMetrics finish(const TokenAccumulator& accumulator) {
    TokenMetrics result;
    if (accumulator.examples == 0U) return result;
    const double inverse = 1.0 / static_cast<double>(accumulator.examples);
    result.cross_entropy = accumulator.cross_entropy * inverse;
    result.bits_per_token = result.cross_entropy / std::log(2.0);
    result.perplexity = std::exp(std::min(result.cross_entropy, 80.0));
    result.top1_accuracy = static_cast<double>(accumulator.top1) * inverse;
    result.top5_accuracy = static_cast<double>(accumulator.top5) * inverse;
    result.mean_target_probability = accumulator.target_probability * inverse;
    return result;
}

struct CountRow {
    std::uint64_t total{};
    std::vector<std::uint32_t> counts;
};

class ConditionalTable {
public:
    explicit ConditionalTable(std::uint32_t vocabulary) : vocabulary_(vocabulary) {}

    void observe(std::uint64_t key, std::uint32_t target) {
        auto [iterator, inserted] = rows_.try_emplace(key);
        if (inserted) iterator->second.counts.assign(vocabulary_, 0U);
        ++iterator->second.total;
        ++iterator->second.counts[target];
    }

    [[nodiscard]] bool distribution(std::uint64_t key,
                                    std::span<const float> fallback,
                                    std::span<float> output,
                                    float smoothing = 0.5F) const {
        const auto iterator = rows_.find(key);
        if (iterator == rows_.end()) {
            std::copy(fallback.begin(), fallback.end(), output.begin());
            return false;
        }
        const auto& row = iterator->second;
        const double denominator = static_cast<double>(row.total) + smoothing;
        for (std::uint32_t token = 0; token < vocabulary_; ++token) {
            output[token] = static_cast<float>(
                (static_cast<double>(row.counts[token]) +
                 smoothing * static_cast<double>(fallback[token])) /
                denominator);
        }
        return true;
    }

private:
    std::uint32_t vocabulary_{};
    std::unordered_map<std::uint64_t, CountRow> rows_;
};

std::uint64_t pair_key(std::uint32_t previous, std::uint32_t current) {
    return (static_cast<std::uint64_t>(previous) << 32U) | current;
}

std::uint32_t token_at_lag(const TokenDataset& dataset,
                           std::size_t sequence_start,
                           std::size_t position,
                           std::size_t lag) {
    return position >= sequence_start + lag
        ? dataset.tokens[position - lag]
        : dataset.tokens[sequence_start];
}

void normalize_logits(std::span<float> values) {
    const float maximum = *std::max_element(values.begin(), values.end());
    double sum = 0.0;
    for (auto& value : values) {
        value = std::exp(value - maximum);
        sum += value;
    }
    const float inverse = static_cast<float>(1.0 / std::max(sum, 1e-30));
    for (auto& value : values) value *= inverse;
}

} // namespace

TokenExperimentResult run_token_experiment(const TokenDataset& dataset,
                                           std::size_t warmup_examples,
                                           Config config,
                                           bool strict_freeze,
                                           std::size_t prefill,
                                           std::size_t prune_interval,
                                           std::size_t merge_interval) {
    if (dataset.vocab_size < 2U || dataset.sequence_offsets.size() < 2U) {
        throw std::invalid_argument("invalid token dataset");
    }
    const std::size_t total_examples = dataset.example_count();
    if (total_examples == 0U || warmup_examples == 0U || warmup_examples >= total_examples) {
        throw std::invalid_argument("warmup must split non-empty token train/eval sets");
    }

    config.objective = ObjectiveKind::TokenCrossEntropy;
    config.token_alphabet = dataset.vocab_size;
    config.vector_dim = dataset.vocab_size;
    config.allow_growth_when_frozen = !strict_freeze;
    SparseBranchMachine model(config);
    if (prefill != 0U) model.prefill_distractors(prefill);

    const auto vocabulary = dataset.vocab_size;
    std::vector<std::uint64_t> unigram_counts(vocabulary, 0U);
    std::uint64_t unigram_total = 0U;
    ConditionalTable current_table(vocabulary);
    ConditionalTable pair_table(vocabulary);
    ConditionalTable lag2_table(vocabulary);
    ConditionalTable lag4_table(vocabulary);

    std::size_t training_index = 0U;
    for (std::size_t sequence = 0; sequence < dataset.sequence_count(); ++sequence) {
        const auto start = static_cast<std::size_t>(dataset.sequence_offsets[sequence]);
        const auto end = static_cast<std::size_t>(dataset.sequence_offsets[sequence + 1U]);
        for (std::size_t position = start; position + 1U < end; ++position) {
            if (training_index >= warmup_examples) break;
            const auto current = dataset.tokens[position];
            const auto target = dataset.tokens[position + 1U];
            const auto lag1 = token_at_lag(dataset, start, position, 1U);
            const auto lag2 = token_at_lag(dataset, start, position, 2U);
            const auto lag4 = token_at_lag(dataset, start, position, 4U);
            ++unigram_counts[target];
            ++unigram_total;
            current_table.observe(current, target);
            pair_table.observe(pair_key(lag1, current), target);
            lag2_table.observe(pair_key(lag2, current), target);
            lag4_table.observe(pair_key(lag4, current), target);
            ++training_index;
        }
        if (training_index >= warmup_examples) break;
    }

    std::vector<float> unigram(vocabulary, 0.0F);
    constexpr double prior = 0.5;
    const double unigram_denominator = static_cast<double>(unigram_total) +
        prior * static_cast<double>(vocabulary);
    for (std::uint32_t token = 0; token < vocabulary; ++token) {
        unigram[token] = static_cast<float>(
            (static_cast<double>(unigram_counts[token]) + prior) /
            unigram_denominator);
    }

    TokenAccumulator train_accumulator;
    TokenAccumulator eval_accumulator;
    TokenAccumulator unigram_accumulator;
    TokenAccumulator current_accumulator;
    TokenAccumulator pair_accumulator;
    TokenAccumulator multiscale_accumulator;
    double oracle_total = 0.0;
    std::uint64_t oracle_examples = 0U;

    std::vector<float> current_probability(vocabulary, 0.0F);
    std::vector<float> pair_probability(vocabulary, 0.0F);
    std::vector<float> lag2_probability(vocabulary, 0.0F);
    std::vector<float> lag4_probability(vocabulary, 0.0F);
    std::vector<float> multiscale_probability(vocabulary, 0.0F);

    std::size_t example_index = 0U;
    const auto start_time = std::chrono::steady_clock::now();
    for (std::size_t sequence = 0; sequence < dataset.sequence_count(); ++sequence) {
        const auto start = static_cast<std::size_t>(dataset.sequence_offsets[sequence]);
        const auto end = static_cast<std::size_t>(dataset.sequence_offsets[sequence + 1U]);
        model.reset_sequence();
        for (std::size_t position = start; position + 1U < end; ++position) {
            const bool learn = example_index < warmup_examples;
            const auto current = dataset.tokens[position];
            const auto target = dataset.tokens[position + 1U];
            const auto stats = model.step_token(current, target, learn);
            (void)stats;
            const auto prediction = model.last_prediction();
            add_distribution(learn ? train_accumulator : eval_accumulator,
                             prediction, target);

            if (!learn) {
                add_distribution(unigram_accumulator, unigram, target);
                (void)current_table.distribution(current, unigram, current_probability);
                add_distribution(current_accumulator, current_probability, target);

                const auto lag1 = token_at_lag(dataset, start, position, 1U);
                const auto lag2 = token_at_lag(dataset, start, position, 2U);
                const auto lag4 = token_at_lag(dataset, start, position, 4U);
                (void)pair_table.distribution(pair_key(lag1, current),
                                        current_probability, pair_probability);
                (void)lag2_table.distribution(pair_key(lag2, current),
                                        current_probability, lag2_probability);
                (void)lag4_table.distribution(pair_key(lag4, current),
                                        current_probability, lag4_probability);
                add_distribution(pair_accumulator, pair_probability, target);

                for (std::uint32_t candidate = 0; candidate < vocabulary; ++candidate) {
                    multiscale_probability[candidate] =
                        (std::log(std::max(pair_probability[candidate], 1e-12F)) +
                         std::log(std::max(lag2_probability[candidate], 1e-12F)) +
                         std::log(std::max(lag4_probability[candidate], 1e-12F))) /
                        3.0F;
                }
                normalize_logits(multiscale_probability);
                add_distribution(multiscale_accumulator, multiscale_probability, target);

                if (!dataset.oracle_nll.empty()) {
                    oracle_total += dataset.oracle_nll[position + 1U];
                    ++oracle_examples;
                }
            }

            if (learn && prune_interval != 0U &&
                (example_index + 1U) % prune_interval == 0U) {
                (void)model.prune();
            }
            if (learn && merge_interval != 0U &&
                (example_index + 1U) % merge_interval == 0U) {
                (void)model.merge_redundant();
            }
            ++example_index;
        }
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();

    TokenExperimentResult result;
    result.diagnostics = model.diagnostics();
    result.train = finish(train_accumulator);
    result.eval = finish(eval_accumulator);
    result.unigram_baseline_eval = finish(unigram_accumulator);
    result.current_token_baseline_eval = finish(current_accumulator);
    result.pair_context_baseline_eval = finish(pair_accumulator);
    result.multiscale_baseline_eval = finish(multiscale_accumulator);
    result.oracle_cross_entropy = oracle_examples == 0U
        ? std::numeric_limits<double>::quiet_NaN()
        : oracle_total / static_cast<double>(oracle_examples);
    result.excess_cross_entropy = oracle_examples == 0U
        ? std::numeric_limits<double>::quiet_NaN()
        : result.eval.cross_entropy - result.oracle_cross_entropy;
    result.steps_per_second = static_cast<double>(total_examples) / elapsed;
    result.elapsed_seconds = elapsed;
    result.strict_freeze = strict_freeze;
    result.dataset_hash = hash_dataset(dataset);
    result.vocab_size = dataset.vocab_size;
    result.train_examples = warmup_examples;
    result.eval_examples = total_examples - warmup_examples;
    result.sequence_count = dataset.sequence_count();
    result.address_lags = config.address_lags;
    result.exact_region_mass = config.exact_region_mass;
    result.edge_score_weight = config.edge_score_weight;
    result.softmax_temperature = config.softmax_temperature;
    result.label_smoothing = config.label_smoothing;
    return result;
}

std::string to_json(const TokenExperimentResult& result) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(8);
    const auto emit_metrics = [&](const char* prefix, const TokenMetrics& metrics,
                                  bool trailing = true) {
        out << "  \"" << prefix << "_cross_entropy\": " << metrics.cross_entropy << ",\n"
            << "  \"" << prefix << "_bits_per_token\": " << metrics.bits_per_token << ",\n"
            << "  \"" << prefix << "_perplexity\": " << metrics.perplexity << ",\n"
            << "  \"" << prefix << "_top1_accuracy\": " << metrics.top1_accuracy << ",\n"
            << "  \"" << prefix << "_top5_accuracy\": " << metrics.top5_accuracy << ",\n"
            << "  \"" << prefix << "_mean_target_probability\": "
            << metrics.mean_target_probability;
        if (trailing) out << ',';
        out << '\n';
    };
    out << "{\n"
        << "  \"task\": \"mathematical_next_token_cross_entropy\",\n"
        << "  \"objective\": \"token_cross_entropy\",\n"
        << "  \"vocab_size\": " << result.vocab_size << ",\n"
        << "  \"sequence_count\": " << result.sequence_count << ",\n"
        << "  \"train_examples\": " << result.train_examples << ",\n"
        << "  \"eval_examples\": " << result.eval_examples << ",\n"
        << "  \"address_lags\": [" << result.address_lags[0] << ", "
        << result.address_lags[1] << ", " << result.address_lags[2] << "],\n"
        << "  \"exact_region_mass\": " << result.exact_region_mass << ",\n"
        << "  \"edge_score_weight\": " << result.edge_score_weight << ",\n"
        << "  \"softmax_temperature\": " << result.softmax_temperature << ",\n"
        << "  \"label_smoothing\": " << result.label_smoothing << ",\n"
        << "  \"live_nodes\": " << result.diagnostics.live_nodes << ",\n"
        << "  \"edges\": " << result.diagnostics.edges << ",\n"
        << "  \"avg_active\": " << result.diagnostics.avg_active << ",\n"
        << "  \"avg_candidates\": " << result.diagnostics.avg_candidates << ",\n"
        << "  \"created_total\": " << result.diagnostics.created_total << ",\n"
        << "  \"estimated_bytes\": " << result.diagnostics.estimated_bytes << ",\n"
        << "  \"simd_enabled\": "
        << (result.diagnostics.simd_enabled ? "true" : "false") << ",\n";
    emit_metrics("train", result.train);
    emit_metrics("eval", result.eval);
    emit_metrics("unigram_baseline_eval", result.unigram_baseline_eval);
    emit_metrics("current_token_baseline_eval", result.current_token_baseline_eval);
    emit_metrics("pair_context_baseline_eval", result.pair_context_baseline_eval);
    emit_metrics("multiscale_baseline_eval", result.multiscale_baseline_eval);
    if (std::isfinite(result.oracle_cross_entropy)) {
        out << "  \"oracle_cross_entropy\": " << result.oracle_cross_entropy << ",\n"
            << "  \"excess_cross_entropy\": " << result.excess_cross_entropy << ",\n";
    } else {
        out << "  \"oracle_cross_entropy\": null,\n"
            << "  \"excess_cross_entropy\": null,\n";
    }
    out << "  \"steps_per_second\": " << result.steps_per_second << ",\n"
        << "  \"elapsed_seconds\": " << result.elapsed_seconds << ",\n"
        << "  \"strict_freeze\": " << (result.strict_freeze ? "true" : "false") << ",\n"
        << "  \"dataset_hash\": " << result.dataset_hash << "\n"
        << "}\n";
    return out.str();
}

} // namespace sbm
