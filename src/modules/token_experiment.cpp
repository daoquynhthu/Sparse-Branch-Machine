#include "sbm/experiment.hpp"

#include "sbm/machine.hpp"

#include <algorithm>
#include <bit>
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
    bool ranking_available{true};
};

void add_step(TokenAccumulator& accumulator, const StepStats& stats) {
    accumulator.cross_entropy += static_cast<double>(stats.cross_entropy);
    accumulator.target_probability += static_cast<double>(stats.target_probability);
    accumulator.top1 += stats.top1_correct ? 1U : 0U;
    accumulator.top5 += stats.top5_correct ? 1U : 0U;
    ++accumulator.examples;
}

void add_target_probability(TokenAccumulator& accumulator, double probability) {
    const double bounded = std::max(probability, 1e-12);
    accumulator.cross_entropy -= std::log(bounded);
    accumulator.target_probability += bounded;
    ++accumulator.examples;
    accumulator.ranking_available = false;
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
    result.ranking_available = accumulator.ranking_available;
    return result;
}

struct CountRow {
    std::uint64_t total{};
    std::unordered_map<std::uint32_t, std::uint32_t> counts;
};

class ConditionalTable {
public:
    void observe(std::uint64_t key, std::uint32_t target) {
        auto [iterator, inserted] = rows_.try_emplace(key);
        (void)inserted;
        ++iterator->second.total;
        ++iterator->second.counts[target];
    }

    [[nodiscard]] double target_probability(std::uint64_t key,
                                            std::uint32_t target,
                                            double fallback,
                                            double smoothing = 0.5) const {
        const auto iterator = rows_.find(key);
        if (iterator == rows_.end()) return fallback;
        const auto& row = iterator->second;
        const double denominator = static_cast<double>(row.total) + smoothing;
        const auto count = row.counts.find(target);
        const auto observed = count == row.counts.end() ? 0U : count->second;
        return (static_cast<double>(observed) + smoothing * fallback) / denominator;
    }

private:
    std::unordered_map<std::uint64_t, CountRow> rows_;
};

std::uint64_t pair_key(std::uint32_t previous, std::uint32_t current) {
    return (static_cast<std::uint64_t>(previous) << 32U) | current;
}

struct TokenDataView {
    std::uint32_t vocab_size{};
    std::span<const std::uint32_t> tokens;
    std::span<const std::uint64_t> sequence_offsets;
    std::span<const float> oracle_nll;
    std::uint64_t dataset_hash{};
    const char* task{};

    [[nodiscard]] std::size_t sequence_count() const noexcept {
        return sequence_offsets.empty() ? 0U : sequence_offsets.size() - 1U;
    }
    [[nodiscard]] std::size_t example_count() const noexcept {
        return tokens.size() - sequence_count();
    }
};

std::uint32_t token_at_lag(const TokenDataView& dataset,
                           std::size_t sequence_start,
                           std::size_t position,
                           std::size_t lag) {
    return position >= sequence_start + lag
        ? dataset.tokens[position - lag]
        : dataset.tokens[sequence_start];
}

} // namespace

static TokenExperimentResult run_token_views(std::span<const TokenDataView> datasets,
                                             std::size_t warmup_examples,
                                             Config config,
                                             bool strict_freeze,
                                             std::size_t prefill,
                                             std::size_t prune_interval,
                                             std::size_t merge_interval,
                                             const char* task) {
    if (datasets.empty() || datasets.front().vocab_size < 2U) {
        throw std::invalid_argument("invalid token dataset");
    }
    const auto vocabulary = datasets.front().vocab_size;
    std::size_t total_examples = 0U;
    std::size_t total_sequences = 0U;
    std::uint64_t combined_hash = datasets.size() == 1U
        ? datasets.front().dataset_hash : 0x53424D434F525055ULL;
    for (std::size_t index = 0U; index < datasets.size(); ++index) {
        const auto& dataset = datasets[index];
        if (dataset.vocab_size != vocabulary || dataset.sequence_offsets.size() < 2U) {
            throw std::invalid_argument("token shards have incompatible vocabularies");
        }
        total_examples += dataset.example_count();
        total_sequences += dataset.sequence_count();
        if (datasets.size() != 1U) {
            combined_hash ^= std::rotl(dataset.dataset_hash,
                static_cast<int>((index * 13U) & 63U));
        }
    }
    if (total_examples == 0U || warmup_examples == 0U || warmup_examples >= total_examples) {
        throw std::invalid_argument("warmup must split non-empty token train/eval sets");
    }

    config.objective = ObjectiveKind::TokenCrossEntropy;
    config.token_alphabet = vocabulary;
    config.vector_dim = vocabulary;
    config.allow_growth_when_frozen = !strict_freeze;
    SparseBranchMachine model(config);
    if (prefill != 0U) model.prefill_distractors(prefill);

    std::vector<std::uint64_t> unigram_counts(vocabulary, 0U);
    std::uint64_t unigram_total = 0U;
    ConditionalTable current_table;
    ConditionalTable pair_table;
    ConditionalTable lag2_table;
    ConditionalTable lag4_table;

    const auto baseline_training_start = std::chrono::steady_clock::now();
    std::size_t training_index = 0U;
    for (const auto& dataset : datasets) {
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
      if (training_index >= warmup_examples) break;
    }
    double baseline_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - baseline_training_start).count();

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
    TokenAccumulator interpolated_multiscale_accumulator;
    double oracle_total = 0.0;
    std::uint64_t oracle_examples = 0U;

    std::size_t example_index = 0U;
    double model_elapsed = 0.0;
    for (const auto& dataset : datasets) {
      for (std::size_t sequence = 0; sequence < dataset.sequence_count(); ++sequence) {
        const auto start = static_cast<std::size_t>(dataset.sequence_offsets[sequence]);
        const auto end = static_cast<std::size_t>(dataset.sequence_offsets[sequence + 1U]);
        model.reset_sequence();
        for (std::size_t position = start; position + 1U < end; ++position) {
            const bool learn = example_index < warmup_examples;
            if (strict_freeze && example_index == warmup_examples) {
                model.freeze_topology();
            }
            const auto current = dataset.tokens[position];
            const auto target = dataset.tokens[position + 1U];
            const auto model_start = std::chrono::steady_clock::now();
            const auto stats = model.step_token(current, target, learn);
            model_elapsed += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - model_start).count();
            add_step(learn ? train_accumulator : eval_accumulator, stats);

            if (!learn) {
                const auto baseline_start = std::chrono::steady_clock::now();
                const double unigram_probability = unigram[target];
                add_target_probability(unigram_accumulator, unigram_probability);
                const double current_probability = current_table.target_probability(
                    current, target, unigram_probability);
                add_target_probability(current_accumulator, current_probability);

                const auto lag1 = token_at_lag(dataset, start, position, 1U);
                const auto lag2 = token_at_lag(dataset, start, position, 2U);
                const auto lag4 = token_at_lag(dataset, start, position, 4U);
                const double pair_probability = pair_table.target_probability(
                    pair_key(lag1, current), target, current_probability);
                const double lag2_probability = lag2_table.target_probability(
                    pair_key(lag2, current), target, current_probability);
                const double lag4_probability = lag4_table.target_probability(
                    pair_key(lag4, current), target, current_probability);
                add_target_probability(pair_accumulator, pair_probability);
                add_target_probability(
                    interpolated_multiscale_accumulator,
                    (pair_probability + lag2_probability + lag4_probability) / 3.0);
                baseline_elapsed += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - baseline_start).count();

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
    }
    const double elapsed = std::max(model_elapsed, 1e-12);

    TokenExperimentResult result;
    result.task = task;
    result.diagnostics = model.diagnostics();
    result.train = finish(train_accumulator);
    result.eval = finish(eval_accumulator);
    result.unigram_baseline_eval = finish(unigram_accumulator);
    result.current_token_baseline_eval = finish(current_accumulator);
    result.pair_context_baseline_eval = finish(pair_accumulator);
    result.interpolated_multiscale_baseline_eval =
        finish(interpolated_multiscale_accumulator);
    result.multiscale_baseline_eval = result.interpolated_multiscale_baseline_eval;
    result.oracle_cross_entropy = oracle_examples == 0U
        ? std::numeric_limits<double>::quiet_NaN()
        : oracle_total / static_cast<double>(oracle_examples);
    result.excess_cross_entropy = oracle_examples == 0U
        ? std::numeric_limits<double>::quiet_NaN()
        : result.eval.cross_entropy - result.oracle_cross_entropy;
    result.steps_per_second = static_cast<double>(total_examples) / elapsed;
    result.elapsed_seconds = elapsed;
    result.baseline_elapsed_seconds = baseline_elapsed;
    result.strict_freeze = strict_freeze;
    result.dataset_hash = combined_hash;
    result.vocab_size = vocabulary;
    result.train_examples = warmup_examples;
    result.eval_examples = total_examples - warmup_examples;
    result.sequence_count = total_sequences;
    result.address_lags = config.address_lags;
    result.learned_address_lags = model.learned_address_lags();
    result.learned_address_programs = model.learned_address_programs();
    result.learned_channel_credit = model.learned_channel_credit();
    result.learned_channel_phase = model.learned_channel_phase();
    result.topology_events = model.topology_events();
    result.exact_region_mass = config.exact_region_mass;
    result.edge_score_weight = config.edge_score_weight;
    result.softmax_temperature = config.softmax_temperature;
    result.label_smoothing = config.label_smoothing;
    result.sparse_token_output = config.sparse_token_output;
    result.output_tree_seed = config.output_tree_seed;
    return result;
}

TokenExperimentResult run_token_experiment(const TokenDataset& dataset,
                                           std::size_t warmup_examples,
                                           Config config,
                                           bool strict_freeze,
                                           std::size_t prefill,
                                           std::size_t prune_interval,
                                           std::size_t merge_interval) {
    const TokenDataView view{
        dataset.vocab_size, dataset.tokens, dataset.sequence_offsets,
        dataset.oracle_nll, hash_dataset(dataset),
        "mathematical_next_token_cross_entropy"};
    return run_token_views(std::span<const TokenDataView>(&view, 1U), warmup_examples,
                           std::move(config), strict_freeze, prefill, prune_interval,
                           merge_interval, "mathematical_next_token_cross_entropy");
}

TokenExperimentResult run_token_experiment(const MappedTokenShard& shard,
                                           std::size_t warmup_examples,
                                           Config config,
                                           bool strict_freeze,
                                           std::size_t prefill,
                                           std::size_t prune_interval,
                                           std::size_t merge_interval) {
    const TokenDataView view{
        shard.vocab_size(), shard.tokens(), shard.sequence_offsets(), {},
        shard.dataset_hash(), "real_corpus_next_token_cross_entropy"};
    return run_token_views(std::span<const TokenDataView>(&view, 1U), warmup_examples,
                           std::move(config), strict_freeze, prefill, prune_interval,
                           merge_interval, "real_corpus_next_token_cross_entropy");
}

TokenExperimentResult run_token_corpus_experiment(
    std::span<const MappedTokenShard* const> train_shards,
    std::span<const MappedTokenShard* const> eval_shards,
    Config config,
    bool strict_freeze,
    std::size_t prefill,
    std::size_t prune_interval,
    std::size_t merge_interval) {
    if (train_shards.empty() || eval_shards.empty()) {
        throw std::invalid_argument("token corpus requires train and eval shards");
    }
    std::vector<TokenDataView> views;
    views.reserve(train_shards.size() + eval_shards.size());
    std::size_t warmup_examples = 0U;
    const auto append = [&](const MappedTokenShard* shard, bool training) {
        if (shard == nullptr) throw std::invalid_argument("null token shard");
        views.push_back(TokenDataView{
            shard->vocab_size(), shard->tokens(), shard->sequence_offsets(), {},
            shard->dataset_hash(), "real_corpus_next_token_cross_entropy"});
        if (training) warmup_examples += views.back().example_count();
    };
    for (const auto* shard : train_shards) append(shard, true);
    for (const auto* shard : eval_shards) append(shard, false);
    return run_token_views(views, warmup_examples, std::move(config), strict_freeze,
                           prefill, prune_interval, merge_interval,
                           "real_corpus_next_token_cross_entropy");
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
            << "  \"" << prefix << "_top1_accuracy\": ";
        if (metrics.ranking_available) out << metrics.top1_accuracy;
        else out << "null";
        out << ",\n  \"" << prefix << "_top5_accuracy\": ";
        if (metrics.ranking_available) out << metrics.top5_accuracy;
        else out << "null";
        out << ",\n"
            << "  \"" << prefix << "_mean_target_probability\": "
            << metrics.mean_target_probability;
        if (trailing) out << ',';
        out << '\n';
    };
    out << "{\n"
        << "  \"task\": \"" << result.task << "\",\n"
        << "  \"objective\": \"token_cross_entropy\",\n"
        << "  \"vocab_size\": " << result.vocab_size << ",\n"
        << "  \"sequence_count\": " << result.sequence_count << ",\n"
        << "  \"train_examples\": " << result.train_examples << ",\n"
        << "  \"eval_examples\": " << result.eval_examples << ",\n"
        << "  \"address_lags\": [";
    for (std::size_t i = 0; i < result.address_lags.size(); ++i) {
        if (i != 0U) out << ", ";
        out << result.address_lags[i];
    }
    out << "],\n  \"learned_address_lags\": [";
    for (std::size_t i = 0; i < result.learned_address_lags.size(); ++i) {
        if (i != 0U) out << ", ";
        out << result.learned_address_lags[i];
    }
    out << "],\n  \"learned_address_programs\": [";
    for (std::size_t i = 0; i < result.learned_address_programs.size(); ++i) {
        if (i != 0U) out << ", ";
        out << '[';
        const auto& program = result.learned_address_programs[i];
        for (std::size_t j = 0; j < program.arity; ++j) {
            if (j != 0U) out << ", ";
            out << program.lags[j];
        }
        out << ']';
    }
    out << "],\n  \"learned_address_operations\": [";
    for (std::size_t i = 0; i < result.learned_address_programs.size(); ++i) {
        if (i != 0U) out << ", ";
        out << static_cast<unsigned>(result.learned_address_programs[i].op);
    }
    out << "],\n  \"learned_channel_credit\": [";
    for (std::size_t i = 0; i < result.learned_channel_credit.size(); ++i) {
        if (i != 0U) out << ", ";
        out << result.learned_channel_credit[i];
    }
    out << "],\n  \"learned_channel_phase\": [";
    for (std::size_t i = 0; i < result.learned_channel_phase.size(); ++i) {
        if (i != 0U) out << ", ";
        out << static_cast<unsigned>(result.learned_channel_phase[i]);
    }
    out << "],\n  \"topology_events\": [";
    for (std::size_t i = 0; i < result.topology_events.size(); ++i) {
        if (i != 0U) out << ", ";
        const auto& event = result.topology_events[i];
        out << "{\"step\":" << event.step << ",\"lags\":[";
        for (std::size_t j = 0; j < event.program.arity; ++j) {
            if (j != 0U) out << ',';
            out << event.program.lags[j];
        }
        out << "],\"op\":" << static_cast<unsigned>(event.program.op)
            << ",\"decision\":" << static_cast<unsigned>(event.decision)
            << ",\"credit\":" << event.credit << "}";
    }
    out << "],\n"
        << "  \"exact_region_mass\": " << result.exact_region_mass << ",\n"
        << "  \"edge_score_weight\": " << result.edge_score_weight << ",\n"
        << "  \"softmax_temperature\": " << result.softmax_temperature << ",\n"
        << "  \"label_smoothing\": " << result.label_smoothing << ",\n"
        << "  \"sparse_token_output\": "
        << (result.sparse_token_output ? "true" : "false") << ",\n"
        << "  \"output_tree_seed\": " << result.output_tree_seed << ",\n"
        << "  \"live_nodes\": " << result.diagnostics.live_nodes << ",\n"
        << "  \"edges\": " << result.diagnostics.edges << ",\n"
        << "  \"avg_active\": " << result.diagnostics.avg_active << ",\n"
        << "  \"avg_candidates\": " << result.diagnostics.avg_candidates << ",\n"
        << "  \"created_total\": " << result.diagnostics.created_total << ",\n"
        << "  \"estimated_bytes\": " << result.diagnostics.estimated_bytes << ",\n"
        << "  \"address_index_bytes\": " << result.diagnostics.address_index_bytes << ",\n"
        << "  \"output_structure_bytes\": "
        << result.diagnostics.output_structure_bytes << ",\n"
        << "  \"max_bucket_candidates_inspected\": "
        << result.diagnostics.max_bucket_candidates_inspected << ",\n"
        << "  \"max_sparse_entries_per_node\": "
        << result.diagnostics.max_sparse_entries_per_node << ",\n"
        << "  \"global_output_prior_bytes\": "
        << result.diagnostics.global_output_prior_bytes << ",\n"
        << "  \"global_output_prior_updates\": "
        << result.diagnostics.global_output_prior_updates << ",\n"
        << "  \"sparse_output_insertions\": "
        << result.diagnostics.sparse_output_insertions << ",\n"
        << "  \"sparse_output_evictions\": "
        << result.diagnostics.sparse_output_evictions << ",\n"
        << "  \"sparse_output_probable_reconstructions\": "
        << result.diagnostics.sparse_output_probable_reconstructions << ",\n"
        << "  \"sparse_output_saturated_nodes\": "
        << result.diagnostics.sparse_output_saturated_nodes << ",\n"
        << "  \"max_sparse_decision_visits\": "
        << result.diagnostics.max_sparse_decision_visits << ",\n"
        << "  \"max_responsibility_mass_error\": "
        << result.diagnostics.max_responsibility_mass_error << ",\n"
        << "  \"sparse_output_entries\": "
        << result.diagnostics.sparse_output_entries << ",\n"
        << "  \"topology_proposals\": " << result.diagnostics.topology_proposals << ",\n"
        << "  \"topology_accepted\": " << result.diagnostics.topology_accepted << ",\n"
        << "  \"topology_rejected\": " << result.diagnostics.topology_rejected << ",\n"
        << "  \"topology_pruned\": " << result.diagnostics.topology_pruned << ",\n"
        << "  \"active_channels\": " << result.diagnostics.active_channels << ",\n"
        << "  \"probe_channels\": " << result.diagnostics.probe_channels << ",\n"
        << "  \"simd_enabled\": "
        << (result.diagnostics.simd_enabled ? "true" : "false") << ",\n";
    emit_metrics("train", result.train);
    emit_metrics("eval", result.eval);
    emit_metrics("unigram_baseline_eval", result.unigram_baseline_eval);
    emit_metrics("current_token_baseline_eval", result.current_token_baseline_eval);
    emit_metrics("pair_context_baseline_eval", result.pair_context_baseline_eval);
    emit_metrics("multiscale_baseline_eval", result.multiscale_baseline_eval);
    emit_metrics("interpolated_multiscale_baseline_eval",
                 result.interpolated_multiscale_baseline_eval);
    if (std::isfinite(result.oracle_cross_entropy)) {
        out << "  \"oracle_cross_entropy\": " << result.oracle_cross_entropy << ",\n"
            << "  \"excess_cross_entropy\": " << result.excess_cross_entropy << ",\n";
    } else {
        out << "  \"oracle_cross_entropy\": null,\n"
            << "  \"excess_cross_entropy\": null,\n";
    }
    out << "  \"steps_per_second\": " << result.steps_per_second << ",\n"
        << "  \"elapsed_seconds\": " << result.elapsed_seconds << ",\n"
        << "  \"baseline_elapsed_seconds\": "
        << result.baseline_elapsed_seconds << ",\n"
        << "  \"strict_freeze\": " << (result.strict_freeze ? "true" : "false") << ",\n"
        << "  \"dataset_hash\": " << result.dataset_hash << "\n"
        << "}\n";
    return out.str();
}

} // namespace sbm
