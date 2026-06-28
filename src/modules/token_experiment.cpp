#include "sbm/experiment.hpp"

#include "sbm/machine.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
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

struct ChannelAttributionAccumulator {
    double credit_sum{};
    std::uint64_t observations{};
    std::uint64_t positive{};
    std::uint64_t documents{};
    std::uint64_t positive_documents{};
};

struct ProgramAttributionAccumulator {
    std::uint8_t channel{};
    std::uint32_t dependency{};
    double credit_sum{};
    double description_cost{};
    double execution_cost{};
    std::uint64_t observations{};
    std::uint64_t positive{};
};

void add_step(TokenAccumulator& accumulator, const StepStats& stats) {
    accumulator.cross_entropy += static_cast<double>(stats.cross_entropy);
    accumulator.target_probability += static_cast<double>(stats.target_probability);
    if (stats.ranking_available) {
        accumulator.top1 += stats.top1_correct ? 1U : 0U;
        accumulator.top5 += stats.top5_correct ? 1U : 0U;
    } else {
        accumulator.ranking_available = false;
    }
    ++accumulator.examples;
}

void add_target_probability(TokenAccumulator& accumulator, double probability) {
    const double bounded = std::max(probability, 1e-12);
    accumulator.cross_entropy -= std::log(bounded);
    accumulator.target_probability += bounded;
    ++accumulator.examples;
    accumulator.ranking_available = false;
}

void add_cross_entropy(TokenAccumulator& accumulator, double cross_entropy) {
    accumulator.cross_entropy += cross_entropy;
    accumulator.target_probability += std::exp(-std::min(cross_entropy, 80.0));
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

std::vector<ChannelAttribution> finish_channel_attribution(
    std::span<const ChannelAttributionAccumulator> accumulators,
    std::span<const AddressProgram> programs,
    std::span<const std::uint8_t> phases) {
    std::vector<ChannelAttribution> result;
    const auto count = std::min({accumulators.size(), programs.size(), phases.size()});
    result.reserve(count);
    for (std::size_t channel = 0U; channel < count; ++channel) {
        const auto& accumulator = accumulators[channel];
        ChannelAttribution attribution;
        attribution.channel = static_cast<std::uint8_t>(channel);
        attribution.program = programs[channel];
        attribution.phase = phases[channel];
        attribution.eval_observations = accumulator.observations;
        attribution.eval_positive = accumulator.positive;
        attribution.eval_documents = accumulator.documents;
        attribution.eval_positive_documents = accumulator.positive_documents;
        attribution.eval_credit_sum = accumulator.credit_sum;
        if (accumulator.observations != 0U) {
            attribution.eval_mean_credit = accumulator.credit_sum /
                static_cast<double>(accumulator.observations);
            attribution.eval_positive_fraction =
                static_cast<double>(accumulator.positive) /
                static_cast<double>(accumulator.observations);
        }
        if (accumulator.documents != 0U) {
            attribution.eval_positive_document_fraction =
                static_cast<double>(accumulator.positive_documents) /
                static_cast<double>(accumulator.documents);
        }
        result.push_back(attribution);
    }
    return result;
}

[[nodiscard]] std::uint64_t program_attribution_key(std::uint8_t channel,
                                                    std::uint32_t dependency) noexcept {
    return (static_cast<std::uint64_t>(channel) << 32U) |
           static_cast<std::uint64_t>(dependency);
}

std::vector<ProgramAttribution> finish_program_attribution(
    const std::unordered_map<std::uint64_t, ProgramAttributionAccumulator>& accumulators,
    std::span<const AddressProgram> programs) {
    std::vector<ProgramAttribution> result;
    result.reserve(accumulators.size());
    for (const auto& [key, accumulator] : accumulators) {
        (void)key;
        if (accumulator.channel >= programs.size()) continue;
        ProgramAttribution attribution;
        attribution.channel = accumulator.channel;
        attribution.program = programs[accumulator.channel];
        attribution.dependency = accumulator.dependency;
        attribution.credit_sum = accumulator.credit_sum;
        attribution.description_cost = accumulator.description_cost;
        attribution.execution_cost = accumulator.execution_cost;
        attribution.observations = accumulator.observations;
        attribution.positive = accumulator.positive;
        result.push_back(attribution);
    }
    std::sort(result.begin(), result.end(),
              [](const ProgramAttribution& left,
                 const ProgramAttribution& right) {
                  if (left.channel != right.channel) {
                      return left.channel < right.channel;
                  }
                  return left.dependency < right.dependency;
              });
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

void emit_progress(std::size_t processed,
                   std::size_t total,
                   std::size_t warmup,
                   const TokenAccumulator& train,
                   const TokenAccumulator& eval,
                   const SparseBranchMachine& model,
                   std::chrono::steady_clock::time_point started) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::max(
        std::chrono::duration<double>(now - started).count(), 1e-9);
    const double speed = static_cast<double>(processed) / elapsed;
    const double eta = speed <= 0.0
        ? std::numeric_limits<double>::infinity()
        : static_cast<double>(total - processed) / speed;
    const auto diagnostics = model.diagnostics();
    const auto train_nll = train.examples == 0U
        ? std::numeric_limits<double>::quiet_NaN()
        : train.cross_entropy / static_cast<double>(train.examples);
    const auto eval_nll = eval.examples == 0U
        ? std::numeric_limits<double>::quiet_NaN()
        : eval.cross_entropy / static_cast<double>(eval.examples);
    std::cerr << '\r' << std::fixed << std::setprecision(2)
              << "[sbm-progress] phase="
              << (processed < warmup ? "train" : "eval")
              << " examples=" << processed << '/' << total
              << " speed=" << speed << "/s"
              << " eta=" << eta << "s"
              << " train_nll=" << train_nll
              << " eval_nll=" << eval_nll
              << " live_nodes=" << diagnostics.live_nodes
              << " state_mb="
              << static_cast<double>(diagnostics.estimated_bytes) /
                     (1024.0 * 1024.0)
              << "        ";
    if (processed == total) {
        std::cerr << '\n';
    }
    std::cerr.flush();
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
    TokenAccumulator seed_only_accumulator;
    TokenAccumulator active_channels_only_accumulator;
    TokenAccumulator content_channels_only_accumulator;
    TokenAccumulator tuple_channels_only_accumulator;
    std::array<ChannelAttributionAccumulator, kMaxAddressChannels>
        eval_channel_attribution{};
    std::unordered_map<std::uint64_t, ProgramAttributionAccumulator>
        eval_program_attribution{};
    std::array<double, kMaxAddressChannels> eval_channel_responsibility_sum{};
    double oracle_total = 0.0;
    std::uint64_t oracle_examples = 0U;

    std::size_t example_index = 0U;
    double model_elapsed = 0.0;
    const bool progress_enabled = total_examples >= 1000000U;
    const auto progress_started = std::chrono::steady_clock::now();
    auto next_progress = progress_started;
    if (progress_enabled) {
        std::cerr << '\r' << "[sbm-progress] start examples=" << total_examples
                  << " train=" << warmup_examples
                  << " eval=" << (total_examples - warmup_examples)
                  << " vocab=" << vocabulary << "        ";
        std::cerr.flush();
        next_progress += std::chrono::seconds(120);
    }
    for (const auto& dataset : datasets) {
      for (std::size_t sequence = 0; sequence < dataset.sequence_count(); ++sequence) {
        const auto start = static_cast<std::size_t>(dataset.sequence_offsets[sequence]);
        const auto end = static_cast<std::size_t>(dataset.sequence_offsets[sequence + 1U]);
        model.reset_sequence();
        std::array<double, kMaxAddressChannels> sequence_channel_credit{};
        std::size_t sequence_channel_credit_count = 0U;
        bool sequence_has_eval_attribution = false;
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
                if (stats.channel_credit_count != 0U) {
                    const auto count = std::min<std::size_t>(
                        stats.channel_credit_count, eval_channel_attribution.size());
                    for (std::size_t channel = 0U; channel < count; ++channel) {
                        auto& accumulator = eval_channel_attribution[channel];
                        const double credit = stats.channel_credit[channel];
                        accumulator.credit_sum += credit;
                        ++accumulator.observations;
                        if (credit > 0.0) ++accumulator.positive;
                        const auto dependency = stats.channel_dependency[channel];
                        const auto key = program_attribution_key(
                            static_cast<std::uint8_t>(channel), dependency);
                        auto [program_iterator, inserted] =
                            eval_program_attribution.try_emplace(key);
                        auto& program_accumulator = program_iterator->second;
                        if (inserted) {
                            program_accumulator.channel =
                                static_cast<std::uint8_t>(channel);
                            program_accumulator.dependency = dependency;
                        }
                        program_accumulator.credit_sum += credit;
                        program_accumulator.description_cost +=
                            static_cast<double>(stats.channel_description_cost[channel]);
                        program_accumulator.execution_cost +=
                            static_cast<double>(stats.channel_execution_cost[channel]);
                        ++program_accumulator.observations;
                        if (credit > 0.0) ++program_accumulator.positive;
                        sequence_channel_credit[channel] += credit;
                        eval_channel_responsibility_sum[channel] +=
                            stats.channel_responsibility_mass[channel];
                    }
                    sequence_channel_credit_count =
                        std::max(sequence_channel_credit_count, count);
                    sequence_has_eval_attribution = true;
                }
                if (stats.channel_subset_available) {
                    add_cross_entropy(seed_only_accumulator,
                                      stats.seed_only_cross_entropy);
                    add_cross_entropy(active_channels_only_accumulator,
                                      stats.active_only_cross_entropy);
                    add_cross_entropy(content_channels_only_accumulator,
                                      stats.content_only_cross_entropy);
                    add_cross_entropy(tuple_channels_only_accumulator,
                                      stats.tuple_only_cross_entropy);
                }
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
            if (progress_enabled) {
                const auto now = std::chrono::steady_clock::now();
                if (now >= next_progress || example_index == warmup_examples ||
                    example_index == total_examples) {
                    emit_progress(example_index, total_examples, warmup_examples,
                                  train_accumulator, eval_accumulator, model,
                                  progress_started);
                    next_progress = now + std::chrono::seconds(120);
                }
            }
        }
        if (sequence_has_eval_attribution) {
            for (std::size_t channel = 0U; channel < sequence_channel_credit_count;
                 ++channel) {
                auto& accumulator = eval_channel_attribution[channel];
                ++accumulator.documents;
                if (sequence_channel_credit[channel] > 0.0) {
                    ++accumulator.positive_documents;
                }
            }
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
    result.seed_only_eval = finish(seed_only_accumulator);
    result.active_channels_only_eval = finish(active_channels_only_accumulator);
    result.content_channels_only_eval = finish(content_channels_only_accumulator);
    result.tuple_channels_only_eval = finish(tuple_channels_only_accumulator);
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
    if (config.record_channel_attribution) {
        result.eval_channel_attribution = finish_channel_attribution(
            eval_channel_attribution, result.learned_address_programs,
            result.learned_channel_phase);
        result.eval_program_attribution = finish_program_attribution(
            eval_program_attribution, result.learned_address_programs);
        result.eval_channel_mean_responsibility.reserve(
            result.eval_channel_attribution.size());
        for (std::size_t channel = 0U;
             channel < result.eval_channel_attribution.size(); ++channel) {
            const auto observations =
                result.eval_channel_attribution[channel].eval_observations;
            result.eval_channel_mean_responsibility.push_back(
                observations == 0U ? 0.0 :
                    eval_channel_responsibility_sum[channel] /
                    static_cast<double>(observations));
        }
    }
    result.topology_events = model.topology_events();
    result.exact_region_mass = config.exact_region_mass;
    result.edge_score_weight = config.edge_score_weight;
    result.softmax_temperature = config.softmax_temperature;
    result.label_smoothing = config.label_smoothing;
    result.sparse_token_output = config.sparse_token_output;
    result.output_tree_seed = config.output_tree_seed;
    result.structural_description_cost_weight =
        config.structural_description_cost_weight;
    result.structural_execution_cost_weight =
        config.structural_execution_cost_weight;
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

namespace {

TokenDataset copy_limited_examples(
    std::span<const MappedTokenShard* const> shards,
    std::size_t max_examples) {
    if (shards.empty()) {
        throw std::invalid_argument("limited token corpus requires shards");
    }
    TokenDataset result;
    result.sequence_offsets.push_back(0U);
    std::size_t copied_examples = 0U;
    for (const auto* shard : shards) {
        if (shard == nullptr) throw std::invalid_argument("null token shard");
        if (result.vocab_size == 0U) result.vocab_size = shard->vocab_size();
        if (result.vocab_size != shard->vocab_size()) {
            throw std::invalid_argument("limited corpus vocabularies differ");
        }
        const auto tokens = shard->tokens();
        const auto offsets = shard->sequence_offsets();
        for (std::size_t sequence = 0U;
             sequence + 1U < offsets.size() && copied_examples < max_examples;
             ++sequence) {
            const auto start = static_cast<std::size_t>(offsets[sequence]);
            const auto end = static_cast<std::size_t>(offsets[sequence + 1U]);
            if (end <= start + 1U) continue;
            const auto available_examples = end - start - 1U;
            const auto take_examples =
                std::min<std::size_t>(available_examples,
                                      max_examples - copied_examples);
            const auto take_tokens = take_examples + 1U;
            result.tokens.insert(result.tokens.end(),
                                 tokens.begin() + static_cast<std::ptrdiff_t>(start),
                                 tokens.begin() +
                                     static_cast<std::ptrdiff_t>(start + take_tokens));
            result.sequence_offsets.push_back(result.tokens.size());
            copied_examples += take_examples;
        }
        if (copied_examples >= max_examples) break;
    }
    if (result.vocab_size == 0U) {
        throw std::invalid_argument("limited corpus has no vocabulary");
    }
    return result;
}

} // namespace

TokenExperimentResult run_token_corpus_experiment_limited(
    std::span<const MappedTokenShard* const> train_shards,
    std::span<const MappedTokenShard* const> eval_shards,
    std::size_t max_train_examples,
    std::size_t max_eval_examples,
    Config config,
    bool strict_freeze,
    std::size_t prefill,
    std::size_t prune_interval,
    std::size_t merge_interval) {
    const auto train = copy_limited_examples(train_shards, max_train_examples);
    const auto eval = copy_limited_examples(eval_shards, max_eval_examples);
    const TokenDataView views[] = {
        {train.vocab_size, train.tokens, train.sequence_offsets, {},
         hash_dataset(train), "real_corpus_next_token_cross_entropy"},
        {eval.vocab_size, eval.tokens, eval.sequence_offsets, {},
         hash_dataset(eval), "real_corpus_next_token_cross_entropy"},
    };
    return run_token_views(views, train.example_count(), std::move(config),
                           strict_freeze, prefill, prune_interval, merge_interval,
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
    out << "],\n  \"eval_channel_attribution\": [";
    for (std::size_t i = 0; i < result.eval_channel_attribution.size(); ++i) {
        if (i != 0U) out << ", ";
        const auto& attribution = result.eval_channel_attribution[i];
        out << "{\"channel\":" << static_cast<unsigned>(attribution.channel)
            << ",\"lags\":[";
        for (std::size_t j = 0; j < attribution.program.arity; ++j) {
            if (j != 0U) out << ',';
            out << attribution.program.lags[j];
        }
        out << "],\"op\":" << static_cast<unsigned>(attribution.program.op)
            << ",\"phase\":" << static_cast<unsigned>(attribution.phase)
            << ",\"eval_observations\":" << attribution.eval_observations
            << ",\"eval_positive\":" << attribution.eval_positive
            << ",\"eval_documents\":" << attribution.eval_documents
            << ",\"eval_positive_documents\":"
            << attribution.eval_positive_documents
            << ",\"eval_credit_sum\":" << attribution.eval_credit_sum
            << ",\"eval_mean_credit\":" << attribution.eval_mean_credit
            << ",\"eval_positive_fraction\":"
            << attribution.eval_positive_fraction
            << ",\"eval_positive_document_fraction\":"
            << attribution.eval_positive_document_fraction << "}";
    }
    out << "],\n  \"eval_channel_mean_responsibility\": [";
    for (std::size_t i = 0; i < result.eval_channel_mean_responsibility.size(); ++i) {
        if (i != 0U) out << ", ";
        out << result.eval_channel_mean_responsibility[i];
    }
    out << "],\n  \"eval_program_attribution\": [";
    for (std::size_t i = 0; i < result.eval_program_attribution.size(); ++i) {
        if (i != 0U) out << ", ";
        const auto& attribution = result.eval_program_attribution[i];
        const double mean_credit = attribution.observations == 0U ? 0.0 :
            attribution.credit_sum / static_cast<double>(attribution.observations);
        const double positive_fraction = attribution.observations == 0U ? 0.0 :
            static_cast<double>(attribution.positive) /
                static_cast<double>(attribution.observations);
        const double structural_value = attribution.credit_sum -
            static_cast<double>(result.structural_description_cost_weight) *
                attribution.description_cost -
            static_cast<double>(result.structural_execution_cost_weight) *
                attribution.execution_cost;
        out << "{\"channel\":" << static_cast<unsigned>(attribution.channel)
            << ",\"lags\":[";
        for (std::size_t j = 0; j < attribution.program.arity; ++j) {
            if (j != 0U) out << ',';
            out << attribution.program.lags[j];
        }
        out << "],\"op\":" << static_cast<unsigned>(attribution.program.op)
            << ",\"dependency\":" << attribution.dependency
            << ",\"observations\":" << attribution.observations
            << ",\"positive\":" << attribution.positive
            << ",\"credit_sum\":" << attribution.credit_sum
            << ",\"mean_credit\":" << mean_credit
            << ",\"positive_fraction\":" << positive_fraction
            << ",\"description_cost\":" << attribution.description_cost
            << ",\"execution_cost\":" << attribution.execution_cost
            << ",\"structural_value\":" << structural_value << "}";
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
        << "  \"structural_description_cost_weight\": "
        << result.structural_description_cost_weight << ",\n"
        << "  \"structural_execution_cost_weight\": "
        << result.structural_execution_cost_weight << ",\n"
        << "  \"live_nodes\": " << result.diagnostics.live_nodes << ",\n"
        << "  \"edges\": " << result.diagnostics.edges << ",\n"
        << "  \"avg_active\": " << result.diagnostics.avg_active << ",\n"
        << "  \"avg_candidates\": " << result.diagnostics.avg_candidates << ",\n"
        << "  \"created_total\": " << result.diagnostics.created_total << ",\n"
        << "  \"estimated_bytes\": " << result.diagnostics.estimated_bytes << ",\n"
        << "  \"address_index_bytes\": " << result.diagnostics.address_index_bytes << ",\n"
        << "  \"address_occupied_buckets\": " << result.diagnostics.address_occupied_buckets << ",\n"
        << "  \"address_full_buckets\": " << result.diagnostics.address_full_buckets << ",\n"
        << "  \"address_max_bucket_residents\": " << result.diagnostics.address_max_bucket_residents << ",\n"
        << "  \"address_capacity_blocked_splits\": " << result.diagnostics.address_capacity_blocked_splits << ",\n"
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
        << "  \"sparse_output_admission_rejections\": "
        << result.diagnostics.sparse_output_admission_rejections << ",\n"
        << "  \"sparse_output_admission_promotions\": "
        << result.diagnostics.sparse_output_admission_promotions << ",\n"
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
        << "  \"address_execution_frames\": "
        << result.diagnostics.address_execution_frames << ",\n"
        << "  \"address_binding_hits\": "
        << result.diagnostics.address_binding_hits << ",\n"
        << "  \"address_binding_misses\": "
        << result.diagnostics.address_binding_misses << ",\n"
        << "  \"quarantined_channels\": "
        << result.diagnostics.quarantined_channels << ",\n"
        << "  \"recoverable_retired_channels\": "
        << result.diagnostics.recoverable_retired_channels << ",\n"
        << "  \"structural_value_nats\": "
        << result.diagnostics.structural_value_nats << ",\n"
        << "  \"structural_description_cost\": "
        << result.diagnostics.structural_description_cost << ",\n"
        << "  \"structural_execution_cost\": "
        << result.diagnostics.structural_execution_cost << ",\n"
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
    emit_metrics("seed_only_eval", result.seed_only_eval);
    emit_metrics("active_channels_only_eval", result.active_channels_only_eval);
    emit_metrics("content_channels_only_eval", result.content_channels_only_eval);
    emit_metrics("tuple_channels_only_eval", result.tuple_channels_only_eval);
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
