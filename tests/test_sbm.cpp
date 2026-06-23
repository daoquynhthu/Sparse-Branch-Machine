#include "sparse_branch_machine.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>

namespace {
bool close(float a, float b, float tolerance = 1e-6F) {
    return std::abs(a - b) <= tolerance;
}
}

int main() {
    const std::uint32_t fixed_window[]{1, 2, 3, 4};
    assert(sbm::rolling_signature(fixed_window) == sbm::rolling_signature(fixed_window));
    assert(sbm::hamming_similarity(42, 42) == 1.0);

    // Compact token packing must use both symbols for a 64-token alphabet.
    const std::array<std::uint32_t, 2> context_a{1, 7};
    const std::array<std::uint32_t, 2> context_b{33, 7};
    const auto signature_a = sbm::token_context_signature(context_a, 64);
    const auto signature_b = sbm::token_context_signature(context_b, 64);
    assert((signature_a >> 52U) != (signature_b >> 52U));

    const std::array<std::uint32_t, 5> lag_context{3, 9, 17, 25, 31};
    const auto lag_one = sbm::lagged_token_signature(lag_context, 64, 1);
    const auto lag_two = sbm::lagged_token_signature(lag_context, 64, 2);
    assert((lag_one >> 52U) != (lag_two >> 52U));

    const std::array<std::uint32_t, 2> program_lags{1U, 4U};
    const std::array<std::uint32_t, 5> program_context_a{3, 9, 17, 25, 31};
    const std::array<std::uint32_t, 5> program_context_b{7, 9, 17, 25, 31};
    const auto program_signature_a = sbm::address_program_signature(
        program_context_a, 64, program_lags, sbm::AddressOp::Tuple);
    const auto program_signature_b = sbm::address_program_signature(
        program_context_b, 64, program_lags, sbm::AddressOp::Tuple);
    assert(program_signature_a != program_signature_b);

    auto dataset = sbm::generate_vector_process(8000, 32, 16, 20, 9);
    assert(dataset.tokens.size() == 8000);
    assert(dataset.targets.size() == 8000 * 16);

    const auto expected_hash = sbm::hash_dataset(dataset);
    const char* path = "sbm_vector_test.bin";
    sbm::save_dataset(dataset, path);
    auto loaded = sbm::load_dataset(path);
    std::remove(path);
    assert(sbm::hash_dataset(loaded) == expected_hash);

    // A target may update the model only after the prediction has been fixed.
    sbm::Config leakage_config;
    leakage_config.token_alphabet = 8;
    leakage_config.vector_dim = 8;
    leakage_config.bucket_bits = 6;
    leakage_config.seed = 11;
    sbm::SparseBranchMachine positive_model(leakage_config);
    sbm::SparseBranchMachine negative_model(leakage_config);
    const std::vector<float> positive_target(8, 1.0F);
    const std::vector<float> negative_target(8, -1.0F);
    (void)positive_model.step(3, positive_target, true);
    (void)negative_model.step(3, negative_target, true);
    const auto positive_prediction = positive_model.last_prediction();
    const auto negative_prediction = negative_model.last_prediction();
    for (std::size_t i = 0; i < positive_prediction.size(); ++i) {
        assert(close(positive_prediction[i], 0.0F));
        assert(close(positive_prediction[i], negative_prediction[i]));
    }

    sbm::Config config;
    config.token_alphabet = 32;
    config.vector_dim = 16;
    config.bucket_bits = 10;
    config.seed = 9;
    sbm::SparseBranchMachine machine(config);
    for (std::size_t i = 0; i < 3000; ++i) {
        const auto stats = machine.step(dataset.tokens[i], dataset.target(i), true);
        assert(std::isfinite(stats.normalized_mse));
        assert(stats.active_nodes <= config.beam_width);
    }
    const auto frozen_nodes = machine.live_nodes();
    for (std::size_t i = 3000; i < 4000; ++i) {
        (void)machine.step(dataset.tokens[i], dataset.target(i), false);
    }
    assert(machine.live_nodes() == frozen_nodes);
    const auto diagnostics = machine.diagnostics();
    assert(diagnostics.anchor_nodes > 0);
    assert(diagnostics.residual_nodes > 0);
    assert(diagnostics.anchor_nodes + diagnostics.residual_nodes == diagnostics.live_nodes);

    const auto result = sbm::run_experiment(dataset, 3000, 9, true, 1000, 0, 0);
    assert(std::isfinite(result.eval.normalized_rmse));
    assert(std::isfinite(result.token_baseline_eval.r2));
    assert(std::isfinite(result.pair_baseline_eval.r2));
    assert(std::isfinite(result.pair_lag2_baseline_eval.r2));
    assert(std::isfinite(result.multiscale_baseline_eval.r2));
    assert(std::abs(result.mean_baseline_eval.r2) < 1e-5);
    assert(result.token_baseline_eval.r2 < 0.85);
    assert(result.pair_baseline_eval.r2 < 0.90);
    assert(result.seen_context_vectors + result.unseen_context_vectors == 5000);
    assert(result.address_lags == sbm::Config{}.address_lags);
    assert(result.diagnostics.steps == 8000);


    auto token_dataset = sbm::generate_math_token_process(8, 512, 16, 13, 0.8F, 0.2F);
    assert(token_dataset.tokens.size() == 4096);
    assert(token_dataset.sequence_count() == 8);
    assert(token_dataset.example_count() == 4088);
    const auto token_hash = sbm::hash_dataset(token_dataset);
    const char* token_path = "sbm_token_test.bin";
    sbm::save_token_dataset(token_dataset, token_path);
    auto loaded_tokens = sbm::load_token_dataset(token_path);
    std::remove(token_path);
    assert(sbm::hash_dataset(loaded_tokens) == token_hash);

    // Real-corpus shards are deterministic, memory mapped and preserve
    // sequence boundaries when producing next-token examples.
    const std::array<std::uint32_t, 5> shard_tokens{10U, 11U, 12U, 20U, 21U};
    const std::array<std::uint64_t, 3> shard_offsets{0U, 3U, 5U};
    const auto shard_dataset = sbm::make_token_dataset(shard_tokens, 32U, shard_offsets);
    const char* shard_path_a = "sbm_token_shard_a.sbt";
    const char* shard_path_b = "sbm_token_shard_b.sbt";
    std::remove(shard_path_a);
    std::remove(shard_path_b);
    sbm::write_token_shard(shard_dataset, shard_path_a);
    sbm::write_token_shard(shard_dataset, shard_path_b);
    {
        std::ifstream first(shard_path_a, std::ios::binary);
        std::ifstream second(shard_path_b, std::ios::binary);
        const std::vector<char> first_bytes{
            std::istreambuf_iterator<char>(first), std::istreambuf_iterator<char>()};
        const std::vector<char> second_bytes{
            std::istreambuf_iterator<char>(second), std::istreambuf_iterator<char>()};
        assert(first_bytes == second_bytes);
    }
    {
        sbm::MappedTokenShard shard(shard_path_a, 7U);
        assert(shard.vocab_size() == 32U);
        assert(shard.token_count() == shard_tokens.size());
        assert(shard.sequence_count() == 2U);
        assert(shard.dataset_hash() == sbm::hash_dataset(shard_dataset));

        sbm::Config shard_config;
        shard_config.objective = sbm::ObjectiveKind::TokenCrossEntropy;
        shard_config.token_alphabet = 32U;
        shard_config.vector_dim = 32U;
        shard_config.bucket_bits = 6U;
        const auto shard_result = sbm::run_token_experiment(
            shard, 2U, shard_config, true, 0U, 0U, 0U);
        assert(shard_result.task == "real_corpus_next_token_cross_entropy");
        assert(shard_result.train_examples == 2U);
        assert(shard_result.eval_examples == 1U);

        sbm::TokenShardCursor cursor{7U, 0U, 0U};
        sbm::TokenExample example{};
        assert(shard.next(cursor, example));
        assert(example.input == 10U && example.target == 11U);
        const auto resume_cursor = cursor;
        assert(shard.next(cursor, example));
        assert(example.input == 11U && example.target == 12U);
        assert(shard.next(cursor, example));
        assert(example.input == 20U && example.target == 21U);
        assert(!shard.next(cursor, example));
        sbm::TokenShardCursor malformed_cursor{
            7U, 0U, std::numeric_limits<std::uint64_t>::max()};
        assert(shard.next(malformed_cursor, example));
        assert(example.input == 20U && example.target == 21U);

        sbm::MappedTokenShard reopened(shard_path_a, 7U);
        auto replay_cursor = resume_cursor;
        assert(reopened.next(replay_cursor, example));
        assert(example.input == 11U && example.target == 12U);
        const std::array<const sbm::MappedTokenShard*, 1> train_shards{&shard};
        const std::array<const sbm::MappedTokenShard*, 1> eval_shards{&reopened};
        const auto corpus_result = sbm::run_token_corpus_experiment(
            train_shards, eval_shards, shard_config, true, 0U, 0U, 0U);
        assert(corpus_result.train_examples == 3U);
        assert(corpus_result.eval_examples == 3U);
        assert(corpus_result.sequence_count == 4U);
        for (const auto& event : corpus_result.topology_events) {
            assert(event.step <= corpus_result.train_examples);
        }
    }
    std::remove(shard_path_a);
    std::remove(shard_path_b);

    sbm::Config token_config;
    token_config.objective = sbm::ObjectiveKind::TokenCrossEntropy;
    token_config.token_alphabet = 16;
    token_config.vector_dim = 16;
    token_config.bucket_bits = 8;
    token_config.seed = 13;
    sbm::SparseBranchMachine token_machine(token_config);
    token_machine.reset_sequence();
    const auto first_token_stats = token_machine.step_token(
        token_dataset.tokens[0], token_dataset.tokens[1], true);
    assert(std::isfinite(first_token_stats.cross_entropy));
    assert(first_token_stats.target_probability > 0.0F);
    assert(first_token_stats.predicted_token < token_config.vector_dim);
    assert(first_token_stats.top5_correct || token_config.vector_dim > 5U);
    assert(token_machine.last_prediction().empty());

    const auto token_result = sbm::run_token_experiment(
        token_dataset, 2500, token_config, true, 0, 0, 0);
    assert(std::isfinite(token_result.eval.cross_entropy));
    assert(std::isfinite(token_result.eval.perplexity));
    assert(token_result.eval_examples == token_dataset.example_count() - 2500);
    assert(token_result.oracle_cross_entropy > 0.0);


    // Strict evaluation freeze must reject an unfinished probe at the train
    // boundary and must never mutate topology using evaluation examples.
    sbm::Config freeze_config = token_config;
    freeze_config.adaptive_topology = true;
    freeze_config.max_address_channels = 3U;
    freeze_config.beam_width = 3U;
    freeze_config.topology_probe_interval = 2480U;
    freeze_config.topology_probe_steps = 128U;
    freeze_config.topology_validation_steps = 32U;
    freeze_config.topology_min_observations = 16U;
    const auto freeze_result = sbm::run_token_experiment(
        token_dataset, 2500, freeze_config, true, 0, 0, 0);
    for (const auto& event : freeze_result.topology_events) {
        assert(event.step <= 2500U);
    }

    // Adaptive topology must propose, validate and safely reject an unhelpful
    // channel without disturbing the seed channel.
    sbm::Config topology_config = token_config;
    topology_config.adaptive_topology = true;
    topology_config.address_lags = {1U};
    topology_config.max_address_channels = 3U;
    topology_config.beam_width = 3U;
    topology_config.topology_probe_interval = 8U;
    topology_config.topology_probe_steps = 48U;
    topology_config.topology_probe_warmup = 8U;
    topology_config.topology_validation_steps = 16U;
    topology_config.topology_min_observations = 8U;
    topology_config.topology_accept_credit = 10.0F;
    topology_config.topology_prune_patience = 10000U;
    sbm::SparseBranchMachine topology_machine(topology_config);
    for (std::size_t i = 0; i < 256; ++i) {
        const auto position = i % (token_dataset.tokens.size() - 1U);
        (void)topology_machine.step_token(token_dataset.tokens[position],
                                          token_dataset.tokens[position + 1U], true);
    }
    const auto topology_diag = topology_machine.diagnostics();
    assert(topology_diag.topology_proposals > 0U);
    assert(topology_diag.topology_rejected > 0U);
    const auto learned_lags = topology_machine.learned_address_lags();
    assert(!learned_lags.empty() && learned_lags.front() == 1U);

    const auto learned_programs = topology_machine.learned_address_programs();
    assert(!learned_programs.empty());
    for (const auto& program : learned_programs) {
        assert(program.arity >= 1U &&
               program.arity <= topology_config.topology_max_arity);
    }

    sbm::Config fixed_topology_config = token_config;
    fixed_topology_config.adaptive_topology = false;
    fixed_topology_config.address_lags = {1U, 2U, 4U};
    sbm::SparseBranchMachine fixed_topology_machine(fixed_topology_config);
    for (std::size_t i = 0; i < 128; ++i) {
        (void)fixed_topology_machine.step_token(token_dataset.tokens[i],
                                                token_dataset.tokens[i + 1U], true);
    }
    assert(fixed_topology_machine.diagnostics().topology_proposals == 0U);
    assert(fixed_topology_machine.learned_address_lags() ==
           fixed_topology_config.address_lags);

    std::cout << "all tests passed; SIMD=" << sbm::simd_available() << '\n';
}
