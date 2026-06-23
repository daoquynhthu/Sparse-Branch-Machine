#include "sparse_branch_machine.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace sbm {
namespace {
constexpr std::uint32_t kDatasetVersion = 1;
constexpr std::array<char, 8> kMagic{'S','B','M','D','A','T','A','1'};

std::uint64_t next_random(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    return mix64(state);
}

template <typename T>
void write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("dataset write failed");
}

template <typename T>
T read_pod(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("dataset read failed");
    return value;
}
}

std::uint64_t mix64(std::uint64_t x) noexcept {
    x ^= x >> 30U;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27U;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31U;
    return x;
}

std::uint64_t rolling_signature(std::span<const std::uint32_t> window,
                                std::uint64_t seed) noexcept {
    auto h = seed;
    for (std::size_t i = 0; i < window.size(); ++i) {
        h ^= mix64(static_cast<std::uint64_t>(window[i]) +
                   0x10001ULL * static_cast<std::uint64_t>(i + 1U));
        h = std::rotl(h, 11);
        h = mix64(h);
    }
    return h;
}

double hamming_similarity(std::uint64_t a, std::uint64_t b) noexcept {
    return 1.0 - static_cast<double>(std::popcount(a ^ b)) / 64.0;
}

std::uint64_t hash_dataset(const Dataset& dataset) noexcept {
    std::uint64_t h = mix64(dataset.alphabet_size);
    for (std::size_t i = 0; i < dataset.sequence.size(); ++i) {
        h ^= mix64(static_cast<std::uint64_t>(dataset.sequence[i]) + i * 0x9E37ULL);
        h = std::rotl(h, 7);
    }
    return mix64(h ^ dataset.sequence.size());
}

SparseBranchMachine::SparseBranchMachine(Config config)
    : config_(config), rng_state_(config.seed),
      buckets_(std::size_t{1} << config.bucket_bits),
      hot_buckets_(std::size_t{1} << config.bucket_bits) {
    if (config_.alphabet_size == 0U) throw std::invalid_argument("alphabet_size must be positive");
    if (config_.bucket_bits == 0U || config_.bucket_bits > 20U)
        throw std::invalid_argument("bucket_bits must be in [1,20]");
    if (config_.beam_width == 0U || config_.bucket_scan_limit == 0U)
        throw std::invalid_argument("beam_width and bucket_scan_limit must be positive");
    if (config_.max_edges_per_node == 0U)
        throw std::invalid_argument("max_edges_per_node must be positive");
}

std::uint32_t SparseBranchMachine::bucket(std::uint64_t signature) const noexcept {
    return static_cast<std::uint32_t>(signature >> (64U - config_.bucket_bits));
}

std::size_t SparseBranchMachine::slot_of(NodeId id) const noexcept {
    if (id >= id_to_slot_.size()) return std::numeric_limits<std::size_t>::max();
    const auto slot = id_to_slot_[id];
    if (slot == UINT32_MAX || slot >= ids_.size() || ids_[slot] != id)
        return std::numeric_limits<std::size_t>::max();
    return slot;
}

bool SparseBranchMachine::contains(NodeId id) const noexcept {
    return slot_of(id) != std::numeric_limits<std::size_t>::max();
}

NodeId SparseBranchMachine::new_node(std::uint64_t signature) {
    if (next_id_ == kInvalidNode) throw std::overflow_error("logical node id space exhausted");
    const NodeId id = next_id_++;
    const auto slot = static_cast<std::uint32_t>(ids_.size());
    if (id_to_slot_.size() <= id) id_to_slot_.resize(static_cast<std::size_t>(id) + 1U, UINT32_MAX);
    id_to_slot_[id] = slot;
    ids_.push_back(id);
    prototypes_.push_back(signature);
    visits_.push_back(0);
    correct_.push_back(0);
    utility_ema_.push_back(0.0F);
    hot_indexed_.push_back(0U);
    output_counts_.insert(output_counts_.end(), config_.alphabet_size, 1.0F);
    edges_.emplace_back();
    edges_.back().reserve(config_.max_edges_per_node);
    buckets_[bucket(signature)].push_back(id);
    ++total_created_;
    return id;
}

bool SparseBranchMachine::push_unique(std::vector<NodeId>& values, NodeId id) const noexcept {
    if (!contains(id)) return false;
    if (std::find(values.begin(), values.end(), id) != values.end()) return false;
    values.push_back(id);
    return true;
}

std::vector<NodeId> SparseBranchMachine::candidate_ids(std::uint64_t signature) {
    const std::size_t hard_limit = static_cast<std::size_t>(config_.bucket_scan_limit) +
        static_cast<std::size_t>(config_.beam_width) * config_.edge_scan_limit;
    std::vector<NodeId> out;
    out.reserve(hard_limit);

    for (const NodeId src : previous_route_) {
        const auto slot = slot_of(src);
        if (slot == std::numeric_limits<std::size_t>::max()) continue;
        const auto& list = edges_[slot];
        const auto limit = std::min<std::size_t>(list.size(), config_.edge_scan_limit);
        for (std::size_t i = 0; i < limit; ++i) {
            if (!contains(list[i].dst)) { ++stale_edge_refs_skipped_; continue; }
            (void)push_unique(out, list[i].dst);
        }
    }

    const auto base = static_cast<std::int64_t>(bucket(signature));
    const auto max_bucket = static_cast<std::int64_t>(buckets_.size());

    // Proven nodes are searched before the cold address space. This keeps
    // retrieval quality stable when the model contains many unused nodes.
    for (std::int64_t radius = 0; out.size() < config_.bucket_scan_limit && radius <= 3; ++radius) {
        const std::array<std::int64_t, 2> probes{base - radius, base + radius};
        const int probe_count = radius == 0 ? 1 : 2;
        for (int p = 0; p < probe_count && out.size() < config_.bucket_scan_limit; ++p) {
            const auto pb = probes[static_cast<std::size_t>(p)];
            if (pb < 0 || pb >= max_bucket) continue;
            for (const NodeId id : hot_buckets_[static_cast<std::size_t>(pb)]) {
                if (!contains(id)) { ++stale_bucket_refs_skipped_; continue; }
                (void)push_unique(out, id);
                if (out.size() >= config_.bucket_scan_limit) break;
            }
        }
    }
    for (std::int64_t radius = 0; out.size() < config_.bucket_scan_limit && radius <= 3; ++radius) {
        const std::array<std::int64_t, 2> probes{base - radius, base + radius};
        const int probe_count = radius == 0 ? 1 : 2;
        for (int p = 0; p < probe_count && out.size() < config_.bucket_scan_limit; ++p) {
            const auto pb = probes[static_cast<std::size_t>(p)];
            if (pb < 0 || pb >= max_bucket) continue;
            const auto& source = buckets_[static_cast<std::size_t>(pb)];
            if (source.empty()) continue;
            const std::size_t attempts_limit = std::min<std::size_t>(
                source.size(), static_cast<std::size_t>(config_.bucket_scan_limit) * 4U);
            std::size_t start = static_cast<std::size_t>(next_random(rng_state_) % source.size());
            for (std::size_t k = 0; k < attempts_limit && out.size() < config_.bucket_scan_limit; ++k) {
                const NodeId id = source[(start + k) % source.size()];
                if (!contains(id)) { ++stale_bucket_refs_skipped_; continue; }
                (void)push_unique(out, id);
            }
        }
    }
    return out;
}

double SparseBranchMachine::score(std::size_t slot, std::uint64_t signature) const noexcept {
    const auto similarity = hamming_similarity(prototypes_[slot], signature);
    const auto reliability = (static_cast<double>(correct_[slot]) + 1.0) /
                             (static_cast<double>(visits_[slot]) + 2.0);
    const auto novelty = 1.0 / std::sqrt(static_cast<double>(visits_[slot]) + 1.0);
    return 0.72 * similarity + 0.23 * reliability + 0.05 * novelty;
}

std::pair<std::vector<SparseBranchMachine::ScoredNode>, std::uint32_t>
SparseBranchMachine::select_route(std::uint64_t signature) {
    auto candidates = candidate_ids(signature);
    std::vector<ScoredNode> scored;
    scored.reserve(candidates.size());
    for (const auto id : candidates) {
        const auto slot = slot_of(id);
        if (slot != std::numeric_limits<std::size_t>::max()) scored.push_back({score(slot, signature), id});
    }
    const auto keep = std::min<std::size_t>(scored.size(), config_.beam_width);
    std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(keep), scored.end(),
        [](const ScoredNode& a, const ScoredNode& b) {
            return a.score == b.score ? a.id > b.id : a.score > b.score;
        });
    scored.resize(keep);
    return {std::move(scored), static_cast<std::uint32_t>(candidates.size())};
}

std::uint32_t SparseBranchMachine::aggregate(const std::vector<ScoredNode>& active) const {
    if (active.empty()) return 0U;
    std::vector<double> vote(config_.alphabet_size, 0.0);
    for (const auto& selected : active) {
        const auto slot = slot_of(selected.id);
        if (slot == std::numeric_limits<std::size_t>::max()) continue;
        const auto base = slot * config_.alphabet_size;
        double total = 0.0;
        for (std::uint32_t s = 0; s < config_.alphabet_size; ++s) total += output_counts_[base + s];
        const double weight = std::max(selected.score, 1e-6);
        for (std::uint32_t s = 0; s < config_.alphabet_size; ++s)
            vote[s] += weight * output_counts_[base + s] / total;
    }
    return static_cast<std::uint32_t>(std::distance(vote.begin(), std::max_element(vote.begin(), vote.end())));
}

void SparseBranchMachine::reinforce_edge(NodeId source, NodeId destination, float delta) {
    const auto slot = slot_of(source);
    if (slot == std::numeric_limits<std::size_t>::max() || !contains(destination)) return;
    auto& list = edges_[slot];
    auto it = std::find_if(list.begin(), list.end(), [destination](const Edge& e){ return e.dst == destination; });
    if (it == list.end()) list.push_back({destination, delta});
    else it->weight = 0.995F * it->weight + delta;
    std::sort(list.begin(), list.end(), [](const Edge& a, const Edge& b){
        return a.weight == b.weight ? a.dst < b.dst : a.weight > b.weight;
    });
    if (list.size() > config_.max_edges_per_node) list.resize(config_.max_edges_per_node);
}

StepStats SparseBranchMachine::step(std::uint32_t token, std::uint32_t target, bool learn) {
    if (token >= config_.alphabet_size || target >= config_.alphabet_size)
        throw std::out_of_range("token and target must be inside alphabet");
    history_.push_back(token);
    if (history_.size() > config_.context_width) history_.pop_front();
    std::vector<std::uint32_t> window(history_.begin(), history_.end());
    const auto signature = rolling_signature(window);
    auto [active, examined] = select_route(signature);
    std::uint32_t created = 0;

    double best_similarity = 0.0;
    for (const auto& selected : active) {
        const auto slot = slot_of(selected.id);
        if (slot != std::numeric_limits<std::size_t>::max())
            best_similarity = std::max(best_similarity, hamming_similarity(prototypes_[slot], signature));
    }

    if ((learn || config_.allow_growth_when_frozen) &&
        (active.empty() || best_similarity < config_.create_similarity)) {
        const auto id = new_node(signature);
        if (active.size() >= config_.beam_width) active.resize(config_.beam_width - 1U);
        active.insert(active.begin(), ScoredNode{1.0, id});
        created = 1;
    }

    const auto prediction = aggregate(active);
    const bool is_correct = prediction == target;
    std::vector<NodeId> route;
    route.reserve(active.size());
    for (const auto& x : active) route.push_back(x.id);

    if (learn) {
        for (std::size_t rank = 0; rank < active.size(); ++rank) {
            const auto slot = slot_of(active[rank].id);
            if (slot == std::numeric_limits<std::size_t>::max()) continue;
            if (visits_[slot] == 0U && hot_indexed_[slot] == 0U) {
                hot_buckets_[bucket(prototypes_[slot])].push_back(ids_[slot]);
                hot_indexed_[slot] = 1U;
            }
            ++visits_[slot];
            correct_[slot] += static_cast<std::uint32_t>(is_correct);
            output_counts_[slot * config_.alphabet_size + target] += 1.0F / static_cast<float>(rank + 1U);
            utility_ema_[slot] = 0.98F * utility_ema_[slot] + 0.02F * (is_correct ? 1.0F : -1.0F);
        }
        for (const auto src : previous_route_)
            for (std::size_t rank = 0; rank < route.size(); ++rank)
                reinforce_edge(src, route[rank], (is_correct ? 1.0F : 0.2F) / static_cast<float>(rank + 1U));

        if (config_.create_on_error && !is_correct && best_similarity >= config_.create_similarity) {
            const auto id = new_node(signature);
            const auto slot = slot_of(id);
            output_counts_[slot * config_.alphabet_size + target] += 2.0F;
            ++created;
        }
        previous_route_ = route;
    }

    ++total_steps_;
    total_candidates_ += examined;
    total_active_ += active.size();
    return {prediction, is_correct, static_cast<std::uint32_t>(active.size()), examined,
            static_cast<std::uint32_t>(ids_.size()), created, std::move(route)};
}

void SparseBranchMachine::erase_slot(std::size_t slot) {
    const NodeId removed = ids_[slot];
    const std::size_t last = ids_.size() - 1U;
    if (slot != last) {
        ids_[slot] = ids_[last];
        prototypes_[slot] = prototypes_[last];
        visits_[slot] = visits_[last];
        correct_[slot] = correct_[last];
        utility_ema_[slot] = utility_ema_[last];
        hot_indexed_[slot] = hot_indexed_[last];
        edges_[slot] = std::move(edges_[last]);
        for (std::uint32_t s = 0; s < config_.alphabet_size; ++s)
            output_counts_[slot * config_.alphabet_size + s] = output_counts_[last * config_.alphabet_size + s];
        id_to_slot_[ids_[slot]] = static_cast<std::uint32_t>(slot);
    }
    ids_.pop_back(); prototypes_.pop_back(); visits_.pop_back(); correct_.pop_back();
    utility_ema_.pop_back(); hot_indexed_.pop_back(); edges_.pop_back();
    output_counts_.resize(ids_.size() * config_.alphabet_size);
    id_to_slot_[removed] = UINT32_MAX;
}

std::size_t SparseBranchMachine::prune(std::uint32_t min_visits, float utility_threshold) {
    std::size_t removed = 0;
    for (std::size_t slot = ids_.size(); slot-- > 0;) {
        if (visits_[slot] < min_visits && utility_ema_[slot] < utility_threshold) {
            erase_slot(slot);
            ++removed;
        }
    }
    total_pruned_ += removed;
    previous_route_.erase(std::remove_if(previous_route_.begin(), previous_route_.end(),
        [this](NodeId id){ return !contains(id); }), previous_route_.end());
    if (removed > 0) rebuild_indexes();
    return removed;
}

void SparseBranchMachine::rebuild_indexes() {
    for (auto& bucket_nodes : buckets_) bucket_nodes.clear();
    for (auto& bucket_nodes : hot_buckets_) bucket_nodes.clear();
    for (std::size_t slot = 0; slot < ids_.size(); ++slot) {
        buckets_[bucket(prototypes_[slot])].push_back(ids_[slot]);
        if (hot_indexed_[slot] != 0U) hot_buckets_[bucket(prototypes_[slot])].push_back(ids_[slot]);
    }
    for (auto& list : edges_) {
        list.erase(std::remove_if(list.begin(), list.end(), [this](const Edge& e){ return !contains(e.dst); }), list.end());
    }
}

void SparseBranchMachine::prefill_distractors(std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) (void)new_node(next_random(rng_state_));
}

Diagnostics SparseBranchMachine::diagnostics() const noexcept {
    std::uint64_t edge_count = 0;
    std::uint64_t edge_capacity = 0;
    for (const auto& list : edges_) { edge_count += list.size(); edge_capacity += list.capacity(); }
    std::uint64_t bucket_capacity = 0;
    for (const auto& b : buckets_) bucket_capacity += b.capacity();
    for (const auto& b : hot_buckets_) bucket_capacity += b.capacity();
    const auto denom = std::max<std::uint64_t>(1, total_steps_);
    std::uint64_t bytes = ids_.capacity()*sizeof(NodeId) + prototypes_.capacity()*sizeof(std::uint64_t) +
        visits_.capacity()*sizeof(std::uint32_t) + correct_.capacity()*sizeof(std::uint32_t) +
        utility_ema_.capacity()*sizeof(float) + hot_indexed_.capacity()*sizeof(std::uint8_t) + output_counts_.capacity()*sizeof(float) +
        id_to_slot_.capacity()*sizeof(std::uint32_t) + edge_capacity*sizeof(Edge) +
        bucket_capacity*sizeof(NodeId);
    return {total_steps_, ids_.size(), next_id_, edge_count,
            static_cast<double>(total_active_)/static_cast<double>(denom),
            static_cast<double>(total_candidates_)/static_cast<double>(denom),
            total_created_, total_pruned_, stale_bucket_refs_skipped_, stale_edge_refs_skipped_, bytes};
}

Dataset generate_hidden_fsm(std::size_t length, std::uint32_t alphabet,
                            std::uint32_t states, std::uint64_t seed) {
    Dataset dataset{alphabet, {}};
    if (length == 0 || alphabet == 0 || states == 0) return dataset;
    std::uint64_t state_rng = seed;
    std::vector<std::uint32_t> transition(static_cast<std::size_t>(states) * alphabet);
    for (auto& x : transition) x = static_cast<std::uint32_t>(next_random(state_rng) % states);
    std::vector<std::uint32_t> emit(states);
    for (auto& x : emit) x = static_cast<std::uint32_t>(next_random(state_rng) % alphabet);
    auto state = static_cast<std::uint32_t>(next_random(state_rng) % states);
    auto x = static_cast<std::uint32_t>(next_random(state_rng) % alphabet);
    dataset.sequence.reserve(length);
    dataset.sequence.push_back(x);
    for (std::size_t t = 0; t + 1U < length; ++t) {
        const auto branch = static_cast<std::uint32_t>((x ^ static_cast<std::uint32_t>(t * 3U) ^ (state >> 1U)) % alphabet);
        state = transition[static_cast<std::size_t>(state) * alphabet + branch];
        x = emit[state];
        dataset.sequence.push_back(x);
    }
    return dataset;
}

void save_dataset(const Dataset& dataset, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open dataset for writing: " + path);
    out.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write_pod(out, kDatasetVersion);
    write_pod(out, dataset.alphabet_size);
    const auto count = static_cast<std::uint64_t>(dataset.sequence.size());
    write_pod(out, count);
    const auto hash = hash_dataset(dataset);
    write_pod(out, hash);
    out.write(reinterpret_cast<const char*>(dataset.sequence.data()),
              static_cast<std::streamsize>(dataset.sequence.size() * sizeof(std::uint32_t)));
    if (!out) throw std::runtime_error("dataset payload write failed");
}

Dataset load_dataset(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open dataset: " + path);
    std::array<char,8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (magic != kMagic) throw std::runtime_error("invalid dataset magic");
    const auto version = read_pod<std::uint32_t>(in);
    if (version != kDatasetVersion) throw std::runtime_error("unsupported dataset version");
    Dataset d;
    d.alphabet_size = read_pod<std::uint32_t>(in);
    const auto count = read_pod<std::uint64_t>(in);
    const auto expected_hash = read_pod<std::uint64_t>(in);
    if (count > (1ULL << 34U)) throw std::runtime_error("dataset too large");
    d.sequence.resize(static_cast<std::size_t>(count));
    in.read(reinterpret_cast<char*>(d.sequence.data()), static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
    if (!in) throw std::runtime_error("truncated dataset payload");
    if (hash_dataset(d) != expected_hash) throw std::runtime_error("dataset checksum mismatch");
    return d;
}

ExperimentResult run_experiment(const Dataset& dataset, std::size_t warmup,
                                std::uint64_t seed, bool strict_freeze,
                                std::size_t prefill, std::size_t prune_interval) {
    if (dataset.sequence.size() < 2) throw std::invalid_argument("dataset needs at least two symbols");
    const std::size_t steps = dataset.sequence.size() - 1U;
    if (warmup > steps) throw std::invalid_argument("warmup exceeds dataset steps");
    Config cfg;
    cfg.alphabet_size = dataset.alphabet_size;
    cfg.seed = seed;
    cfg.allow_growth_when_frozen = !strict_freeze;
    SparseBranchMachine model(cfg);
    model.prefill_distractors(prefill);
    std::uint64_t train_correct = 0, eval_correct = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < steps; ++i) {
        const bool learn = i < warmup;
        const auto stats = model.step(dataset.sequence[i], dataset.sequence[i+1U], learn);
        if (learn) train_correct += stats.correct; else eval_correct += stats.correct;
        if (learn && prune_interval > 0 && (i + 1U) % prune_interval == 0) (void)model.prune();
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ExperimentResult r;
    r.diagnostics = model.diagnostics();
    r.train_accuracy = static_cast<double>(train_correct) / static_cast<double>(std::max<std::size_t>(1, warmup));
    r.eval_accuracy = static_cast<double>(eval_correct) / static_cast<double>(std::max<std::size_t>(1, steps - warmup));
    r.steps_per_second = static_cast<double>(steps) / elapsed;
    r.elapsed_seconds = elapsed;
    r.strict_freeze = strict_freeze;
    r.dataset_hash = hash_dataset(dataset);
    return r;
}

std::string to_json(const ExperimentResult& r) {
    std::ostringstream o; o.setf(std::ios::fixed); o.precision(8);
    o << "{\n"
      << "  \"steps\": " << r.diagnostics.steps << ",\n"
      << "  \"live_nodes\": " << r.diagnostics.live_nodes << ",\n"
      << "  \"logical_ids_issued\": " << r.diagnostics.logical_ids_issued << ",\n"
      << "  \"edges\": " << r.diagnostics.edges << ",\n"
      << "  \"avg_active\": " << r.diagnostics.avg_active << ",\n"
      << "  \"avg_candidates\": " << r.diagnostics.avg_candidates << ",\n"
      << "  \"created_total\": " << r.diagnostics.created_total << ",\n"
      << "  \"pruned_total\": " << r.diagnostics.pruned_total << ",\n"
      << "  \"stale_bucket_refs_skipped\": " << r.diagnostics.stale_bucket_refs_skipped << ",\n"
      << "  \"stale_edge_refs_skipped\": " << r.diagnostics.stale_edge_refs_skipped << ",\n"
      << "  \"estimated_bytes\": " << r.diagnostics.estimated_bytes << ",\n"
      << "  \"train_accuracy\": " << r.train_accuracy << ",\n"
      << "  \"eval_accuracy\": " << r.eval_accuracy << ",\n"
      << "  \"steps_per_second\": " << r.steps_per_second << ",\n"
      << "  \"elapsed_seconds\": " << r.elapsed_seconds << ",\n"
      << "  \"strict_freeze\": " << (r.strict_freeze ? "true" : "false") << ",\n"
      << "  \"dataset_hash\": " << r.dataset_hash << "\n}\n";
    return o.str();
}

} // namespace sbm
