#include "sbm/machine.hpp"
#include "sbm/detail/random.hpp"
#include "sbm/math.hpp"
#include "sbm/detail/vector_ops.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace sbm {
bool SparseBranchMachine::push_candidate(std::vector<CandidateNode>& values,NodeId id,float edge_prior) const noexcept {
    if(!contains(id)) return false;
    const auto it=std::find_if(values.begin(),values.end(),[&](const CandidateNode& c){return c.id==id;});
    if(it!=values.end()){
        if(edge_prior>it->edge_prior) it->edge_prior=edge_prior;
        return false;
    }
    values.push_back({id,edge_prior});
    return true;
}
std::vector<SparseBranchMachine::CandidateNode> SparseBranchMachine::candidate_ids(std::uint64_t sig) {
    std::vector<CandidateNode> out;
    out.reserve(config_.bucket_scan_limit + config_.beam_width * config_.edge_scan_limit);
    const auto exact = static_cast<std::size_t>(bucket(sig));

    // Exact address-region residents must never be displaced by incoming edges.
    auto append_bucket = [&](const std::vector<NodeId>& source, std::size_t limit, bool randomize) {
        if (source.empty()) return;
        const std::size_t start = randomize
            ? static_cast<std::size_t>(detail::next_random(rng_state_) % source.size()) : 0U;
        const std::size_t attempts = std::min<std::size_t>(source.size(), limit * 4U + 1U);
        for (std::size_t k = 0; k < attempts && out.size() < limit; ++k) {
            const NodeId id = source[(start + k) % source.size()];
            if (!contains(id)) { ++stale_bucket_refs_skipped_; continue; }
            (void)push_candidate(out, id, 0.0F);
        }
    };
    append_bucket(hot_buckets_[exact], config_.bucket_scan_limit, false);
    append_bucket(buckets_[exact], config_.bucket_scan_limit, true);

    // Learned control-flow edges supplement, rather than replace, content addressing.
    const std::size_t hard_limit = static_cast<std::size_t>(config_.bucket_scan_limit) +
        static_cast<std::size_t>(config_.beam_width) * config_.edge_scan_limit;
    for (const NodeId src : previous_route_) {
        const auto slot = slot_of(src);
        if (slot == SIZE_MAX) continue;
        const auto limit = std::min<std::size_t>(edges_[slot].size(), config_.edge_scan_limit);
        for (std::size_t i = 0; i < limit && out.size() < hard_limit; ++i) {
            if (!contains(edges_[slot][i].dst)) { ++stale_edge_refs_skipped_; continue; }
            const float prior=edges_[slot][i].weight/(1.0F+std::max(0.0F,edges_[slot][i].weight));
            (void)push_candidate(out, edges_[slot][i].dst, prior);
        }
    }

    // Nearby regions are fallback exploration only.
    const auto base = static_cast<std::int64_t>(exact);
    const auto max_bucket = static_cast<std::int64_t>(buckets_.size());
    for (std::int64_t radius = 1; out.size() < hard_limit && radius <= 3; ++radius) {
        for (const auto probe : {base - radius, base + radius}) {
            if (probe < 0 || probe >= max_bucket) continue;
            append_bucket(hot_buckets_[static_cast<std::size_t>(probe)], hard_limit, false);
            append_bucket(buckets_[static_cast<std::size_t>(probe)], hard_limit, true);
            if (out.size() >= hard_limit) break;
        }
    }
    return out;
}
NodePhase SparseBranchMachine::phase_of_slot(std::size_t s) const noexcept{return(NodePhase)phases_[s];}
NodePhase SparseBranchMachine::phase(NodeId id) const noexcept{auto s=slot_of(id);return s==SIZE_MAX?NodePhase::Dormant:phase_of_slot(s);}
double SparseBranchMachine::score(std::size_t s, std::uint64_t sig,float edge_prior) const noexcept {
    const double sim = hamming_similarity(prototypes_[s], sig);
    const double rel = 1.0 / (1.0 + loss_ema_[s]);
    const double nov = 1.0 / std::sqrt(static_cast<double>(visits_[s]) + 1.0);
    const double exact_region = bucket(prototypes_[s]) == bucket(sig) ? 0.42 : 0.0;
    const double phase_bias = phase_of_slot(s) == NodePhase::Dormant ? -0.18 : 0.0;
    return exact_region + 0.72 * sim + 0.10 * rel + 0.02 * nov +
           static_cast<double>(config_.edge_score_weight*edge_prior) + phase_bias;
}
std::pair<std::vector<SparseBranchMachine::ScoredNode>,std::uint32_t> SparseBranchMachine::select_route(std::uint64_t sig){auto c=candidate_ids(sig);std::vector<ScoredNode>s;s.reserve(c.size());for(const auto& candidate:c){auto sl=slot_of(candidate.id);if(sl!=SIZE_MAX)s.push_back({score(sl,sig,candidate.edge_prior),candidate.id,0.0F,0.0F,bucket(prototypes_[sl])==bucket(sig)});}auto keep=std::min<std::size_t>(s.size(),config_.beam_width);std::partial_sort(s.begin(),s.begin()+keep,s.end(),[](auto&a,auto&b){return a.score==b.score?a.id>b.id:a.score>b.score;});s.resize(keep);assign_responsibilities(s);return{std::move(s),(std::uint32_t)c.size()};}
void SparseBranchMachine::assign_responsibilities(std::vector<ScoredNode>& active) const {
    if(active.empty()) return;
    const bool has_exact=std::any_of(active.begin(),active.end(),[](const ScoredNode& node){return node.exact_region;});
    const auto normalize_group=[&](bool exact,float mass){
        double max_score=-std::numeric_limits<double>::infinity();
        for(const auto& node:active)if(node.exact_region==exact)max_score=std::max(max_score,node.score);
        if(!std::isfinite(max_score))return;
        double sum=0.0;
        for(auto& node:active){
            if(node.exact_region!=exact)continue;
            node.responsibility=static_cast<float>(std::exp(static_cast<double>(config_.responsibility_temperature)*(node.score-max_score)));
            sum+=node.responsibility;
        }
        if(sum<=0.0)return;
        for(auto& node:active)if(node.exact_region==exact)node.responsibility*=mass/static_cast<float>(sum);
    };
    if(has_exact){
        normalize_group(true,std::clamp(config_.exact_region_mass,0.0F,1.0F));
        normalize_group(false,1.0F-std::clamp(config_.exact_region_mass,0.0F,1.0F));
    }else{
        normalize_group(false,1.0F);
    }
}
void SparseBranchMachine::aggregate(const std::vector<ScoredNode>& active,std::span<float> out) const {
    std::fill(out.begin(),out.end(),0.0F);
    for(const auto& node:active){
        const auto slot=slot_of(node.id);
        if(slot==SIZE_MAX) continue;
        detail::axpy(out.data(),output_vectors_.data()+slot*config_.vector_dim,node.responsibility,config_.vector_dim);
    }
}
void SparseBranchMachine::compute_counterfactual_contributions(std::vector<ScoredNode>& active,
                                                               std::span<const float> target,
                                                               float target_energy,
                                                               float full_loss) const {
    std::vector<float> without(config_.vector_dim,0.0F);
    for(auto& node:active){
        const auto slot=slot_of(node.id);
        const float remaining=1.0F-node.responsibility;
        if(slot==SIZE_MAX||remaining<1e-5F){node.contribution=0.0F;continue;}
        const float* local=output_vectors_.data()+slot*config_.vector_dim;
        for(std::uint32_t j=0;j<config_.vector_dim;++j){
            without[j]=(prediction_buffer_[j]-node.responsibility*local[j])/remaining;
        }
        const float mse_without=detail::squared_distance(without.data(),target.data(),config_.vector_dim)/
                                static_cast<float>(config_.vector_dim);
        const float loss_without=mse_without/std::max(target_energy,1e-5F);
        node.contribution=loss_without-full_loss;
    }
}

} // namespace sbm
