#include "sparse_branch_machine.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace sbm {
namespace {
constexpr std::uint32_t kDatasetVersion = 2;
constexpr std::array<char, 8> kMagic{'S','B','M','V','E','C','0','2'};

std::uint64_t next_random(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    return mix64(state);
}

template <typename T> void write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("dataset write failed");
}
template <typename T> T read_pod(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("dataset read failed");
    return value;
}

float dot_scalar(const float* a, const float* b, std::size_t n) noexcept {
    float s=0; for(std::size_t i=0;i<n;++i) s += a[i]*b[i]; return s;
}
float sqdist_scalar(const float* a, const float* b, std::size_t n) noexcept {
    float s=0; for(std::size_t i=0;i<n;++i){ const float d=a[i]-b[i]; s += d*d; } return s;
}
void axpy_scalar(float* dst, const float* src, float alpha, std::size_t n) noexcept {
    for(std::size_t i=0;i<n;++i) dst[i] += alpha*src[i];
}
void lerp_scalar(float* dst, const float* target, float lr, std::size_t n) noexcept {
    for(std::size_t i=0;i<n;++i) dst[i] += lr*(target[i]-dst[i]);
}

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float dot_avx2(const float* a, const float* b, std::size_t n) noexcept {
    __m256 sum=_mm256_setzero_ps(); std::size_t i=0;
    for(;i+8<=n;i+=8) sum=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),sum);
    alignas(32) float tmp[8]; _mm256_store_ps(tmp,sum); float s=0; for(float v:tmp)s+=v;
    for(;i<n;++i)s+=a[i]*b[i];
    return s;
}
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float sqdist_avx2(const float* a, const float* b, std::size_t n) noexcept {
    __m256 sum=_mm256_setzero_ps(); std::size_t i=0;
    for(;i+8<=n;i+=8){ auto d=_mm256_sub_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i)); sum=_mm256_fmadd_ps(d,d,sum); }
    alignas(32) float tmp[8]; _mm256_store_ps(tmp,sum); float s=0; for(float v:tmp)s+=v;
    for(;i<n;++i){float d=a[i]-b[i];s+=d*d;} return s;
}
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void axpy_avx2(float* dst,const float* src,float alpha,std::size_t n) noexcept {
    auto av=_mm256_set1_ps(alpha); std::size_t i=0;
    for(;i+8<=n;i+=8){auto d=_mm256_loadu_ps(dst+i);auto s=_mm256_loadu_ps(src+i);_mm256_storeu_ps(dst+i,_mm256_fmadd_ps(av,s,d));}
    for(;i<n;++i)dst[i]+=alpha*src[i];
}
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void lerp_avx2(float* dst,const float* target,float lr,std::size_t n) noexcept {
    auto l=_mm256_set1_ps(lr); std::size_t i=0;
    for(;i+8<=n;i+=8){auto d=_mm256_loadu_ps(dst+i);auto t=_mm256_loadu_ps(target+i);_mm256_storeu_ps(dst+i,_mm256_fmadd_ps(l,_mm256_sub_ps(t,d),d));}
    for(;i<n;++i)dst[i]+=lr*(target[i]-dst[i]);
}
#endif

bool has_avx2_fma() noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}
float dotv(const float* a,const float* b,std::size_t n) noexcept { return has_avx2_fma()?dot_avx2(a,b,n):dot_scalar(a,b,n); }
float sqdistv(const float* a,const float* b,std::size_t n) noexcept { return has_avx2_fma()?sqdist_avx2(a,b,n):sqdist_scalar(a,b,n); }
void axpyv(float* d,const float* s,float a,std::size_t n) noexcept { if(has_avx2_fma())axpy_avx2(d,s,a,n);else axpy_scalar(d,s,a,n); }
void lerpv(float* d,const float* t,float lr,std::size_t n) noexcept { if(has_avx2_fma())lerp_avx2(d,t,lr,n);else lerp_scalar(d,t,lr,n); }

float cosine_of(std::span<const float> a,std::span<const float> b) noexcept {
    const float ab=dotv(a.data(),b.data(),a.size());
    const float aa=dotv(a.data(),a.data(),a.size()); const float bb=dotv(b.data(),b.data(),b.size());
    return ab/std::sqrt(std::max(aa*bb,1e-12F));
}
}

bool simd_available() noexcept { return has_avx2_fma(); }
std::uint64_t mix64(std::uint64_t x) noexcept { x^=x>>30U;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27U;x*=0x94d049bb133111ebULL;x^=x>>31U;return x; }
std::uint64_t rolling_signature(std::span<const std::uint32_t> w,std::uint64_t seed) noexcept {
    // Preserve recent-token locality in the high bits used by the bucket index.
    // The lower bits remain mixed to separate longer histories inside a bucket.
    std::uint64_t h = seed;
    for (std::size_t i = 0; i < w.size(); ++i) {
        h ^= std::rotl(mix64(static_cast<std::uint64_t>(w[i]) + 0x10001ULL * (i + 1U)),
                       static_cast<int>((i * 9U) & 63U));
    }
    h = mix64(h);
    std::uint64_t recent = 0;
    const std::size_t take = std::min<std::size_t>(4, w.size());
    // Most recent token occupies the most significant byte.
    for (std::size_t k = 0; k < take; ++k) {
        recent |= static_cast<std::uint64_t>(w[w.size() - 1U - k] & 0xFFU)
                  << (56U - static_cast<unsigned>(8U * k));
    }
    const unsigned bits = static_cast<unsigned>(take * 8U);
    if (bits != 0U) {
        const std::uint64_t low_mask = bits == 64U ? 0U : ((std::uint64_t{1} << (64U - bits)) - 1U);
        h = recent | (h & low_mask);
    }
    return h;
}
std::uint64_t token_context_signature(std::span<const std::uint32_t> window,
                                      std::uint32_t alphabet,
                                      std::uint64_t seed) noexcept {
    if(window.empty()) return mix64(seed);
    const unsigned symbol_bits=std::max(1U,static_cast<unsigned>(std::bit_width(alphabet>0?alphabet-1U:0U)));
    std::uint64_t mixed=seed;
    for(std::size_t i=0;i<window.size();++i){
        mixed^=std::rotl(mix64(static_cast<std::uint64_t>(window[i])+0x10001ULL*(i+1U)),
                         static_cast<int>((i*9U)&63U));
    }
    mixed=mix64(mixed);

    std::uint64_t packed=0;
    unsigned used=0;
    for(std::size_t offset=0;offset<window.size()&&used+symbol_bits<=64U;++offset){
        const std::uint64_t symbol=window[window.size()-1U-offset];
        const unsigned shift=64U-used-symbol_bits;
        packed|=(symbol&((std::uint64_t{1}<<symbol_bits)-1U))<<shift;
        used+=symbol_bits;
    }
    if(used==64U) return packed;
    const std::uint64_t low_mask=(std::uint64_t{1}<<(64U-used))-1U;
    return packed|(mixed&low_mask);
}
double hamming_similarity(std::uint64_t a,std::uint64_t b) noexcept { return 1.0-static_cast<double>(std::popcount(a^b))/64.0; }
std::uint64_t hash_dataset(const VectorDataset& d) noexcept {
    std::uint64_t h=mix64(d.token_alphabet)^std::rotl(mix64(d.vector_dim),13);
    for(std::size_t i=0;i<d.tokens.size();++i){h^=mix64(static_cast<std::uint64_t>(d.tokens[i])+i*0x9E37ULL);h=std::rotl(h,7);} 
    for(std::size_t i=0;i<d.targets.size();++i){std::uint32_t bits;std::memcpy(&bits,&d.targets[i],4);h^=mix64(static_cast<std::uint64_t>(bits)+i*0x85EBULL);h=std::rotl(h,9);} return mix64(h);
}

SparseBranchMachine::SparseBranchMachine(Config c):config_(c),rng_state_(c.seed),buckets_(std::size_t{1}<<c.bucket_bits),hot_buckets_(std::size_t{1}<<c.bucket_bits),bucket_last_split_step_(std::size_t{1}<<c.bucket_bits,0),prediction_buffer_(c.vector_dim,0.0F){
    if(c.token_alphabet==0||c.vector_dim==0)throw std::invalid_argument("token_alphabet and vector_dim must be positive");
    if(c.bucket_bits==0||c.bucket_bits>20)throw std::invalid_argument("bucket_bits must be in [1,20]");
    if(c.beam_width==0||c.bucket_scan_limit==0||c.max_edges_per_node==0||
       c.max_specializations_per_bucket==0)throw std::invalid_argument("invalid zero budget");
}
std::uint32_t SparseBranchMachine::bucket(std::uint64_t s) const noexcept{return static_cast<std::uint32_t>(s>>(64U-config_.bucket_bits));}
std::size_t SparseBranchMachine::slot_of(NodeId id) const noexcept {if(id>=id_to_slot_.size())return SIZE_MAX;auto s=id_to_slot_[id];return(s==UINT32_MAX||s>=ids_.size()||ids_[s]!=id)?SIZE_MAX:s;}
bool SparseBranchMachine::contains(NodeId id) const noexcept{return slot_of(id)!=SIZE_MAX;}
NodeId SparseBranchMachine::new_node(std::uint64_t sig,std::span<const float> initial,NodeId parent){
    if(next_id_==kInvalidNode)throw std::overflow_error("node ids exhausted");
    const NodeId id=next_id_++;
    const auto slot=static_cast<std::uint32_t>(ids_.size());
    if(id_to_slot_.size()<=id)id_to_slot_.resize(static_cast<std::size_t>(id)+1,UINT32_MAX);
    id_to_slot_[id]=slot;
    ids_.push_back(id);prototypes_.push_back(sig);visits_.push_back(0);address_visits_.push_back(0);utility_ema_.push_back(0);loss_ema_.push_back(1);address_loss_ema_.push_back(1);phases_.push_back((std::uint8_t)NodePhase::Cold);hot_indexed_.push_back(0);parents_.push_back(parent);
    output_vectors_.insert(output_vectors_.end(),initial.begin(),initial.end());edges_.emplace_back();edges_.back().reserve(config_.max_edges_per_node);buckets_[bucket(sig)].push_back(id);++total_created_;return id;
}
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
            ? static_cast<std::size_t>(next_random(rng_state_) % source.size()) : 0U;
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
        axpyv(out.data(),output_vectors_.data()+slot*config_.vector_dim,node.responsibility,config_.vector_dim);
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
        const float mse_without=sqdistv(without.data(),target.data(),config_.vector_dim)/
                                static_cast<float>(config_.vector_dim);
        const float loss_without=mse_without/std::max(target_energy,1e-5F);
        node.contribution=loss_without-full_loss;
    }
}
void SparseBranchMachine::reinforce_edge(NodeId src,NodeId dst,float delta){
    auto s=slot_of(src);
    if(s==SIZE_MAX||!contains(dst)||delta<=0.0F)return;
    auto& l=edges_[s];
    auto it=std::find_if(l.begin(),l.end(),[&](const Edge&e){return e.dst==dst;});
    if(it==l.end()){
        l.push_back({dst,std::min(delta,1.0F),delta});
    }else{
        it->eligibility=config_.trace_decay*it->eligibility+delta;
        const float target=std::min(it->eligibility,4.0F);
        it->weight=config_.edge_decay*it->weight+
                   config_.edge_learning_rate*(target-it->weight);
        it->weight=std::clamp(it->weight,0.0F,4.0F);
    }
    std::sort(l.begin(),l.end(),[](const auto&a,const auto&b){return a.weight==b.weight?a.dst<b.dst:a.weight>b.weight;});
    if(l.size()>config_.max_edges_per_node)l.resize(config_.max_edges_per_node);
}
void SparseBranchMachine::decay_edges(NodeId source){
    const auto slot=slot_of(source);
    if(slot==SIZE_MAX)return;
    auto& list=edges_[slot];
    for(auto& edge:list){
        edge.weight*=config_.edge_decay;
        edge.eligibility*=config_.trace_decay;
    }
    list.erase(std::remove_if(list.begin(),list.end(),[](const Edge& edge){
        return edge.weight<1e-4F&&edge.eligibility<1e-4F;
    }),list.end());
}
void SparseBranchMachine::update_phase(std::size_t s){NodePhase p=NodePhase::Cold;if(utility_ema_[s]<=config_.dormant_utility)p=NodePhase::Dormant;else if(visits_[s]>=config_.mature_visits)p=NodePhase::Mature;else if(visits_[s]>=config_.warm_visits)p=NodePhase::Warm;phases_[s]=(std::uint8_t)p;}
void SparseBranchMachine::apply_trace_credit(float loss){
    const float global_credit=1.0F-std::min(loss,2.0F);
    float decay=1.0F;
    for(auto it=trace_.rbegin();it!=trace_.rend();++it){
        for(std::size_t i=0;i<it->route.size();++i){
            const auto s=slot_of(it->route[i]);
            if(s==SIZE_MAX)continue;
            const float local=i<it->contribution.size()?std::clamp(it->contribution[i],-1.0F,1.0F):0.0F;
            utility_ema_[s]=.97F*utility_ema_[s]+.03F*(0.65F*global_credit+0.35F*local)*decay;
            update_phase(s);
        }
        decay*=config_.trace_decay;
    }
}
StepStats SparseBranchMachine::step(std::uint32_t token,std::span<const float> target,bool learn){
    if(token>=config_.token_alphabet||target.size()!=config_.vector_dim)
        throw std::out_of_range("invalid token or target dimension");

    history_.push_back(token);
    if(history_.size()>config_.context_width)history_.pop_front();
    std::vector<std::uint32_t> window(history_.begin(),history_.end());
    const auto sig=token_context_signature(window,config_.token_alphabet);
    auto [active,examined]=select_route(sig);

    // Prediction and metrics are fixed before the target is allowed to alter structure.
    aggregate(active,prediction_buffer_);
    const float dimension=static_cast<float>(target.size());
    const float target_energy=dotv(target.data(),target.data(),target.size())/dimension;
    const float mse=sqdistv(prediction_buffer_.data(),target.data(),target.size())/dimension;
    const float nmse=mse/std::max(target_energy,1e-5F);
    const float cos=cosine_of(prediction_buffer_,target);
    compute_counterfactual_contributions(active,target,target_energy,nmse);

    const auto exact_bucket=static_cast<std::size_t>(bucket(sig));
    std::size_t exact_count=0;
    NodeId nearest_exact=kInvalidNode;
    double nearest_similarity=-1.0;
    float nearest_local_loss=std::numeric_limits<float>::infinity();
    float nearest_persistent_loss=std::numeric_limits<float>::infinity();
    std::uint32_t nearest_visits=0;
    for(const NodeId id:buckets_[exact_bucket]){
        const auto slot=slot_of(id);
        if(slot==SIZE_MAX)continue;
        ++exact_count;
        const double similarity=hamming_similarity(prototypes_[slot],sig);
        if(similarity>nearest_similarity){
            nearest_similarity=similarity;
            nearest_exact=id;
            nearest_visits=address_visits_[slot];
            nearest_persistent_loss=address_loss_ema_[slot];
            const float local_mse=sqdistv(output_vectors_.data()+slot*config_.vector_dim,
                                         target.data(),config_.vector_dim)/
                                  static_cast<float>(config_.vector_dim);
            nearest_local_loss=local_mse/std::max(target_energy,1e-5F);
        }
    }

    const bool can_grow=learn||config_.allow_growth_when_frozen;
    const bool missing_exact=exact_count==0;
    const bool cooldown_ready=total_steps_>=bucket_last_split_step_[exact_bucket]+config_.split_cooldown;
    const bool persistent_conflict=nearest_exact!=kInvalidNode&&
        nearest_visits>=config_.split_min_visits&&
        exact_count<config_.max_specializations_per_bucket&&
        cooldown_ready&&
        nearest_similarity<config_.split_context_similarity&&
        nearest_persistent_loss>config_.split_loss_threshold&&
        nearest_local_loss>config_.split_loss_threshold;

    std::uint32_t created=0;
    if(can_grow&&(active.empty()||missing_exact||persistent_conflict)){
        const NodeId parent=nearest_exact!=kInvalidNode?nearest_exact:
                            (active.empty()?kInvalidNode:active.front().id);
        const NodeId id=new_node(sig,target,parent);
        const auto slot=slot_of(id);
        if(slot!=SIZE_MAX){
            visits_[slot]=1;
            address_visits_[slot]=1;
            loss_ema_[slot]=nearest_local_loss;
            address_loss_ema_[slot]=nearest_local_loss;
            utility_ema_[slot]=0.0F;
            hot_buckets_[exact_bucket].push_back(id);
            hot_indexed_[slot]=1;
            update_phase(slot);
        }
        bucket_last_split_step_[exact_bucket]=total_steps_;
        created=1;
    }

    std::vector<NodeId> route;
    std::vector<float> contributions;
    route.reserve(active.size());
    contributions.reserve(active.size());
    for(const auto& node:active){route.push_back(node.id);contributions.push_back(node.contribution);}

    if(learn){
        apply_trace_credit(nmse);
        for(std::size_t rank=0;rank<active.size();++rank){
            auto s=slot_of(active[rank].id);
            if(s==SIZE_MAX)continue;
            if(active[rank].responsibility<config_.min_update_responsibility)continue;
            const bool exact=active[rank].exact_region;
            if(!exact&&active[rank].contribution<=config_.edge_min_contribution)continue;
            if(visits_[s]==0&&!hot_indexed_[s]){
                hot_buckets_[bucket(prototypes_[s])].push_back(ids_[s]);
                hot_indexed_[s]=1;
            }
            ++visits_[s];
            if(exact)++address_visits_[s];
            const float local_mse=sqdistv(output_vectors_.data()+s*config_.vector_dim,
                                          target.data(),config_.vector_dim)/
                                   static_cast<float>(config_.vector_dim);
            const float local_loss=local_mse/std::max(target_energy,1e-5F);
            const float base_lr=phase_of_slot(s)==NodePhase::Mature?
                                config_.mature_learning_rate:config_.node_learning_rate;
            const float responsibility_scale=std::min(1.0F,active[rank].responsibility*
                                                       static_cast<float>(config_.beam_width));
            const float contribution_scale=std::clamp(0.70F+2.0F*active[rank].contribution,
                                                       0.15F,1.25F);
            const float lr=base_lr*responsibility_scale*contribution_scale;
            lerpv(output_vectors_.data()+s*config_.vector_dim,target.data(),lr,config_.vector_dim);
            loss_ema_[s]=.96F*loss_ema_[s]+.04F*local_loss;
            if(exact)address_loss_ema_[s]=.96F*address_loss_ema_[s]+.04F*local_loss;
            utility_ema_[s]=.985F*utility_ema_[s]+.015F*
                std::clamp(active[rank].contribution,-1.0F,1.0F);
            update_phase(s);
        }

        const std::size_t source_limit=std::min<std::size_t>(previous_route_.size(),config_.edge_reinforce_width);
        const std::size_t dest_limit=std::min<std::size_t>(active.size(),config_.edge_reinforce_width);
        for(std::size_t si=0;si<source_limit;++si){
            decay_edges(previous_route_[si]);
            const float source_resp=si<previous_responsibilities_.size()?previous_responsibilities_[si]:1.0F;
            for(std::size_t di=0;di<dest_limit;++di){
                const float helpful=std::max(0.0F,active[di].contribution);
                if(helpful<config_.edge_min_contribution)continue;
                const float delta=source_resp*active[di].responsibility*helpful;
                reinforce_edge(previous_route_[si],active[di].id,delta);
            }
        }
        trace_.push_back({route,contributions,nmse});
        while(trace_.size()>config_.trace_horizon)trace_.pop_front();
    }

    previous_route_=route;
    previous_responsibilities_.clear();
    previous_responsibilities_.reserve(active.size());
    for(const auto& node:active)previous_responsibilities_.push_back(node.responsibility);
    ++total_steps_;
    total_candidates_+=examined;
    total_active_+=route.size();
    return{nmse,cos,(std::uint32_t)route.size(),examined,(std::uint32_t)ids_.size(),created,std::move(route)};
}

void SparseBranchMachine::erase_slot(std::size_t s){auto victim=ids_[s];auto last=ids_.size()-1;if(s!=last){ids_[s]=ids_[last];prototypes_[s]=prototypes_[last];visits_[s]=visits_[last];address_visits_[s]=address_visits_[last];utility_ema_[s]=utility_ema_[last];loss_ema_[s]=loss_ema_[last];address_loss_ema_[s]=address_loss_ema_[last];phases_[s]=phases_[last];hot_indexed_[s]=hot_indexed_[last];parents_[s]=parents_[last];std::copy_n(output_vectors_.data()+last*config_.vector_dim,config_.vector_dim,output_vectors_.data()+s*config_.vector_dim);edges_[s]=std::move(edges_[last]);id_to_slot_[ids_[s]]=(std::uint32_t)s;}ids_.pop_back();prototypes_.pop_back();visits_.pop_back();address_visits_.pop_back();utility_ema_.pop_back();loss_ema_.pop_back();address_loss_ema_.pop_back();phases_.pop_back();hot_indexed_.pop_back();parents_.pop_back();output_vectors_.resize(ids_.size()*config_.vector_dim);edges_.pop_back();id_to_slot_[victim]=UINT32_MAX;}
std::size_t SparseBranchMachine::prune(std::uint32_t minv,float uth){std::size_t n=0;for(std::size_t s=ids_.size();s-->0;){if(visits_[s]<minv&&utility_ema_[s]<uth){erase_slot(s);++n;}}total_pruned_+=n;if(n)rebuild_indexes();return n;}
double SparseBranchMachine::output_distance(std::size_t a,std::size_t b) const noexcept {
    return std::sqrt(sqdistv(output_vectors_.data()+a*config_.vector_dim,
                             output_vectors_.data()+b*config_.vector_dim,
                             config_.vector_dim)/static_cast<float>(config_.vector_dim));
}
void SparseBranchMachine::absorb_node(std::size_t a,std::size_t b){float wa=(float)std::max(1U,visits_[a]),wb=(float)std::max(1U,visits_[b]),inv=1.0F/(wa+wb);auto*av=output_vectors_.data()+a*config_.vector_dim;auto*bv=output_vectors_.data()+b*config_.vector_dim;for(std::uint32_t i=0;i<config_.vector_dim;++i)av[i]=(wa*av[i]+wb*bv[i])*inv;visits_[a]+=visits_[b];address_visits_[a]+=address_visits_[b];utility_ema_[a]=(wa*utility_ema_[a]+wb*utility_ema_[b])*inv;loss_ema_[a]=(wa*loss_ema_[a]+wb*loss_ema_[b])*inv;address_loss_ema_[a]=(wa*address_loss_ema_[a]+wb*address_loss_ema_[b])*inv;for(auto&e:edges_[b])reinforce_edge(ids_[a],e.dst,e.weight);NodeId victim=ids_[b],survivor=ids_[a];for(auto&l:edges_)for(auto&e:l)if(e.dst==victim)e.dst=survivor;for(auto&x:previous_route_)if(x==victim)x=survivor;for(auto&f:trace_)for(auto&x:f.route)if(x==victim)x=survivor;erase_slot(b);}
std::size_t SparseBranchMachine::merge_redundant(std::size_t maxm){std::size_t m=0;for(std::size_t a=0;a<ids_.size()&&m<maxm;++a){for(std::size_t b=a+1;b<ids_.size()&&m<maxm;++b){if(hamming_similarity(prototypes_[a],prototypes_[b])<config_.merge_similarity)continue;if(output_distance(a,b)>config_.merge_vector_distance)continue;if(visits_[b]>visits_[a]){absorb_node(b,a);}else absorb_node(a,b);++m;break;}}total_merged_+=m;if(m)rebuild_indexes();return m;}
void SparseBranchMachine::rebuild_indexes(){for(auto&b:buckets_)b.clear();for(auto&b:hot_buckets_)b.clear();for(std::size_t s=0;s<ids_.size();++s){buckets_[bucket(prototypes_[s])].push_back(ids_[s]);if(hot_indexed_[s])hot_buckets_[bucket(prototypes_[s])].push_back(ids_[s]);}for(auto&l:edges_)l.erase(std::remove_if(l.begin(),l.end(),[&](auto&e){return!contains(e.dst);}),l.end());}
void SparseBranchMachine::prefill_distractors(std::size_t count) {
    std::vector<float> value(config_.vector_dim);
    for(std::size_t i=0;i<count;++i){
        for(auto& component:value){
            const auto sample=static_cast<int>(next_random(rng_state_)%2001U)-1000;
            component=static_cast<float>(sample)/1000.0F;
        }
        (void)new_node(next_random(rng_state_),value);
    }
}
Diagnostics SparseBranchMachine::diagnostics() const noexcept {
    std::uint64_t edge_count=0;
    std::uint64_t edge_capacity=0;
    std::uint64_t bucket_capacity=0;
    std::uint64_t cold=0;
    std::uint64_t warm=0;
    std::uint64_t mature=0;
    std::uint64_t dormant=0;
    for(const auto& list:edges_){
        edge_count+=list.size();
        edge_capacity+=list.capacity();
    }
    for(const auto& entries:buckets_)bucket_capacity+=entries.capacity();
    for(const auto& entries:hot_buckets_)bucket_capacity+=entries.capacity();
    for(std::size_t slot=0;slot<ids_.size();++slot){
        switch(phase_of_slot(slot)){
            case NodePhase::Cold:++cold;break;
            case NodePhase::Warm:++warm;break;
            case NodePhase::Mature:++mature;break;
            case NodePhase::Dormant:++dormant;break;
        }
    }
    const auto denominator=std::max<std::uint64_t>(1,total_steps_);
    const std::uint64_t bytes=ids_.capacity()*sizeof(NodeId)+
        prototypes_.capacity()*sizeof(std::uint64_t)+
        visits_.capacity()*sizeof(std::uint32_t)+
        address_visits_.capacity()*sizeof(std::uint32_t)+
        utility_ema_.capacity()*sizeof(float)+loss_ema_.capacity()*sizeof(float)+
        address_loss_ema_.capacity()*sizeof(float)+phases_.capacity()*sizeof(std::uint8_t)+
        hot_indexed_.capacity()*sizeof(std::uint8_t)+parents_.capacity()*sizeof(NodeId)+
        output_vectors_.capacity()*sizeof(float)+id_to_slot_.capacity()*sizeof(std::uint32_t)+
        edge_capacity*sizeof(Edge)+bucket_capacity*sizeof(NodeId);
    return {total_steps_,ids_.size(),next_id_,edge_count,
            static_cast<double>(total_active_)/static_cast<double>(denominator),
            static_cast<double>(total_candidates_)/static_cast<double>(denominator),
            total_created_,total_merged_,total_pruned_,cold,warm,mature,dormant,
            stale_bucket_refs_skipped_,stale_edge_refs_skipped_,bytes,simd_available()};
}

VectorDataset generate_vector_process(std::size_t length, std::uint32_t alphabet,
                                             std::uint32_t dim, std::uint32_t hidden_dim,
                                             std::uint64_t seed, float noise_std) {
    VectorDataset d{alphabet, dim, {}, {}};
    if (!length || !alphabet || !dim || !hidden_dim) return d;

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::normal_distribution<float> noise(0.0F, noise_std);
    std::uniform_int_distribution<std::uint32_t> token_draw(0, alphabet - 1U);

    // The observed token history fully determines the noise-free target.  The
    // hidden dimension is a feature bank, not an unobservable stochastic state.
    std::vector<float> token_embedding(static_cast<std::size_t>(alphabet) * hidden_dim);
    constexpr std::uint32_t regime_count = 8U;
    std::vector<float> regime_embedding(regime_count * hidden_dim);
    std::vector<float> projection(static_cast<std::size_t>(dim) * hidden_dim);
    for (auto& v : token_embedding) v = 0.55F * normal(rng);
    for (auto& v : regime_embedding) v = 0.22F * normal(rng);
    for (auto& v : projection) v = normal(rng) / std::sqrt(static_cast<float>(hidden_dim));

    d.tokens.reserve(length);
    d.targets.resize(length * dim);
    std::array<std::uint32_t, 6> history{};
    for (auto& x : history) x = token_draw(rng);

    std::vector<float> features(hidden_dim);
    for (std::size_t t = 0; t < length; ++t) {
        // The token stream has broad support instead of nearly deterministic
        // recurrence.  Every target driver remains observable in token history.
        const std::uint32_t innovation = token_draw(rng);
        const std::uint32_t token = static_cast<std::uint32_t>(
            (5U * history[5] + 3U * history[3] + innovation) % alphabet);
        for (std::size_t i = 0; i + 1U < history.size(); ++i) history[i] = history[i + 1U];
        history.back() = token;
        d.tokens.push_back(token);

        const std::uint32_t regime = static_cast<std::uint32_t>(
            (history[5] + 3U * history[3] + 5U * history[1]) % regime_count);
        for (std::uint32_t j = 0; j < hidden_dim; ++j) {
            const float e0 = token_embedding[static_cast<std::size_t>(history[5]) * hidden_dim + j];
            const float e1 = token_embedding[static_cast<std::size_t>(history[4]) * hidden_dim + j];
            const float e2 = token_embedding[static_cast<std::size_t>(history[3]) * hidden_dim + j];
            const float e4 = token_embedding[static_cast<std::size_t>(history[1]) * hidden_dim + j];
            const float rg = regime_embedding[static_cast<std::size_t>(regime) * hidden_dim + j];
            const float cross = e0 * e2 - 0.55F * e1 * e4;
            const float gate = ((history[5] ^ history[2] ^ j) & 1U) ? 1.0F : -1.0F;
            features[j] = std::tanh(0.30F * e0 + 0.28F * e1 + 0.25F * e2 +
                                    0.20F * e4 + 0.24F * rg +
                                    0.16F * gate * cross);
        }
        for (std::uint32_t k = 0; k < dim; ++k) {
            float y = dotv(projection.data() + static_cast<std::size_t>(k) * hidden_dim,
                           features.data(), hidden_dim);
            y += 0.16F * features[(k * 5U + regime) % hidden_dim] *
                 features[(k * 11U + 3U) % hidden_dim];
            d.targets[t * dim + k] = std::tanh(y) + noise(rng);
        }
    }
    return d;
}
void save_dataset(const VectorDataset& d,const std::string&p){std::ofstream o(p,std::ios::binary);if(!o)throw std::runtime_error("cannot open dataset for writing");o.write(kMagic.data(),8);write_pod(o,kDatasetVersion);write_pod(o,d.token_alphabet);write_pod(o,d.vector_dim);auto n=(std::uint64_t)d.tokens.size();write_pod(o,n);auto h=hash_dataset(d);write_pod(o,h);o.write((char*)d.tokens.data(),(std::streamsize)(n*4));o.write((char*)d.targets.data(),(std::streamsize)(n*d.vector_dim*4));if(!o)throw std::runtime_error("dataset write failed");}
VectorDataset load_dataset(const std::string&p){std::ifstream i(p,std::ios::binary);if(!i)throw std::runtime_error("cannot open dataset");std::array<char,8>m{};i.read(m.data(),8);if(m!=kMagic)throw std::runtime_error("invalid dataset magic");if(read_pod<std::uint32_t>(i)!=kDatasetVersion)throw std::runtime_error("unsupported dataset version");VectorDataset d;d.token_alphabet=read_pod<std::uint32_t>(i);d.vector_dim=read_pod<std::uint32_t>(i);auto n=read_pod<std::uint64_t>(i);auto expected=read_pod<std::uint64_t>(i);if(n>(1ULL<<34))throw std::runtime_error("dataset too large");d.tokens.resize((std::size_t)n);d.targets.resize((std::size_t)n*d.vector_dim);i.read((char*)d.tokens.data(),(std::streamsize)(n*4));i.read((char*)d.targets.data(),(std::streamsize)(n*d.vector_dim*4));if(!i||hash_dataset(d)!=expected)throw std::runtime_error("dataset corrupt");return d;}

ExperimentResult run_experiment(const VectorDataset& dataset,
                                      std::size_t warmup,
                                      std::uint64_t seed,
                                      bool strict,
                                      std::size_t prefill,
                                      std::size_t prune_interval,
                                      std::size_t merge_interval) {
    if (dataset.tokens.size() < 2 ||
        dataset.targets.size() != dataset.tokens.size() * dataset.vector_dim) {
        throw std::invalid_argument("invalid dataset");
    }
    if (warmup == 0 || warmup >= dataset.tokens.size()) {
        throw std::invalid_argument("warmup must leave non-empty train and eval partitions");
    }

    Config config;
    config.token_alphabet = dataset.token_alphabet;
    config.vector_dim = dataset.vector_dim;
    config.seed = seed;
    config.allow_growth_when_frozen = !strict;
    SparseBranchMachine model(config);
    model.prefill_distractors(prefill);

    const std::size_t dim = dataset.vector_dim;
    const std::size_t alphabet = dataset.token_alphabet;
    std::vector<double> global_sum(dim, 0.0);
    std::vector<double> token_sum(alphabet * dim, 0.0);
    std::vector<std::uint64_t> token_count(alphabet, 0);
    std::vector<double> pair_sum(alphabet * alphabet * dim, 0.0);
    std::vector<std::uint64_t> pair_count(alphabet * alphabet, 0);
    std::unordered_set<std::uint64_t> training_contexts;
    training_contexts.reserve(warmup * 2U);

    const auto context_key = [&](std::size_t index) noexcept {
        std::uint64_t key = mix64(dataset.tokens[index]);
        if (index > 0) key ^= std::rotl(mix64(dataset.tokens[index - 1U]), 21);
        if (index > 1) key ^= std::rotl(mix64(dataset.tokens[index - 2U]), 42);
        return mix64(key);
    };

    for (std::size_t t = 0; t < warmup; ++t) {
        const auto token = static_cast<std::size_t>(dataset.tokens[t]);
        const auto target = dataset.target(t);
        ++token_count[token];
        for (std::size_t j = 0; j < dim; ++j) {
            global_sum[j] += target[j];
            token_sum[token * dim + j] += target[j];
        }
        if (t > 0) {
            const auto previous = static_cast<std::size_t>(dataset.tokens[t - 1U]);
            const std::size_t pair = previous * alphabet + token;
            ++pair_count[pair];
            for (std::size_t j = 0; j < dim; ++j) pair_sum[pair * dim + j] += target[j];
        }
        training_contexts.insert(context_key(t));
    }

    std::vector<float> global_mean(dim, 0.0F);
    std::vector<float> token_mean(alphabet * dim, 0.0F);
    std::vector<float> pair_mean(alphabet * alphabet * dim, 0.0F);
    for (std::size_t j = 0; j < dim; ++j) {
        global_mean[j] = static_cast<float>(global_sum[j] / static_cast<double>(warmup));
    }
    for (std::size_t token = 0; token < alphabet; ++token) {
        for (std::size_t j = 0; j < dim; ++j) {
            token_mean[token * dim + j] = token_count[token] == 0
                ? global_mean[j]
                : static_cast<float>(token_sum[token * dim + j] /
                                     static_cast<double>(token_count[token]));
        }
    }
    for (std::size_t previous = 0; previous < alphabet; ++previous) {
        for (std::size_t token = 0; token < alphabet; ++token) {
            const std::size_t pair = previous * alphabet + token;
            for (std::size_t j = 0; j < dim; ++j) {
                pair_mean[pair * dim + j] = pair_count[pair] == 0
                    ? token_mean[token * dim + j]
                    : static_cast<float>(pair_sum[pair * dim + j] /
                                         static_cast<double>(pair_count[pair]));
            }
        }
    }

    struct Accumulator {
        double squared_error{};
        double target_energy{};
        double cosine_sum{};
        double centered_energy{};
        std::uint64_t vectors{};
    };
    auto add = [&](Accumulator& acc, std::span<const float> prediction,
                   std::span<const float> target) {
        acc.squared_error += sqdistv(prediction.data(), target.data(), dim);
        acc.target_energy += dotv(target.data(), target.data(), dim);
        acc.cosine_sum += cosine_of(prediction, target);
        for (std::size_t j = 0; j < dim; ++j) {
            const double centered = static_cast<double>(target[j]) - global_mean[j];
            acc.centered_energy += centered * centered;
        }
        ++acc.vectors;
    };
    const auto metrics = [](const Accumulator& acc) {
        VectorMetrics result;
        if (acc.vectors == 0) return result;
        result.normalized_rmse = std::sqrt(acc.squared_error /
                                           std::max(acc.target_energy, 1e-12));
        result.cosine = acc.cosine_sum / static_cast<double>(acc.vectors);
        result.r2 = 1.0 - acc.squared_error / std::max(acc.centered_energy, 1e-12);
        return result;
    };

    Accumulator train_acc;
    Accumulator eval_acc;
    Accumulator mean_acc;
    Accumulator token_acc;
    Accumulator pair_acc;
    Accumulator seen_acc;
    Accumulator unseen_acc;

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t t = 0; t < dataset.tokens.size(); ++t) {
        const bool learn = t < warmup;
        const auto target = dataset.target(t);
        const auto stats = model.step(dataset.tokens[t], target, learn);
        (void)stats;
        const auto prediction = model.last_prediction();
        add(learn ? train_acc : eval_acc, prediction, target);

        if (!learn) {
            add(mean_acc, global_mean, target);
            const std::size_t token = dataset.tokens[t];
            add(token_acc,
                std::span<const float>(token_mean.data() + token * dim, dim), target);
            if (t > 0) {
                const std::size_t previous = dataset.tokens[t - 1U];
                const std::size_t pair = previous * alphabet + token;
                add(pair_acc,
                    std::span<const float>(pair_mean.data() + pair * dim, dim), target);
            } else {
                add(pair_acc, global_mean, target);
            }
            if (training_contexts.contains(context_key(t))) add(seen_acc, prediction, target);
            else add(unseen_acc, prediction, target);
        }

        if (learn && prune_interval != 0 && (t + 1U) % prune_interval == 0) {
            (void)model.prune();
        }
        if (learn && merge_interval != 0 && (t + 1U) % merge_interval == 0) {
            (void)model.merge_redundant();
        }
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    ExperimentResult result;
    result.diagnostics = model.diagnostics();
    result.train = metrics(train_acc);
    result.eval = metrics(eval_acc);
    result.mean_baseline_eval = metrics(mean_acc);
    result.token_baseline_eval = metrics(token_acc);
    result.pair_baseline_eval = metrics(pair_acc);
    result.seen_context_eval = metrics(seen_acc);
    result.unseen_context_eval = metrics(unseen_acc);
    result.seen_context_vectors = seen_acc.vectors;
    result.unseen_context_vectors = unseen_acc.vectors;
    result.steps_per_second = static_cast<double>(dataset.tokens.size()) / elapsed;
    result.elapsed_seconds = elapsed;
    result.strict_freeze = strict;
    result.dataset_hash = hash_dataset(dataset);
    result.vector_dim = dataset.vector_dim;
    result.token_alphabet = dataset.token_alphabet;
    return result;
}
std::string to_json(const ExperimentResult& result) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(8);
    const auto emit_metrics = [&](const char* prefix, const VectorMetrics& metrics) {
        out << "  \"" << prefix << "_nrmse\": " << metrics.normalized_rmse << ",\n"
            << "  \"" << prefix << "_cosine\": " << metrics.cosine << ",\n"
            << "  \"" << prefix << "_r2\": " << metrics.r2 << ",\n";
    };
    out << "{\n"
        << "  \"task\": \"token_conditioned_vector_prediction\",\n"
        << "  \"steps\": " << result.diagnostics.steps << ",\n"
        << "  \"vector_dim\": " << result.vector_dim << ",\n"
        << "  \"token_alphabet\": " << result.token_alphabet << ",\n"
        << "  \"live_nodes\": " << result.diagnostics.live_nodes << ",\n"
        << "  \"edges\": " << result.diagnostics.edges << ",\n"
        << "  \"avg_active\": " << result.diagnostics.avg_active << ",\n"
        << "  \"avg_candidates\": " << result.diagnostics.avg_candidates << ",\n"
        << "  \"created_total\": " << result.diagnostics.created_total << ",\n"
        << "  \"merged_total\": " << result.diagnostics.merged_total << ",\n"
        << "  \"pruned_total\": " << result.diagnostics.pruned_total << ",\n"
        << "  \"estimated_bytes\": " << result.diagnostics.estimated_bytes << ",\n"
        << "  \"simd_enabled\": " << (result.diagnostics.simd_enabled ? "true" : "false") << ",\n";
    emit_metrics("train", result.train);
    emit_metrics("eval", result.eval);
    emit_metrics("mean_baseline_eval", result.mean_baseline_eval);
    emit_metrics("token_baseline_eval", result.token_baseline_eval);
    emit_metrics("pair_baseline_eval", result.pair_baseline_eval);
    emit_metrics("seen_context_eval", result.seen_context_eval);
    emit_metrics("unseen_context_eval", result.unseen_context_eval);
    out << "  \"seen_context_vectors\": " << result.seen_context_vectors << ",\n"
        << "  \"unseen_context_vectors\": " << result.unseen_context_vectors << ",\n"
        << "  \"steps_per_second\": " << result.steps_per_second << ",\n"
        << "  \"elapsed_seconds\": " << result.elapsed_seconds << ",\n"
        << "  \"strict_freeze\": " << (result.strict_freeze ? "true" : "false") << ",\n"
        << "  \"dataset_hash\": " << result.dataset_hash << "\n"
        << "}\n";
    return out.str();
}

} // namespace sbm
