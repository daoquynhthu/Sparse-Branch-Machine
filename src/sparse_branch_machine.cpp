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
    T value{}; in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("dataset read failed"); return value;
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
    for(;i<n;++i)s+=a[i]*b[i]; return s;
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
double hamming_similarity(std::uint64_t a,std::uint64_t b) noexcept { return 1.0-static_cast<double>(std::popcount(a^b))/64.0; }
std::uint64_t hash_dataset(const VectorDataset& d) noexcept {
    std::uint64_t h=mix64(d.token_alphabet)^std::rotl(mix64(d.vector_dim),13);
    for(std::size_t i=0;i<d.tokens.size();++i){h^=mix64(static_cast<std::uint64_t>(d.tokens[i])+i*0x9E37ULL);h=std::rotl(h,7);} 
    for(std::size_t i=0;i<d.targets.size();++i){std::uint32_t bits;std::memcpy(&bits,&d.targets[i],4);h^=mix64(static_cast<std::uint64_t>(bits)+i*0x85EBULL);h=std::rotl(h,9);} return mix64(h);
}

SparseBranchMachine::SparseBranchMachine(Config c):config_(c),rng_state_(c.seed),buckets_(std::size_t{1}<<c.bucket_bits),hot_buckets_(std::size_t{1}<<c.bucket_bits),prediction_buffer_(c.vector_dim,0.0F){
    if(c.token_alphabet==0||c.vector_dim==0)throw std::invalid_argument("token_alphabet and vector_dim must be positive");
    if(c.bucket_bits==0||c.bucket_bits>20)throw std::invalid_argument("bucket_bits must be in [1,20]");
    if(c.beam_width==0||c.bucket_scan_limit==0||c.max_edges_per_node==0)throw std::invalid_argument("invalid zero budget");
}
std::uint32_t SparseBranchMachine::bucket(std::uint64_t s) const noexcept{return static_cast<std::uint32_t>(s>>(64U-config_.bucket_bits));}
std::size_t SparseBranchMachine::slot_of(NodeId id) const noexcept {if(id>=id_to_slot_.size())return SIZE_MAX;auto s=id_to_slot_[id];return(s==UINT32_MAX||s>=ids_.size()||ids_[s]!=id)?SIZE_MAX:s;}
bool SparseBranchMachine::contains(NodeId id) const noexcept{return slot_of(id)!=SIZE_MAX;}
NodeId SparseBranchMachine::new_node(std::uint64_t sig,std::span<const float> initial,NodeId parent){
    if(next_id_==kInvalidNode)throw std::overflow_error("node ids exhausted"); NodeId id=next_id_++; auto slot=static_cast<std::uint32_t>(ids_.size());
    if(id_to_slot_.size()<=id)id_to_slot_.resize(static_cast<std::size_t>(id)+1,UINT32_MAX);id_to_slot_[id]=slot;
    ids_.push_back(id);prototypes_.push_back(sig);visits_.push_back(0);utility_ema_.push_back(0);loss_ema_.push_back(1);phases_.push_back((std::uint8_t)NodePhase::Cold);hot_indexed_.push_back(0);parents_.push_back(parent);
    output_vectors_.insert(output_vectors_.end(),initial.begin(),initial.end());edges_.emplace_back();edges_.back().reserve(config_.max_edges_per_node);buckets_[bucket(sig)].push_back(id);++total_created_;return id;
}
bool SparseBranchMachine::push_unique(std::vector<NodeId>& v,NodeId id) const noexcept{if(!contains(id)||std::find(v.begin(),v.end(),id)!=v.end())return false;v.push_back(id);return true;}
std::vector<NodeId> SparseBranchMachine::candidate_ids(std::uint64_t sig) {
    std::vector<NodeId> out;
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
            (void)push_unique(out, id);
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
            (void)push_unique(out, edges_[slot][i].dst);
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
double SparseBranchMachine::score(std::size_t s, std::uint64_t sig) const noexcept {
    const double sim = hamming_similarity(prototypes_[s], sig);
    const double rel = 1.0 / (1.0 + loss_ema_[s]);
    const double nov = 1.0 / std::sqrt(static_cast<double>(visits_[s]) + 1.0);
    const double exact_region = bucket(prototypes_[s]) == bucket(sig) ? 0.42 : 0.0;
    const double phase_bias = phase_of_slot(s) == NodePhase::Dormant ? -0.18 : 0.0;
    return exact_region + 0.72 * sim + 0.10 * rel + 0.02 * nov + phase_bias;
}
std::pair<std::vector<SparseBranchMachine::ScoredNode>,std::uint32_t> SparseBranchMachine::select_route(std::uint64_t sig){auto c=candidate_ids(sig);std::vector<ScoredNode>s;s.reserve(c.size());for(auto id:c){auto sl=slot_of(id);if(sl!=SIZE_MAX)s.push_back({score(sl,sig),id});}auto keep=std::min<std::size_t>(s.size(),config_.beam_width);std::partial_sort(s.begin(),s.begin()+keep,s.end(),[](auto&a,auto&b){return a.score==b.score?a.id>b.id:a.score>b.score;});s.resize(keep);return{std::move(s),(std::uint32_t)c.size()};}
void SparseBranchMachine::aggregate(const std::vector<ScoredNode>& a,std::span<float> out) const {std::fill(out.begin(),out.end(),0);if(a.empty())return;double ws=0;for(auto&x:a){auto s=slot_of(x.id);if(s==SIZE_MAX)continue;float w=static_cast<float>(std::exp(3.0 * std::clamp(x.score, -2.0, 2.0)));axpyv(out.data(),output_vectors_.data()+s*config_.vector_dim,w,config_.vector_dim);ws+=w;}if(ws>0){float inv=(float)(1.0/ws);for(auto&v:out)v*=inv;}}
void SparseBranchMachine::reinforce_edge(NodeId src,NodeId dst,float delta){auto s=slot_of(src);if(s==SIZE_MAX||!contains(dst))return;auto&l=edges_[s];auto it=std::find_if(l.begin(),l.end(),[&](auto&e){return e.dst==dst;});if(it==l.end())l.push_back({dst,delta,delta});else{it->eligibility=config_.trace_decay*it->eligibility+delta;it->weight=config_.edge_decay*it->weight+it->eligibility;}std::sort(l.begin(),l.end(),[](auto&a,auto&b){return a.weight==b.weight?a.dst<b.dst:a.weight>b.weight;});if(l.size()>config_.max_edges_per_node)l.resize(config_.max_edges_per_node);}
void SparseBranchMachine::update_phase(std::size_t s){NodePhase p=NodePhase::Cold;if(utility_ema_[s]<=config_.dormant_utility)p=NodePhase::Dormant;else if(visits_[s]>=config_.mature_visits)p=NodePhase::Mature;else if(visits_[s]>=config_.warm_visits)p=NodePhase::Warm;phases_[s]=(std::uint8_t)p;}
void SparseBranchMachine::apply_trace_credit(float loss){float credit=1.0F-std::min(loss,2.0F);float decay=1;for(auto it=trace_.rbegin();it!=trace_.rend();++it){for(auto id:it->route){auto s=slot_of(id);if(s==SIZE_MAX)continue;utility_ema_[s]=.97F*utility_ema_[s]+.03F*credit*decay;update_phase(s);}decay*=config_.trace_decay;}}
StepStats SparseBranchMachine::step(std::uint32_t token,std::span<const float> target,bool learn){
    if(token>=config_.token_alphabet||target.size()!=config_.vector_dim)throw std::out_of_range("invalid token or target dimension");history_.push_back(token);if(history_.size()>config_.context_width)history_.pop_front();std::vector<std::uint32_t>w(history_.begin(),history_.end());auto sig=rolling_signature(w);auto[active,examined]=select_route(sig);
    aggregate(active,prediction_buffer_);float target_energy=dotv(target.data(),target.data(),target.size())/target.size();float mse=sqdistv(prediction_buffer_.data(),target.data(),target.size())/target.size();float nmse=mse/std::max(target_energy,1e-5F);float cos=cosine_of(prediction_buffer_,target);std::uint32_t created=0;
    double best=0;for(auto&x:active){auto s=slot_of(x.id);if(s!=SIZE_MAX)best=std::max(best,hamming_similarity(prototypes_[s],sig));}
    std::size_t same_bucket = 0;
    for (const auto& selected : active) {
        const auto slot = slot_of(selected.id);
        if (slot != SIZE_MAX && bucket(prototypes_[slot]) == bucket(sig)) ++same_bucket;
    }
    bool exact_region_has_live_node = false;
    for (const NodeId id : buckets_[bucket(sig)]) {
        if (contains(id)) { exact_region_has_live_node = true; break; }
    }
    const bool missing_exact_region = !exact_region_has_live_node;
    const bool unresolved_conflict = false; // v4: growth is address novelty, not per-sample error.
    if ((learn || config_.allow_growth_when_frozen) &&
        (active.empty() || missing_exact_region || unresolved_conflict)) {
        NodeId parent = active.empty() ? kInvalidNode : active.front().id;
        auto id = new_node(sig, target, parent);
        if (active.size() >= config_.beam_width) active.resize(config_.beam_width - 1U);
        active.insert(active.begin(), {1.0, id});
        created = 1;
        aggregate(active, prediction_buffer_);
        mse = sqdistv(prediction_buffer_.data(), target.data(), target.size()) / static_cast<float>(target.size());
        nmse = mse / std::max(target_energy, 1e-5F);
        cos = cosine_of(prediction_buffer_, target);
    }
    std::vector<NodeId> route;for(auto&x:active)route.push_back(x.id);
    if(learn){apply_trace_credit(nmse);for(std::size_t rank=0;rank<active.size();++rank){auto s=slot_of(active[rank].id);if(s==SIZE_MAX)continue;if(visits_[s]==0&&!hot_indexed_[s]){hot_buckets_[bucket(prototypes_[s])].push_back(ids_[s]);hot_indexed_[s]=1;}++visits_[s];float lr=(phase_of_slot(s)==NodePhase::Mature?config_.mature_learning_rate:config_.node_learning_rate)/(1.0F+.15F*(float)rank);lerpv(output_vectors_.data()+s*config_.vector_dim,target.data(),lr,config_.vector_dim);loss_ema_[s]=.96F*loss_ema_[s]+.04F*nmse;utility_ema_[s]=.985F*utility_ema_[s]+.015F*(1.0F-std::min(nmse,2.0F));update_phase(s);}for(auto src:previous_route_)for(std::size_t r=0;r<route.size();++r)reinforce_edge(src,route[r],1.0F/(r+1));trace_.push_back({route,nmse});while(trace_.size()>config_.trace_horizon)trace_.pop_front();}
    previous_route_=route;++total_steps_;total_candidates_+=examined;total_active_+=route.size();return{nmse,cos,(std::uint32_t)route.size(),examined,(std::uint32_t)ids_.size(),created,std::move(route)};
}

void SparseBranchMachine::erase_slot(std::size_t s){auto victim=ids_[s];auto last=ids_.size()-1;if(s!=last){ids_[s]=ids_[last];prototypes_[s]=prototypes_[last];visits_[s]=visits_[last];utility_ema_[s]=utility_ema_[last];loss_ema_[s]=loss_ema_[last];phases_[s]=phases_[last];hot_indexed_[s]=hot_indexed_[last];parents_[s]=parents_[last];std::copy_n(output_vectors_.data()+last*config_.vector_dim,config_.vector_dim,output_vectors_.data()+s*config_.vector_dim);edges_[s]=std::move(edges_[last]);id_to_slot_[ids_[s]]=(std::uint32_t)s;}ids_.pop_back();prototypes_.pop_back();visits_.pop_back();utility_ema_.pop_back();loss_ema_.pop_back();phases_.pop_back();hot_indexed_.pop_back();parents_.pop_back();output_vectors_.resize(ids_.size()*config_.vector_dim);edges_.pop_back();id_to_slot_[victim]=UINT32_MAX;}
std::size_t SparseBranchMachine::prune(std::uint32_t minv,float uth){std::size_t n=0;for(std::size_t s=ids_.size();s-->0;){if(visits_[s]<minv&&utility_ema_[s]<uth){erase_slot(s);++n;}}total_pruned_+=n;if(n)rebuild_indexes();return n;}
double SparseBranchMachine::output_distance(std::size_t a,std::size_t b) const noexcept{return std::sqrt(sqdistv(output_vectors_.data()+a*config_.vector_dim,output_vectors_.data()+b*config_.vector_dim,config_.vector_dim)/config_.vector_dim);}
void SparseBranchMachine::absorb_node(std::size_t a,std::size_t b){float wa=(float)std::max(1U,visits_[a]),wb=(float)std::max(1U,visits_[b]),inv=1.0F/(wa+wb);auto*av=output_vectors_.data()+a*config_.vector_dim;auto*bv=output_vectors_.data()+b*config_.vector_dim;for(std::uint32_t i=0;i<config_.vector_dim;++i)av[i]=(wa*av[i]+wb*bv[i])*inv;visits_[a]+=visits_[b];utility_ema_[a]=(wa*utility_ema_[a]+wb*utility_ema_[b])*inv;loss_ema_[a]=(wa*loss_ema_[a]+wb*loss_ema_[b])*inv;for(auto&e:edges_[b])reinforce_edge(ids_[a],e.dst,e.weight);NodeId victim=ids_[b],survivor=ids_[a];for(auto&l:edges_)for(auto&e:l)if(e.dst==victim)e.dst=survivor;for(auto&x:previous_route_)if(x==victim)x=survivor;for(auto&f:trace_)for(auto&x:f.route)if(x==victim)x=survivor;erase_slot(b);}
std::size_t SparseBranchMachine::merge_redundant(std::size_t maxm){std::size_t m=0;for(std::size_t a=0;a<ids_.size()&&m<maxm;++a){for(std::size_t b=a+1;b<ids_.size()&&m<maxm;++b){if(hamming_similarity(prototypes_[a],prototypes_[b])<config_.merge_similarity)continue;if(output_distance(a,b)>config_.merge_vector_distance)continue;if(visits_[b]>visits_[a]){absorb_node(b,a);}else absorb_node(a,b);++m;break;}}total_merged_+=m;if(m)rebuild_indexes();return m;}
void SparseBranchMachine::rebuild_indexes(){for(auto&b:buckets_)b.clear();for(auto&b:hot_buckets_)b.clear();for(std::size_t s=0;s<ids_.size();++s){buckets_[bucket(prototypes_[s])].push_back(ids_[s]);if(hot_indexed_[s])hot_buckets_[bucket(prototypes_[s])].push_back(ids_[s]);}for(auto&l:edges_)l.erase(std::remove_if(l.begin(),l.end(),[&](auto&e){return!contains(e.dst);}),l.end());}
void SparseBranchMachine::prefill_distractors(std::size_t n){std::vector<float> z(config_.vector_dim);for(std::size_t i=0;i<n;++i){for(auto&v:z)v=((int)(next_random(rng_state_)%2001)-1000)/1000.0F;new_node(next_random(rng_state_),z);}}
Diagnostics SparseBranchMachine::diagnostics() const noexcept{std::uint64_t ec=0,ecap=0,bcap=0,c=0,w=0,m=0,d=0;for(auto&l:edges_){ec+=l.size();ecap+=l.capacity();}for(auto&b:buckets_)bcap+=b.capacity();for(auto&b:hot_buckets_)bcap+=b.capacity();for(std::size_t s=0;s<ids_.size();++s)switch(phase_of_slot(s)){case NodePhase::Cold:++c;break;case NodePhase::Warm:++w;break;case NodePhase::Mature:++m;break;case NodePhase::Dormant:++d;break;}auto den=std::max<std::uint64_t>(1,total_steps_);std::uint64_t bytes=ids_.capacity()*4+prototypes_.capacity()*8+visits_.capacity()*4+utility_ema_.capacity()*4+loss_ema_.capacity()*4+phases_.capacity()+hot_indexed_.capacity()+parents_.capacity()*4+output_vectors_.capacity()*4+id_to_slot_.capacity()*4+ecap*sizeof(Edge)+bcap*4;return{total_steps_,ids_.size(),next_id_,ec,(double)total_active_/den,(double)total_candidates_/den,total_created_,total_merged_,total_pruned_,c,w,m,d,stale_bucket_refs_skipped_,stale_edge_refs_skipped_,bytes,simd_available()};}

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
    std::vector<float> regime_embedding(4U * hidden_dim);
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
        // Mostly structured second-order transitions, with a small observed
        // innovation.  The innovation affects the current token and is thus
        // available to the predictor; it is not hidden target noise.
        const std::uint32_t innovation = token_draw(rng) % 7U;
        const std::uint32_t token = static_cast<std::uint32_t>(
            (13U * history[5] + 7U * history[4] + 3U * history[2] + innovation) % alphabet);
        for (std::size_t i = 0; i + 1U < history.size(); ++i) history[i] = history[i + 1U];
        history.back() = token;
        d.tokens.push_back(token);

        const std::uint32_t regime = static_cast<std::uint32_t>(
            (history[5] + 2U * history[3] + history[1]) & 3U);
        for (std::uint32_t j = 0; j < hidden_dim; ++j) {
            const float e0 = token_embedding[static_cast<std::size_t>(history[5]) * hidden_dim + j];
            const float e1 = token_embedding[static_cast<std::size_t>(history[4]) * hidden_dim + j];
            const float e2 = token_embedding[static_cast<std::size_t>(history[3]) * hidden_dim + j];
            const float e4 = token_embedding[static_cast<std::size_t>(history[1]) * hidden_dim + j];
            const float rg = regime_embedding[static_cast<std::size_t>(regime) * hidden_dim + j];
            const float cross = e0 * e2 - 0.45F * e1 * e4;
            const float gate = ((history[5] ^ history[2] ^ j) & 1U) ? 1.0F : -1.0F;
            features[j] = std::tanh(0.80F * e0 + 0.12F * e1 + 0.07F * e2 +
                                    0.08F * rg + 0.05F * gate * cross);
        }
        for (std::uint32_t k = 0; k < dim; ++k) {
            float y = dotv(projection.data() + static_cast<std::size_t>(k) * hidden_dim,
                           features.data(), hidden_dim);
            y += 0.10F * features[(k * 5U + regime) % hidden_dim] *
                 features[(k * 11U + 3U) % hidden_dim];
            d.targets[t * dim + k] = std::tanh(y) + noise(rng);
        }
    }
    return d;
}
void save_dataset(const VectorDataset& d,const std::string&p){std::ofstream o(p,std::ios::binary);if(!o)throw std::runtime_error("cannot open dataset for writing");o.write(kMagic.data(),8);write_pod(o,kDatasetVersion);write_pod(o,d.token_alphabet);write_pod(o,d.vector_dim);auto n=(std::uint64_t)d.tokens.size();write_pod(o,n);auto h=hash_dataset(d);write_pod(o,h);o.write((char*)d.tokens.data(),(std::streamsize)(n*4));o.write((char*)d.targets.data(),(std::streamsize)(n*d.vector_dim*4));if(!o)throw std::runtime_error("dataset write failed");}
VectorDataset load_dataset(const std::string&p){std::ifstream i(p,std::ios::binary);if(!i)throw std::runtime_error("cannot open dataset");std::array<char,8>m{};i.read(m.data(),8);if(m!=kMagic)throw std::runtime_error("invalid dataset magic");if(read_pod<std::uint32_t>(i)!=kDatasetVersion)throw std::runtime_error("unsupported dataset version");VectorDataset d;d.token_alphabet=read_pod<std::uint32_t>(i);d.vector_dim=read_pod<std::uint32_t>(i);auto n=read_pod<std::uint64_t>(i);auto expected=read_pod<std::uint64_t>(i);if(n>(1ULL<<34))throw std::runtime_error("dataset too large");d.tokens.resize((std::size_t)n);d.targets.resize((std::size_t)n*d.vector_dim);i.read((char*)d.tokens.data(),(std::streamsize)(n*4));i.read((char*)d.targets.data(),(std::streamsize)(n*d.vector_dim*4));if(!i||hash_dataset(d)!=expected)throw std::runtime_error("dataset corrupt");return d;}

ExperimentResult run_experiment(const VectorDataset& d,std::size_t warmup,std::uint64_t seed,bool strict,std::size_t prefill,std::size_t pi,std::size_t mi){if(d.tokens.size()<2||d.targets.size()!=d.tokens.size()*d.vector_dim)throw std::invalid_argument("invalid dataset");if(warmup>d.tokens.size())throw std::invalid_argument("warmup too large");Config c;c.token_alphabet=d.token_alphabet;c.vector_dim=d.vector_dim;c.seed=seed;c.allow_growth_when_frozen=!strict;SparseBranchMachine model(c);model.prefill_distractors(prefill);
    std::vector<double> mean(d.vector_dim,0);for(std::size_t t=0;t<warmup;++t)for(std::uint32_t j=0;j<d.vector_dim;++j)mean[j]+=d.targets[t*d.vector_dim+j];for(auto&v:mean)v/=std::max<std::size_t>(1,warmup);
    struct Acc { double se=0, te=0, cos=0, base=0; std::uint64_t vectors=0; };
    Acc tr, ev;
    auto start=std::chrono::steady_clock::now();
    for(std::size_t t=0;t<d.tokens.size();++t){bool learn=t<warmup;auto target=d.target(t);auto st=model.step(d.tokens[t],target,learn);auto pred=model.last_prediction();double se=sqdistv(pred.data(),target.data(),d.vector_dim),te=dotv(target.data(),target.data(),d.vector_dim),be=0;for(std::uint32_t j=0;j<d.vector_dim;++j){double x=target[j]-mean[j];be+=x*x;}auto&a=learn?tr:ev;a.se+=se;a.te+=te;a.cos+=st.cosine;a.base+=be;++a.vectors;if(learn&&pi&&(t+1)%pi==0)(void)model.prune();if(learn&&mi&&(t+1)%mi==0)(void)model.merge_redundant();}
    auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    auto metric=[](const Acc&a){VectorMetrics m;m.normalized_rmse=std::sqrt(a.se/std::max(a.te,1e-12));m.cosine=a.cos/std::max<double>(1.0,static_cast<double>(a.vectors));m.r2=1.0-a.se/std::max(a.base,1e-12);return m;};ExperimentResult r;r.diagnostics=model.diagnostics();r.train=metric(tr);r.eval=metric(ev);r.mean_baseline_eval={1.0,0.0,0.0};r.steps_per_second=d.tokens.size()/elapsed;r.elapsed_seconds=elapsed;r.strict_freeze=strict;r.dataset_hash=hash_dataset(d);r.vector_dim=d.vector_dim;r.token_alphabet=d.token_alphabet;return r;}
std::string to_json(const ExperimentResult&r){std::ostringstream o;o.setf(std::ios::fixed);o.precision(8);o<<"{\n  \"task\": \"token_conditioned_vector_prediction\",\n  \"steps\": "<<r.diagnostics.steps<<",\n  \"vector_dim\": "<<r.vector_dim<<",\n  \"token_alphabet\": "<<r.token_alphabet<<",\n  \"live_nodes\": "<<r.diagnostics.live_nodes<<",\n  \"edges\": "<<r.diagnostics.edges<<",\n  \"avg_active\": "<<r.diagnostics.avg_active<<",\n  \"avg_candidates\": "<<r.diagnostics.avg_candidates<<",\n  \"created_total\": "<<r.diagnostics.created_total<<",\n  \"merged_total\": "<<r.diagnostics.merged_total<<",\n  \"pruned_total\": "<<r.diagnostics.pruned_total<<",\n  \"estimated_bytes\": "<<r.diagnostics.estimated_bytes<<",\n  \"simd_enabled\": "<<(r.diagnostics.simd_enabled?"true":"false")<<",\n  \"train_nrmse\": "<<r.train.normalized_rmse<<",\n  \"train_cosine\": "<<r.train.cosine<<",\n  \"train_r2\": "<<r.train.r2<<",\n  \"eval_nrmse\": "<<r.eval.normalized_rmse<<",\n  \"eval_cosine\": "<<r.eval.cosine<<",\n  \"eval_r2\": "<<r.eval.r2<<",\n  \"steps_per_second\": "<<r.steps_per_second<<",\n  \"elapsed_seconds\": "<<r.elapsed_seconds<<",\n  \"strict_freeze\": "<<(r.strict_freeze?"true":"false")<<",\n  \"dataset_hash\": "<<r.dataset_hash<<"\n}\n";return o.str();}

} // namespace sbm
