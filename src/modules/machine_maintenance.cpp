#include "sbm/machine.hpp"
#include "sbm/detail/random.hpp"
#include "sbm/math.hpp"
#include "sbm/detail/vector_ops.hpp"
#include <algorithm>
#include <cmath>

namespace sbm {
void SparseBranchMachine::erase_slot(std::size_t s){auto victim=ids_[s];auto last=ids_.size()-1;if(s!=last){ids_[s]=ids_[last];prototypes_[s]=prototypes_[last];visits_[s]=visits_[last];address_visits_[s]=address_visits_[last];utility_ema_[s]=utility_ema_[last];loss_ema_[s]=loss_ema_[last];address_loss_ema_[s]=address_loss_ema_[last];phases_[s]=phases_[last];hot_indexed_[s]=hot_indexed_[last];parents_[s]=parents_[last];std::copy_n(output_vectors_.data()+last*config_.vector_dim,config_.vector_dim,output_vectors_.data()+s*config_.vector_dim);edges_[s]=std::move(edges_[last]);id_to_slot_[ids_[s]]=(std::uint32_t)s;}ids_.pop_back();prototypes_.pop_back();visits_.pop_back();address_visits_.pop_back();utility_ema_.pop_back();loss_ema_.pop_back();address_loss_ema_.pop_back();phases_.pop_back();hot_indexed_.pop_back();parents_.pop_back();output_vectors_.resize(ids_.size()*config_.vector_dim);edges_.pop_back();id_to_slot_[victim]=UINT32_MAX;}
std::size_t SparseBranchMachine::prune(std::uint32_t minv,float uth){std::size_t n=0;for(std::size_t s=ids_.size();s-->0;){if(visits_[s]<minv&&utility_ema_[s]<uth){erase_slot(s);++n;}}total_pruned_+=n;if(n)rebuild_indexes();return n;}
double SparseBranchMachine::output_distance(std::size_t a,std::size_t b) const noexcept {
    return std::sqrt(detail::squared_distance(output_vectors_.data()+a*config_.vector_dim,
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
            const auto sample=static_cast<int>(detail::next_random(rng_state_)%2001U)-1000;
            component=static_cast<float>(sample)/1000.0F;
        }
        (void)new_node(detail::next_random(rng_state_),value);
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


} // namespace sbm
