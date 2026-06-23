#include "sbm/machine.hpp"
#include <stdexcept>

namespace sbm {
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

} // namespace sbm
