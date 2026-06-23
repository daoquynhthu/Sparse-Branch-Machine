#include "sbm/machine.hpp"
#include "sbm/detail/vector_ops.hpp"
#include "sbm/math.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sbm {
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
    const float target_energy=detail::dot(target.data(),target.data(),target.size())/dimension;
    const float mse=detail::squared_distance(prediction_buffer_.data(),target.data(),target.size())/dimension;
    const float nmse=mse/std::max(target_energy,1e-5F);
    const float cos=detail::cosine(prediction_buffer_,target);
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
            const float local_mse=detail::squared_distance(output_vectors_.data()+slot*config_.vector_dim,
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
            const float local_mse=detail::squared_distance(output_vectors_.data()+s*config_.vector_dim,
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
            detail::lerp(output_vectors_.data()+s*config_.vector_dim,target.data(),lr,config_.vector_dim);
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


} // namespace sbm
