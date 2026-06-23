#include "sparse_branch_machine.hpp"
#include <cassert>
#include <cstdio>
#include <iostream>

int main(){
    const std::uint32_t x[]{1,2,3,4};
    assert(sbm::rolling_signature(x)==sbm::rolling_signature(x));
    assert(sbm::hamming_similarity(42,42)==1.0);

    auto data=sbm::generate_hidden_fsm(10001,8,64,9);
    const auto h=sbm::hash_dataset(data);
    const char* path="sbm_test_data.bin";
    sbm::save_dataset(data,path);
    auto loaded=sbm::load_dataset(path);
    std::remove(path);
    assert(sbm::hash_dataset(loaded)==h);
    assert(loaded.sequence==data.sequence);

    sbm::Config cfg; cfg.alphabet_size=4; cfg.context_width=4; cfg.bucket_bits=6; cfg.seed=42;
    sbm::SparseBranchMachine m(cfg);
    m.prefill_distractors(1000);
    const auto before=m.live_nodes();
    for(std::uint32_t i=0;i<3000;++i) (void)m.step(i%4U,(i+1U)%4U,true);
    const auto removed=m.prune(100,0.5F);
    assert(removed>0);
    assert(m.live_nodes()<before+3000);
    for(std::uint32_t i=0;i<1000;++i) (void)m.step(i%4U,(i+1U)%4U,false);
    auto d=m.diagnostics();
    assert(d.avg_active<=cfg.beam_width);
    assert(d.live_nodes==m.live_nodes());

    auto r=sbm::run_experiment(data,2000,9,true,10000,500);
    assert(r.diagnostics.steps==10000);
    assert(r.eval_accuracy>=0.0&&r.eval_accuracy<=1.0);
    std::cout<<"all tests passed\n";
}
