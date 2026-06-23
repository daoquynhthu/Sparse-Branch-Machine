#include "sparse_branch_machine.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
int main(){const std::uint32_t x[]{1,2,3,4};assert(sbm::rolling_signature(x)==sbm::rolling_signature(x));assert(sbm::hamming_similarity(42,42)==1.0);auto d=sbm::generate_vector_process(6000,32,16,20,9);assert(d.tokens.size()==6000&&d.targets.size()==6000*16);auto h=sbm::hash_dataset(d);const char*p="sbm_vector_test.bin";sbm::save_dataset(d,p);auto l=sbm::load_dataset(p);std::remove(p);assert(sbm::hash_dataset(l)==h);sbm::Config c;c.token_alphabet=32;c.vector_dim=16;c.bucket_bits=8;c.seed=9;sbm::SparseBranchMachine m(c);for(std::size_t i=0;i<2000;++i){auto s=m.step(d.tokens[i],d.target(i),true);assert(std::isfinite(s.normalized_mse));assert(s.active_nodes<=c.beam_width);}auto r=sbm::run_experiment(d,2000,9,true,1000,0,0);assert(std::isfinite(r.eval.normalized_rmse));assert(r.diagnostics.steps==6000);std::cout<<"all tests passed; SIMD="<<sbm::simd_available()<<"\n";}
