#include "sparse_branch_machine.hpp"
#include <charconv>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
template<class T> T number(std::string_view s, const char* n) {
    T v{}; auto [p,e]=std::from_chars(s.data(),s.data()+s.size(),v);
    if(e!=std::errc{}||p!=s.data()+s.size()) throw std::invalid_argument(std::string("invalid ")+n);
    return v;
}
void usage(){
    std::cout << "sbm_benchmark [--length N] [--warmup N] [--seed N] [--dataset FILE] "
                 "[--write-dataset FILE] [--prefill N] [--prune-interval N] [--merge-interval N] "
                 "[--allow-eval-growth] [--output FILE]\n";
}
}
int main(int argc,char**argv){
    try {
        std::size_t length=100001,warmup=20000,prefill=0,prune_interval=0,merge_interval=0;
        std::uint64_t seed=7; bool strict=true;
        std::string dataset_path,write_dataset_path,output="results_cpp_v2.json";
        for(int i=1;i<argc;++i){
            std::string_view a(argv[i]);
            if(a=="--help"||a=="-h"){usage();return 0;}
            if(a=="--allow-eval-growth"){strict=false;continue;}
            if(i+1>=argc) throw std::invalid_argument("missing value after "+std::string(a));
            std::string_view v(argv[++i]);
            if(a=="--length") length=number<std::size_t>(v,"length");
            else if(a=="--warmup") warmup=number<std::size_t>(v,"warmup");
            else if(a=="--seed") seed=number<std::uint64_t>(v,"seed");
            else if(a=="--prefill") prefill=number<std::size_t>(v,"prefill");
            else if(a=="--prune-interval") prune_interval=number<std::size_t>(v,"prune-interval");
            else if(a=="--merge-interval") merge_interval=number<std::size_t>(v,"merge-interval");
            else if(a=="--dataset") dataset_path=v;
            else if(a=="--write-dataset") write_dataset_path=v;
            else if(a=="--output") output=v;
            else throw std::invalid_argument("unknown argument: "+std::string(a));
        }
        auto dataset = dataset_path.empty()?sbm::generate_hidden_fsm(length,8,64,seed):sbm::load_dataset(dataset_path);
        if(!write_dataset_path.empty()) sbm::save_dataset(dataset,write_dataset_path);
        auto result=sbm::run_experiment(dataset,warmup,seed,strict,prefill,prune_interval,merge_interval);
        auto json=sbm::to_json(result);
        std::ofstream out(output,std::ios::binary); if(!out) throw std::runtime_error("cannot open output");
        out<<json; std::cout<<json;
    } catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}
}
