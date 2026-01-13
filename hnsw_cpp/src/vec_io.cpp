#include "vec_io.h"

namespace vecio{
    std::vector<std::vector<float>> read_fvecs(const std::string& filename){
        std::ifstream input(filename,std::ios::binary);
        if (!input){
            throw std::runtime_error("Cannot open fvces files");
        }

        std::vector<std::vector<float>> vectors;

        while(true){
            int32_t dim;
            if(!input.read(reinterpret_cast<char*>(&dim),sizeof(int32_t))){
                break; // End of file
            }

            std::vector<float> vec(dim);
            input.read(reinterpret_cast<char*>(vec.data()),dim*sizeof(float));
            if (!input){
                throw std::runtime_error("Corrupted fvecs files");
            }

            vectors.push_back(std::move(vec));

        }
        return vectors;
    }

    std::vector<std::vector<int>> read_ivecs(const std::string& filename){
        std::ifstream input(filename,std::ios::binary);
        if (!input){
            throw std::runtime_error("Cannot open ivecs files");
        }
        std::vector<std::vector<int>> vectors;

        while (true){
            int32_t dim;
            if (!input.read(reinterpret_cast<char*>(&dim),sizeof(int32_t))){break;}

            std::vector<int> vec(dim);
            input.read(reinterpret_cast<char*>(vec.data()),dim*sizeof(int));
            if (!input){
                throw std::runtime_error("Corrupted ivecs files");
            }
            vectors.push_back(std::move(vec));
        }
        return vectors;
    }
}