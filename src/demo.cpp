

#include <Predictor.h>
#include <fstream>


using namespace Predictor;


#define WINDOW 20
#define NODES  5
#define TARGET 1.0
#define TSIGMA 10.0


bool loadSequence(std::vector<float> *seq, const char *file)
{
    seq->clear();
    std::ifstream ifs;
    ifs.open(file, std::ios::in);
    if (ifs.fail())
    {
        std::cerr << "failed to read sequence!\n";
        return false;
    }
    float val;
    for ( ; ; )
    {
        ifs >> val;
        if (ifs.eof()) break;
        seq->push_back(val);
    }
    ifs.close();
    return true;
}


bool dumpSequence(const std::vector<float> &seq, const char *file)
{
    std::ofstream ofs;
    ofs.open(file, std::ios::out);
    if (ofs.fail())
    {
        std::cerr << "failed to write sequence!\n";
        return false;
    }
    for (uint i = 0; i < seq.size(); ++i)
    {
        ofs << seq[i] << std::endl;
    }
    ofs.close();
    return true;
}


int main(int argc, char **argv)
{
    std::vector<float> indata, outdata;




    
      
    /*

    Kernel<float, WINDOW, NODES> krnl, krnl2;
    KernelOptimizer<float, WINDOW, NODES, GAUSSIAN>::optimize(&krnl, data, TARGET, TSIGMA, 50u, 0.0f, 1.0f, 100u, 0.1f);
    KernelOperation<float, WINDOW, NODES, GAUSSIAN>::applyKernel(krnl, data, &out, 1u);
    KernelOperation<float, WINDOW, NODES, GAUSSIAN>::print(krnl);

    KernelOperation<float, WINDOW, NODES, GAUSSIAN>::reset(&krnl2, 1.0f, 1.0f, 1.0f);

    std::vector<Kernel<float, WINDOW, NODES> > vec, vec2;
    vec.push_back(krnl);
    vec.push_back(krnl2);
    KernelOperation<float, WINDOW, NODES, GAUSSIAN>::storeKernelVector(vec, "vec.bin");
    KernelOperation<float, WINDOW, NODES, GAUSSIAN>::loadKernelVector(&vec2, "vec.bin");
    KernelOperation<float, WINDOW, NODES, GAUSSIAN>::print(vec[0]);
    KernelOperation<float, WINDOW, NODES, GAUSSIAN>::print(vec[1]);




    ofs.open("out.txt");
    for (uint i = 0; i < out.size(); ++i)
    {
        ofs << i << " " << out[i] << std::endl;
    }
    ofs.close();


    ofs.open("kernel.txt");
    for (uint i = 0; i < WINDOW; ++i)
    {
        for (uint k = 0; k < NODES; ++k)
        {
            const uint idx = i * NODES + k;
            if (krnl.data[idx].scale > 1.0 && krnl.data[idx].sigma > 5.0)
            {
                ofs << i << " " << krnl.data[idx].mu << std::endl;
            }
        }
    }
    ofs.close();
    /*
    ofs.open("scale.txt");
    for (uint i = 0; i < KERNEL_SIZE; ++i)
    {
       ofs << i << " " << krnl.data[i].scale << std::endl;
    }
    ofs.close();
    ofs.open("sigma.txt");
    for (uint i = 0; i < KERNEL_SIZE; ++i)
    {
       ofs << i << " " << krnl.data[i].sigma << std::endl;
    }
    ofs.close();*/


    return 0;
}
