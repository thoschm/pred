

#include <Predictor.h>
#include <fstream>
#include <stdlib.h>


using namespace Predictor;


#define WINDOW 50u
#define NODES  2u
#define LOOK_AHEAD 10u


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
    // check args
    if (argc != 2)
    {
        std::cerr << "Usage:\n   predict <sequence.txt>\n";
        return EXIT_FAILURE;
    }

    // load input sequence
    std::vector<float> indata;
    if (!loadSequence(&indata, argv[1]))
    {
        return EXIT_FAILURE;
    }

    // store result
    std::vector<Kernel<float, WINDOW, NODES> > vec;
    KernelOperation<float, WINDOW, NODES>::loadKernelVector(&vec, "kernels.bin");
    std::cerr << "loaded kernels.bin\n";

    // get activations
    std::vector<float> activations;
    KernelOperation<float, WINDOW, NODES>::queryKernels(&activations, vec, indata, 11700u);
    std::cerr << "results:" << std::endl;
    for (uint i = 0; i < vec.size(); ++i)
    {
        std::cerr << "target " << vec[i].targetVal << ": " << activations[i] << std::endl;
    }
    float mu, sig;
    KernelOptimizer<float, WINDOW, NODES>::weightedMeanSigma(&mu, &sig, vec, activations);
    std::cerr << "MU.: " << mu << std::endl
              << "SIG: " << sig << std::endl;

    return 0;
}
