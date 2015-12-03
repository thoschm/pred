

#include <Predictor.h>
#include <fstream>
#include <stdlib.h>


using namespace Predictor;


#define WINDOW 48u
#define NODES  5u
#define PARTICLES 50u
#define LOOK_AHEAD 3u
#define BREAK_ERROR 0.01f

#define TSIGMA 10.0f
#define KRNL_MIN -1.0f
#define KRNL_MAX  2.0f
#define KRNL_STEP 0.1f


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
        std::cerr << "Usage:\n   learn <sequence.txt>\n";
        return EXIT_FAILURE;
    }

    // load input sequence
    std::vector<float> indata;
    if (!loadSequence(&indata, argv[1]))
    {
        return EXIT_FAILURE;
    }

    // learn kernels
    uint c = 0;
    std::vector<Kernel<float, WINDOW, NODES> > vec;
    for (float k = KRNL_MIN; k <= KRNL_MAX; k += KRNL_STEP, ++c)
    {
        std::cerr << "*** learning kernel " << c << " ***" << std::endl
                  << "kernel target: " << k << ", target sigma: " << TSIGMA << std::endl;
        Kernel<float, WINDOW, NODES> krnl;
        KernelOptimizer<float, WINDOW, NODES>::optimize(&krnl,
                                                        indata, k,
                                                        TSIGMA,
                                                        LOOK_AHEAD,
                                                        0.0f, 1.0f,
                                                        PARTICLES,
                                                        BREAK_ERROR,
                                                        1000u);
        std::cerr << "--> optimization done." << std::endl << std::endl;
        vec.push_back(krnl);
    }

    // store result
    KernelOperation<float, WINDOW, NODES>::storeKernelVector(vec, "kernels.bin");
    std::cerr << "results stored as kernels.bin\n";

    // get activations
    std::vector<float> activations, prediction;
    KernelOperation<float, WINDOW, NODES>::queryKernels(&activations, &prediction, vec, indata, 11600);

    std::cerr << "results:" << std::endl;
    for (uint i = 0; i < vec.size(); ++i)
    {
        std::cerr << "target " << prediction[i] << ": " << activations[i] << std::endl;
    }
    float mu, sig;
    KernelOptimizer<float, WINDOW, NODES>::weightedMeanSigma(&mu, &sig, activations, prediction);
    std::cerr << "MU.: " << mu << std::endl
              << "SIG: " << sig << std::endl;

    return 0;
}
