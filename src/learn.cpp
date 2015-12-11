

#include <Predictor.h>
#include <fstream>
#include <stdlib.h>


using namespace Predictor;


#define WINDOW 500u
#define NODES  2u
#define LOOK_AHEAD 100u

#define PARTICLES 100u
#define BREAK_ERROR 0.1f
#define BREAK_LOOPS 5000u

#define TSIGMA 10.0f
#define KRNL_MIN  -2.0f
#define KRNL_MAX  3.0f
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
    /*if (!loadSequence(&indata, argv[1]))
    {
        return EXIT_FAILURE;
    }*/

    for (uint i = 0; i < 2000u; ++i)
    {
        indata.push_back(std::sin(0.1 * i) + std::sin(0.05 * (i + 17)) * std::cos(0.02 * (i + 23)) + 0.01f * i + 5.0f * std::sin(0.01f * (i + 100)));
    }

    dumpSequence(indata, "sine.txt");

    // learn kernels
    uint c = 0;
    std::vector<Kernel<float, WINDOW, NODES> > vec;
    for (float k = KRNL_MIN; k <= KRNL_MAX; k += KRNL_STEP, ++c)
    {
        std::cerr << "*** learning kernel " << c << " ***" << std::endl
                  << "kernel target: " << k << ", target sigma: " << TSIGMA << std::endl;
        Kernel<float, WINDOW, NODES> krnl;
        KernelOptimizer<float, WINDOW, NODES>::optimizeOCL(&krnl,
                                                        indata, k,
                                                        TSIGMA,
                                                        LOOK_AHEAD,
                                                        0.0f, 1.0f,
                                                        PARTICLES,
                                                        BREAK_ERROR,
                                                        BREAK_LOOPS);
        std::cerr << "--> optimization done." << std::endl << std::endl;
        vec.push_back(krnl);

        //KernelOperation<float, WINDOW, NODES>::print(krnl);
    }

    // store result
    KernelOperation<float, WINDOW, NODES>::storeKernelVector(vec, "kernels.bin");
    std::cerr << "results stored as kernels.bin\n";

    return 0;
}
