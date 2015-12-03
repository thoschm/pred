

#include <Predictor.h>
#include <fstream>
#include <stdlib.h>


using namespace Predictor;


#define WINDOW 48u
#define NODES  5u
#define LOOK_AHEAD 3u


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
    std::vector<float> indata, outdata;
    if (!loadSequence(&indata, argv[1]))
    {
        return EXIT_FAILURE;
    }

    // store result
    std::vector<Kernel<float, WINDOW, NODES> > vec;
    KernelOperation<float, WINDOW, NODES>::loadKernelVector(&vec, "kernels.bin");
    std::cerr << "loaded kernels.bin\n";

    // get activations
    std::vector<float> activations, prediction;
    float mu = 0.0f, sig = 0.0f;
    const uint size = indata.size() - WINDOW;
    outdata.resize(indata.size() + LOOK_AHEAD, 0.0f);
    for (uint i = 0; i <= size; ++i)
    {
        KernelOperation<float, WINDOW, NODES>::queryKernels(&activations, &prediction, vec, indata, i);
        KernelOptimizer<float, WINDOW, NODES>::weightedMeanSigma(&mu, &sig, activations, prediction);
        outdata[i + WINDOW + LOOK_AHEAD - 1] = mu;
    }

    // write
    dumpSequence(outdata, "pred.txt");

    return 0;
}
