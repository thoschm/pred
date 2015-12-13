

#include <Predictor.h>
#include <fstream>
#include <stdlib.h>
#include <deque>
#include <omp.h>


using namespace Predictor;


#define WINDOW 200u
#define NODES  2u
#define LOOK_AHEAD 50u

#define PARTICLES 100u
#define BREAK_ERROR 0.1f
#define BREAK_LOOPS 5000u

#define TSIGMA 10.0f
#define KRNL_MIN  -10.0f
#define KRNL_MAX  11.0f
#define KRNL_STEP 0.2f


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
    if (argc != 3)
    {
        std::cerr << "Usage:\n   learn <sequence.txt> <devices>\n";
        return EXIT_FAILURE;
    }

    // load input sequence
    std::vector<float> indata;
    if (!loadSequence(&indata, argv[1]))
    {
        return EXIT_FAILURE;
    }
    uint devs = atoi(argv[2]);

/*
    for (uint i = 0; i < 20000u; ++i)
    {
        indata.push_back(std::sin(0.1 * i) + std::sin(0.05 * (i + 17)) * std::cos(0.02 * (i + 23)) + 0.01f * i + 5.0f * std::sin(0.01f * (i + 100)));
    }
*/
    dumpSequence(indata, "sine.txt");

    std::deque<float> targets;
    std::vector<Kernel<float, WINDOW, NODES> > vec;
    for (float k = KRNL_MIN; k <= KRNL_MAX; k += KRNL_STEP)
    {
        targets.push_back(k);
    }

    omp_set_num_threads(devs);
#pragma omp parallel shared(targets) shared(vec)
    for (bool run = true; run; )
    {
        float target;
        const uint tid = omp_get_thread_num();

#pragma omp critical
        {
            if (targets.size() > 0)
            {
                target = targets.front();
                targets.pop_front();
            }
            else
            {
                run = false;
            }
        }

        Kernel<float, WINDOW, NODES> krnl;
        float score;
        if (run)
        {
            score = KernelOptimizer<float, WINDOW, NODES>::optimizeOCL(&krnl,
                                                                       indata, target,
                                                                       TSIGMA,
                                                                       LOOK_AHEAD,
                                                                       0.0f, 1.0f,
                                                                       PARTICLES,
                                                                       BREAK_ERROR,
                                                                       BREAK_LOOPS,
                                                                       false,
                                                                       tid);
        }

#pragma omp critical
        {
            if (run)
            {
                std::cerr << "GPU" << tid << ": target = " << target << ", error = " << score << std::endl;
                vec.push_back(krnl);
            }
            else
            {
                std::cerr << "GPU" << tid << ": done" << std::endl;
            }
        }
    }

/*
    // learn kernels
    uint c = 0;
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
                                                        BREAK_LOOPS);
        std::cerr << "--> optimization done." << std::endl << std::endl;
        vec.push_back(krnl);

        //KernelOperation<float, WINDOW, NODES>::print(krnl);
    }
*/
    // store result
    KernelOperation<float, WINDOW, NODES>::storeKernelVector(vec, "kernels.bin");
    std::cerr << "results stored as kernels.bin\n";

    return 0;
}
