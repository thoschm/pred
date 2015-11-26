

#include <Predictor.h>
#include <fstream>
#include <PSO.h>


using namespace Predictor;


#define WINDOW 100
#define NODES  2
#define TARGET 1.0
#define TSIGMA 10.0

#define RAD(_a) ((_a) * M_PI / 180.0)

float func(uint i)
{
    return std::sin(2.0 * RAD(i));// + std::sin(13.0 * RAD(i)) + std::cos(5.0 * RAD(i));// +
           //std::sin(0.5 * RAD(i + 10)) + std::sin(0.2 * RAD(i + 50)) + std::cos(2.0 * RAD(i + 100)) +
           //std::sin(1.5 * RAD(i + 250)) * std::sin(0.5 * RAD(i + 150));
}


float scoreFunc(const float *array, const uint dim, const void *payload)
{
    if (array[0] < 5.0) return FLT_MAX;
    if (array[1] < 5.0) return FLT_MAX;
    float sum = 0.0f;
    for (uint i = 0; i < dim; ++i)
    {
        sum += array[i] * array[i];
    }
    return sum;
}


int main(int argc, char **argv)
{


    std::vector<float> data, out, err;
    std::ofstream ofs;
    ofs.open("sine.txt");
    for (uint i = 0; i < 1000u; ++i)
    {
        //data.push_back(1.0f);
        data.push_back(func(i));
        ofs << i << " " << data.back() << std::endl;
    }
    /*for (uint i = 1000; i < 2000u; ++i)
    {
        data.push_back(0.0f);
        //data.push_back(std::sin(i * M_PI / 180.0f));
        ofs << i << " " << data.back() << std::endl;
    }
    for (uint i = 2000; i < 3000u; ++i)
    {
        data.push_back(-3.0f);
        //data.push_back(std::sin(i * M_PI / 180.0f));
        ofs << i << " " << data.back() << std::endl;
    }
    for (uint i = 3000; i < 4000u; ++i)
    {
        data.push_back(-10.0f);
        //data.push_back(std::sin(i * M_PI / 180.0f));
        ofs << i << " " << data.back() << std::endl;
    }*/
    ofs.close();
/*

    KernelOperation<float, WINDOW, NODES>::normWindow(data, &out, 120, 10);
    ofs.open("norm.txt");
    for (uint i = 0; i < out.size(); ++i)
    {
        ofs << i << " " << out[i] << std::endl;
    }
    ofs.close();

*/

    Kernel<float, WINDOW, NODES> krnl;
    KernelOptimizer<float, WINDOW, NODES>::optimize(krnl, data, TARGET, TSIGMA, 1u, 0.0f, 1.0f, 50u, 0.001f);
    KernelOperation<float, WINDOW, NODES>::applyKernel(krnl, data, &out, 1u);
    KernelOperation<float, WINDOW, NODES>::print(krnl);
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
