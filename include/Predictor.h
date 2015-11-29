
#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

/////////////////////////////////
// INCLUDES
/////////////////////////////////
#include "PSO.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <string.h>
#include <float.h>
#include <limits.h>


/////////////////////////////////
// TYPEDEFS
/////////////////////////////////
typedef unsigned int uint;


/////////////////////////////////
// NAMESPACE
/////////////////////////////////
namespace Predictor
{


/////////////////////////////////
// KERNEL
/////////////////////////////////
template <typename NumericalType>
struct MuSigmaScale
{
    NumericalType mu, sigma, scale;
};

template <typename NumericalType, int Window, int Nodes>
struct Kernel
{
    MuSigmaScale<NumericalType> data[Window * Nodes];
};


/////////////////////////////////
// KERNEL MANAGER
/////////////////////////////////
enum TeacherFunction
{
    GAUSSIAN,
    GAUSSIAN_UP_1,
    GAUSSIAN_DOWN_1
};
template <typename NumericalType, int Window, int Nodes, TeacherFunction TFunc = GAUSSIAN>
class KernelOperation
{
    KernelOperation();
    KernelOperation(const KernelOperation &other);
    KernelOperation &operator=(const KernelOperation &other);

public:
    // print
    static void print(const Kernel<NumericalType, Window, Nodes> &krnl)
    {
        for (uint i = 0; i < Window; ++i)
        {
            std::cerr << i << ": ";
            for (uint k = 0; k < Nodes; ++k)
            {
                const uint idx = i * Nodes + k;
                std::cerr << "(" << krnl.data[idx].mu
                          << "," << krnl.data[idx].sigma
                          << "," << krnl.data[idx].scale << ") ";
            }
            std::cerr << std::endl;
        }
    }

    // reset kernel
    static void reset(Kernel<NumericalType, Window, Nodes> *krnl,
                      const NumericalType mu,
                      const NumericalType sigma,
                      const NumericalType scale)
    {
        for (uint i = 0; i < Window * Nodes; ++i)
        {
            krnl->data[i].mu = mu;
            krnl->data[i].sigma = sigma;
            krnl->data[i].scale = scale;
        }
    }

    // compute simple gaussian with custom scaling
    static NumericalType gaussian(const NumericalType mu,
                                  const NumericalType sigma,
                                  const NumericalType scale,
                                  const NumericalType value)
    {
        const NumericalType d = value - mu;
        //return scale * std::exp((NumericalType)-0.5 * d * d / (sigma * sigma));
        return scale / ((NumericalType)1.0 + sigma * d * d);
    }

    // gaussian that is 1 above target
    static NumericalType gaussian_up1(const NumericalType mu,
                                      const NumericalType sigma,
                                      const NumericalType scale,
                                      const NumericalType value)
    {
        if (value > mu) return (NumericalType)1.0;
        return gaussian(mu, sigma, scale, value);
    }

    // gaussian that is 1 above target
    static NumericalType gaussian_down1(const NumericalType mu,
                                        const NumericalType sigma,
                                        const NumericalType scale,
                                        const NumericalType value)
    {
        if (value < mu) return (NumericalType)1.0;
        return gaussian(mu, sigma, scale, value);
    }

    // compute kernel response to an input vector
    static NumericalType response(const Kernel<NumericalType, Window, Nodes> &krnl,
                                  const std::vector<NumericalType> &data,
                                  const uint startAt,
                                  const NumericalType minVal,
                                  const NumericalType scale)
    {
        NumericalType sum = (NumericalType)0.0;
        for (uint i = 0; i < Window; ++i)
        {
            for (uint k = 0; k < Nodes; ++k)
            {
                const uint idx = i * Nodes + k;
                sum += gaussian(krnl.data[idx].mu,
                                krnl.data[idx].sigma,
                                krnl.data[idx].scale,
                                scale * (data[startAt + i] - minVal));
            }
        }
        return sum / (NumericalType)(Window * Nodes);
    }

    // compute min max normalization
    static void normalize(const std::vector<NumericalType> &data,
                          const uint startAt,
                          NumericalType *minVal,
                          NumericalType *scaling)
    {
        // normalize data window to 0-1
        NumericalType vmin = (NumericalType)FLT_MAX,
                      vmax = (NumericalType)-FLT_MAX;
        for (uint k = 0; k < Window; ++k)
        {
            const uint idx = startAt + k;
            if (data[idx] < vmin) vmin = data[idx];
            if (data[idx] > vmax) vmax = data[idx];
        }
        NumericalType scale;
        if (vmin == vmax)
        {
            scale = (NumericalType)1.0;
            vmin -= (NumericalType)0.5;
        }
        else
        {
            scale = (NumericalType)1.0 / (vmax - vmin);
        }

        // commit
        *minVal = vmin;
        *scaling = scale;
    }

    // compute convolution error of input sequence
    static NumericalType convolution(const Kernel<NumericalType, Window, Nodes> &krnl,
                                     const std::vector<NumericalType> &data,
                                     const NumericalType targetValue,
                                     const NumericalType targetSigma,
                                     const uint ahead = 1)
    {
        const uint limit = data.size() - Window - ahead;
        NumericalType error = (NumericalType)0.0;
        for (uint i = 0; i <= limit; ++i)
        {
            // normalize window
            NumericalType vmin, scale;
            normalize(data, i, &vmin, &scale);
            // compute teacher and kernel response
            NumericalType teacher;
            const NumericalType testVal = scale * (data[i + Window + ahead - 1] - vmin);
            if (TFunc == GAUSSIAN)
            {
                teacher = gaussian(targetValue, targetSigma,
                                  (NumericalType)1.0, testVal);
            }
            else if (TFunc == GAUSSIAN_UP_1)
            {
                teacher = gaussian_up1(targetValue, targetSigma,
                                      (NumericalType)1.0, testVal);
            }
            else if (TFunc == GAUSSIAN_DOWN_1)
            {
                teacher = gaussian_down1(targetValue, targetSigma,
                                        (NumericalType)1.0, testVal);
            }
            const NumericalType resp = response(krnl, data, i, vmin, scale);
            const NumericalType tmp = resp - teacher;
            error += tmp * tmp;
        }
        return error / (NumericalType)(limit + 1u);
    }

    // compute convolution error of input sequence
    static void applyKernel(const Kernel<NumericalType, Window, Nodes> &krnl,
                            const std::vector<NumericalType> &data,
                            std::vector<NumericalType> *out,
                            const uint ahead = 1)
    {
        out->clear();
        out->resize(data.size(), (NumericalType)0.0);
        const uint limit = data.size() - Window - ahead;
        for (uint i = 0; i <= limit; ++i)
        {
            // normalize window
            NumericalType vmin, scale;
            normalize(data, i, &vmin, &scale);
            out->at(i + Window + ahead - 1) = response(krnl, data, i, vmin, scale);
        }
    }

    // dump normalized window
    static void normWindow(const std::vector<NumericalType> &data,
                           std::vector<NumericalType> *out,
                           const uint startAt,
                           const uint ahead)
    {
        out->clear();
        out->resize(Window + ahead, (NumericalType)0.0);
        NumericalType vmin, scale;
        normalize(data, startAt, &vmin, &scale);
        for (uint i = 0; i < Window + ahead; ++i)
        {
            // normalize window
            out->at(i) = scale * (data[startAt + i] - vmin);
        }
    }

    // store kernel to file
    static void storeKernel(const Kernel<NumericalType, Window, Nodes> &krnl,
                            const char *file)
    {
        std::ofstream ofs;
        ofs.open(file, std::ios::binary | std::ios::out);
        const uint32_t w = Window,
                       n = Nodes,
                       s = sizeof(NumericalType);
        ofs.write((char *)&w, sizeof(const uint32_t));
        ofs.write((char *)&n, sizeof(const uint32_t));
        ofs.write((char *)&s, sizeof(const uint32_t));
        for (uint i = 0; i < Window * Nodes; ++i)
        {
            ofs.write((char *)&(krnl.data[i].mu), sizeof(NumericalType));
            ofs.write((char *)&(krnl.data[i].sigma), sizeof(NumericalType));
            ofs.write((char *)&(krnl.data[i].scale), sizeof(NumericalType));
        }
        ofs.close();
    }

    // store kernel vector to file
    static void storeKernelVector(const std::vector<Kernel<NumericalType, Window, Nodes> > &vec,
                                  const char *file)
    {
        std::ofstream ofs;
        ofs.open(file, std::ios::binary | std::ios::out);
        const uint32_t w = Window,
                       n = Nodes,
                       s = sizeof(NumericalType),
                       size = vec.size();
        ofs.write((char *)&w, sizeof(const uint32_t));
        ofs.write((char *)&n, sizeof(const uint32_t));
        ofs.write((char *)&s, sizeof(const uint32_t));
        ofs.write((char *)&size, sizeof(const uint32_t));
        for (uint v = 0; v < size; ++v)
        {
            for (uint i = 0; i < Window * Nodes; ++i)
            {
                ofs.write((char *)&(vec[v].data[i].mu), sizeof(NumericalType));
                ofs.write((char *)&(vec[v].data[i].sigma), sizeof(NumericalType));
                ofs.write((char *)&(vec[v].data[i].scale), sizeof(NumericalType));
            }
        }
        ofs.close();
    }

    // store kernel to file
    static bool loadKernel(Kernel<NumericalType, Window, Nodes> *krnl,
                           const char *file)
    {
        std::ifstream ifs;
        ifs.open(file, std::ios::binary | std::ios::in);
        uint32_t w, n, s;
        ifs.read((char *)&w, sizeof(uint32_t));
        ifs.read((char *)&n, sizeof(uint32_t));
        ifs.read((char *)&s, sizeof(uint32_t));
        if (w != Window)
        {
            std::cerr << "loading kernel failed: wrong window size!\n";
            return false;
        }
        if (n != Nodes)
        {
            std::cerr << "loading kernel failed: wrong number of nodes!\n";
            return false;
        }
        if (s != sizeof(NumericalType))
        {
            std::cerr << "loading kernel failed: wrong floating point type!\n";
            return false;
        }
        for (uint i = 0; i < Window * Nodes; ++i)
        {
            ifs.read((char *)&(krnl->data[i].mu), sizeof(NumericalType));
            ifs.read((char *)&(krnl->data[i].sigma), sizeof(NumericalType));
            ifs.read((char *)&(krnl->data[i].scale), sizeof(NumericalType));
        }
        ifs.close();
        return true;
    }

    // store kernel vector to file
    static bool loadKernelVector(std::vector<Kernel<NumericalType, Window, Nodes> > *vec,
                                 const char *file)
    {
        vec->clear();
        std::ifstream ifs;
        ifs.open(file, std::ios::binary | std::ios::in);
        uint32_t w, n, s, size;
        ifs.read((char *)&w, sizeof(uint32_t));
        ifs.read((char *)&n, sizeof(uint32_t));
        ifs.read((char *)&s, sizeof(uint32_t));
        ifs.read((char *)&size, sizeof(uint32_t));
        if (w != Window)
        {
            std::cerr << "loading kernel failed: wrong window size!\n";
            return false;
        }
        if (n != Nodes)
        {
            std::cerr << "loading kernel failed: wrong number of nodes!\n";
            return false;
        }
        if (s != sizeof(NumericalType))
        {
            std::cerr << "loading kernel failed: wrong floating point type!\n";
            return false;
        }
        vec->resize(size);
        for (uint v = 0; v < size; ++v)
        {
            for (uint i = 0; i < Window * Nodes; ++i)
            {
                ifs.read((char *)&(vec->at(v).data[i].mu), sizeof(NumericalType));
                ifs.read((char *)&(vec->at(v).data[i].sigma), sizeof(NumericalType));
                ifs.read((char *)&(vec->at(v).data[i].scale), sizeof(NumericalType));
            }
        }
        ifs.close();
        return true;
    }
};


/////////////////////////////////
// KERNEL MANAGER
/////////////////////////////////
template <typename NumericalType, int Window, int Nodes, TeacherFunction TFunc = GAUSSIAN>
class KernelOptimizer
{
    KernelOptimizer();
    KernelOptimizer(const KernelOptimizer &other);
    KernelOptimizer &operator=(const KernelOptimizer &other);

public:
    struct Payload
    {
        NumericalType minSigma;
        const std::vector<NumericalType> *data;
        NumericalType targetValue,
                      targetSigma;
        uint targetAhead;
    };


    static NumericalType optimize(Kernel<NumericalType, Window, Nodes> *result,
                                  const std::vector<NumericalType> &data,
                                  const NumericalType targetValue,
                                  const NumericalType targetSigma,
                                  const uint targetAhead,
                                  const NumericalType lowerLimit,
                                  const NumericalType upperLimit,
                                  const uint particleCount,
                                  const NumericalType breakScore)
    {
        Payload pl;
        pl.data = &data;
        pl.targetValue = targetValue;
        pl.targetSigma = targetSigma;
        pl.targetAhead = targetAhead;
        pl.minSigma = 0.00001f;

        PSO<NumericalType, Window * Nodes * 3> pso(particleCount, scoreFunc, (const void *)&pl);
        pso.init(lowerLimit, upperLimit);
        NumericalType s;
        while ((s = pso.step()) > breakScore)
        {
            std::cout << s << std::endl;
        }
        const NumericalType *values = pso.getBest();

        uint cnt = 0;
        for (uint i = 0; i < Window * Nodes; ++i)
        {
            result->data[i].mu    = values[cnt++];
            result->data[i].sigma = std::fabs(values[cnt++]) + pl.minSigma;
            result->data[i].scale = values[cnt++];
        }

        return pso.getScore();
    }


    static NumericalType scoreFunc(const NumericalType *values, const uint dim, const void *payload)
    {
        // check dim
        if (dim != Window * Nodes * 3)
        {
            std::cerr << "score: invalid dimensions\n";
            return (NumericalType)FLT_MAX;
        }

        // get payload
        const Payload *pl = (const Payload *)payload;

        // assemble kernel
        Kernel<NumericalType, Window, Nodes> krnl;
        uint cnt = 0;
        for (uint i = 0; i < Window * Nodes; ++i)
        {
            krnl.data[i].mu    = values[cnt++];
            krnl.data[i].sigma = std::fabs(values[cnt++]) + pl->minSigma;
            krnl.data[i].scale = values[cnt++];            
        }

        // compute cost
        return KernelOperation<NumericalType, Window, Nodes, TFunc>::convolution(krnl,
                                                                                 *(pl->data),
                                                                                 pl->targetValue,
                                                                                 pl->targetSigma,
                                                                                 pl->targetAhead);
    }


};

}

#endif
