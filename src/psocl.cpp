

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include "XorShift.h"
#include <float.h>


using namespace Predictor;


/////////////////////////////////
// Particle
/////////////////////////////////
template <typename NumericalType, int Dim>
struct Particle
{
    NumericalType x[Dim],
                  v[Dim],
                  best[Dim],
                  score,
                  tmp;
};


/////////////////////////////////
// PSO class
/////////////////////////////////
template <typename NumericalType, int Dim>
class PSO
{
    PSO(const PSO &other);
    PSO &operator=(const PSO &other);

public:
    // alloc particles
    PSO(const uint particleCount,
        NumericalType (*scoreFunction)(const NumericalType *, const uint, const void *),
        const void *payload) :
        mParticleCount(particleCount),
        mParticles(NULL),
        scoreFunc(scoreFunction),
        mPayload(payload),
        mW((NumericalType)0.0),
        mCP((NumericalType)0.0),
        mCG((NumericalType)0.0),
        mBestScore((NumericalType)FLT_MAX)
    {
        memset(mBestPos, 0, Dim * sizeof(NumericalType));
        mParticles = new Particle<NumericalType, Dim>[particleCount];
    }

    // release particles
    ~PSO()
    {
        delete[] mParticles;
    }

    // init
    void init(const NumericalType lowerLimit,
              const NumericalType upperLimit,
              const NumericalType w  = (NumericalType)0.7,
              const NumericalType cp = (NumericalType)1.4,
              const NumericalType cg = (NumericalType)1.4)
    {
        // params
        mW = w;
        mCP = cp;
        mCG = cg;
        const NumericalType diff = upperLimit - lowerLimit;

        // init particles
        for (uint i = 0; i < mParticleCount; ++i)
        {
            // for each particle
            Particle<NumericalType, Dim> &par = mParticles[i];
            par.score = (NumericalType)FLT_MAX;
            par.tmp = (NumericalType)FLT_MAX;

            // init
            for (uint d = 0; d < Dim; ++d)
            {
                // x, v, best
                par.x[d] = mRnd.uniform() * diff + lowerLimit;
                par.v[d] = mRnd.uniform() * (NumericalType)2.0 * diff - diff;
                par.best[d] = mRnd.uniform() * diff + lowerLimit;
            }
        }

        // reset swarm
        mBestScore = (NumericalType)FLT_MAX;
        memset(mBestPos, 0, Dim * sizeof(NumericalType));
    }

    // one optimization step
    NumericalType step()
    {
        for (uint i = 0; i < mParticleCount; ++i)
        {
            // for each particle
            Particle<NumericalType, Dim> &par = mParticles[i];

            // compute cost
            par.tmp = scoreFunc(par.x, Dim, mPayload);
        }

        for (uint i = 0; i < mParticleCount; ++i)
        {
            // for each particle
            Particle<NumericalType, Dim> &par = mParticles[i];

            // update scores
            if (par.tmp < par.score)
            {
                par.score = par.tmp;
                memcpy(par.best, par.x, Dim * sizeof(NumericalType));

                // swarm
                if (par.tmp < mBestScore)
                {
                    mBestScore = par.tmp;
                    memcpy(mBestPos, par.x, Dim * sizeof(NumericalType));
                }
            }

            // update position x
            for (uint d = 0; d < Dim; ++d)
            {
                // uniform random values
                const NumericalType rp = mRnd.uniform(),
                                    rg = mRnd.uniform();
                // update velocity and pos
                par.v[d] = mW * par.v[d] +
                           mCP * rp * (par.best[d] - par.x[d]) +
                           mCG * rg * (mBestPos[d] - par.x[d]);
                par.x[d] += par.v[d];
            }
        }
        return mBestScore;
    }

    // get particle pointer
    const Particle<NumericalType, Dim> *getParticles()
    {
        return mParticles;
    }

    // get current best
    const NumericalType *getBest()
    {
        return mBestPos;
    }

    NumericalType getScore()
    {
        return mBestScore;
    }

    // print particles
    void print()
    {
        std::cerr << "best score: " << mBestScore << std::endl;
        std::cerr << "best pos..: ";
        for (uint i = 0; i < Dim; ++i)
        {
            std::cerr << mBestPos[i] << " ";
        }
        std::cerr << std::endl;
        std::cerr << "particles.: " << std::endl;
        for (uint i = 0; i < mParticleCount; ++i)
        {
            const Particle<NumericalType, Dim> &par = mParticles[i];
            for (uint d = 0; d < Dim; ++d)
            {
                std::cerr << par.x[d] << " ";
            }
            std::cerr << std::endl;
        }
        std::cerr << std::endl;
    }

    // dump particles to file
    void dump(const char *file)
    {
        std::ofstream of;
        of.open(file, std::ios::out);
        for (uint i = 0; i < mParticleCount; ++i)
        {
            const Particle<NumericalType, Dim> &par = mParticles[i];
            for (uint d = 0; d < Dim; ++d)
            {
                of << par.x[d] << " ";
            }
            of << std::endl;
        }
        of.close();
    }

protected:
    uint mParticleCount;
    Particle<NumericalType, Dim> *mParticles;
    NumericalType (*scoreFunc)(const NumericalType *, const uint, const void *);
    const void *mPayload;
    XorShift<NumericalType> mRnd;

    // Params
    NumericalType mW,  // velocity inertia weight
                  mCP, // personal best weight
                  mCG; // group best weight

    // Swarm
    NumericalType mBestPos[Dim],
                  mBestScore;
};




int main(int argc, char **argv)
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;


/*
    // create platform
    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // create context
    cl::Context context(devices);

    // create command queue
    cl::CommandQueue queue(context, devices[0]);

    // load opencl source
    std::ifstream cl_file("opencl_hello_world.cl");
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
        cl_string.length() + 1));

    // create program
    cl::Program program(context, source);

    // compile opencl source
    program.build(devices);

    // load named kernel from opencl source
    cl::Kernel kernel(program, "hello_world");

    // create a message to send to kernel
    char* message = "Hello World!";
    int messageSize = 12;

    // allocate device buffer to hold message
    cl::Buffer buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(char) * messageSize, message);

    // set message as kernel argument
    kernel.setArg(0, buffer);
    kernel.setArg(1, sizeof(int), &messageSize);

    // execute kernel
    queue.enqueueTask(kernel);

    // wait for completion
    queue.finish();

    std::cout << std::endl;


*/
    return 0;
}
