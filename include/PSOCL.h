

#ifndef _PSOCL_H_
#define _PSOCL_H_


/////////////////////////////////
// INCLUDES
/////////////////////////////////
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <OpenCL/opencl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "XorShift.h"
#include <float.h>
#include <string.h>
#include <assert.h>


/////////////////////////////////
// NAMESPACE
/////////////////////////////////
namespace Predictor
{


/////////////////////////////////
// Particle
/////////////////////////////////
template <int Dim>
struct OCLParticle
{
    float x[Dim],
          v[Dim],
          best[Dim],
          score,
          tmp;
};


/////////////////////////////////
// PSO class
/////////////////////////////////
template <int Window, int Nodes>
class PSOCL
{
    PSOCL(const PSOCL &other);
    PSOCL &operator=(const PSOCL &other);

public:
    enum { Dim = 3 * Nodes * Window };

    // alloc particles
    PSOCL(const uint particleCount,
          const float targetValue,
          const float targetSigma,
          const uint  targetAhead,
          const float minSigma,
          const std::vector<float> &data) :
          mParticleCount(particleCount),
          mParticles(NULL),
          mW(0.0f),
          mCP(0.0f),
          mCG(0.0f),
          mBestScore(FLT_MAX)
    {
        // check used types
        assert(sizeof(float) == sizeof(cl_float));
        assert(sizeof(uint) == sizeof(cl_uint));

        // fix rnd
        mRnd.setSeed(0x12345678u);

        // reset best pos and create particles
        memset(mBestPos, 0, Dim * sizeof(float));
        mParticles = new OCLParticle<Dim>[particleCount];

        // init opencl platform and device
        int err;
        cl_device_id device_id;
        size_t workGroupSize,
               kernelWSize;
        char devName[100];
        cl_uint units,
                width;
        cl_ulong constMemSize,
                 locMemSize;

        // get device
        err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "failed to get device id!\n";
            exit(EXIT_FAILURE);
        }

        // print some info about gpu
        clGetDeviceInfo(device_id, CL_DEVICE_NAME, 100 * sizeof(char), devName, NULL);
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &workGroupSize, NULL);
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &units, NULL);
        clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &width, NULL);
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &constMemSize, NULL);
        clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &locMemSize, NULL);
        std::cerr << "selected device: " << devName << std::endl;
        std::cerr << "work group size: " << workGroupSize << std::endl;
        std::cerr << "computing units: " << units << std::endl;
        std::cerr << "float vector wi: " << width << std::endl;
        std::cerr << "max const size.: " << constMemSize << std::endl;
        if (Dim * sizeof(float) > constMemSize)
        {
            std::cerr << "WARNING: particle size exceeds constant memory capacity\n";
        }
        std::cerr << "max local size.: " << locMemSize << std::endl;

        // context
        mCtx = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
        if (!mCtx)
        {
            std::cerr << "failed to create opencl context!\n";
            exit(EXIT_FAILURE);
        }

        // Q
        mCmd = clCreateCommandQueue(mCtx, device_id, 0, &err);
        if (!mCmd)
        {
            std::cerr << "failed to create queue!\n";
            exit(EXIT_FAILURE);
        }

        // read kernel from file
        std::ifstream ifs;
        ifs.open("kernel", std::ios::in);
        std::string infile;
        infile.assign(std::istreambuf_iterator<char>(ifs),
                      std::istreambuf_iterator<char>());
        ifs.close();
        const char *cptr = infile.c_str();
        mProg = clCreateProgramWithSource(mCtx, 1, (const char **)&cptr, NULL, &err);
        if (!mProg)
        {
            std::cerr << "loading kernel failed!\n";
            exit(EXIT_FAILURE);
        }

        // build kernel
        err = clBuildProgram(mProg, 1, &device_id, NULL, NULL, NULL);
        char buffer[2048];
        clGetProgramBuildInfo(mProg, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "failed to build kernel!\n";
            exit(EXIT_FAILURE);
        }

        // create actual kernel function
        mKrnl = clCreateKernel(mProg, "pso", &err);
        if (!mKrnl || err != CL_SUCCESS)
        {
            std::cerr << "failed load kernel function!\n";
            exit(EXIT_FAILURE);
        }

        // Get the maximum work group size for executing the kernel on the device
        err = clGetKernelWorkGroupInfo(mKrnl, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernelWSize, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "failed to get kernel max worksize!\n";
            exit(EXIT_FAILURE);
        }
        assert(!(kernelWSize & (kernelWSize - 1))); // check power of 2

        // print preferred kernel work size
        std::cerr << "kernel wrk size: " << kernelWSize << std::endl;
        mWorkSize = kernelWSize;

        // compute global size
        mDataNoWindowSize = data.size() - (Window - 1) - targetAhead;
        std::cerr << "items needed...: " << mDataNoWindowSize << std::endl;
        mGlobalSize = std::ceil(1.0f * mDataNoWindowSize / mWorkSize) * mWorkSize;
        std::cerr << "global size....: " << mGlobalSize << std::endl;

        // append dummy data to fill last work group
        std::vector<float> dcopy = data;
        dcopy.resize(mGlobalSize + (Window - 1) + targetAhead, 0.0f);
        std::cerr << "inflated data..: " << data.size() << " to " << dcopy.size() << std::endl;

        // create buffers
        const uint localWindowSize = mWorkSize + (Window - 1) + targetAhead;
        std::cerr << "local window sz: " << localWindowSize << std::endl;
        if (localWindowSize * sizeof(float) > locMemSize)
        {
            std::cerr << "WARNING: local window size exceeds local memory capacity\n";
        }

        // alloc host mem for results
        mResults = new float[mDataNoWindowSize];

        float pf[3];
        pf[0] = targetValue;
        pf[1] = targetSigma;
        pf[2] = minSigma;
        uint pi[3];
        pi[0] = targetAhead;
        pi[1] = Window;
        pi[2] = Nodes;
        mParamsf  = clCreateBuffer(mCtx, CL_MEM_READ_ONLY, sizeof(pf), NULL, NULL);
        mParamsi  = clCreateBuffer(mCtx, CL_MEM_READ_ONLY, sizeof(pi), NULL, NULL);
        mData     = clCreateBuffer(mCtx, CL_MEM_READ_ONLY, dcopy.size() * sizeof(float), NULL, NULL);
        mParticle = clCreateBuffer(mCtx, CL_MEM_READ_ONLY, Dim * sizeof(float), NULL, NULL);
        mResult   = clCreateBuffer(mCtx, CL_MEM_WRITE_ONLY, mGlobalSize * sizeof(float), NULL, NULL);
        if (!mParamsf || !mParamsi || !mData || !mParticle || !mResult)
        {
            std::cerr << "failed to allocate device memory!\n";
            exit(EXIT_FAILURE);
        }

        // write constant data to device
        err = clEnqueueWriteBuffer(mCmd, mParamsf, CL_TRUE, 0, sizeof(pf), pf, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(mCmd, mParamsi, CL_TRUE, 0, sizeof(pi), pi, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "failed to write params!\n";
            exit(EXIT_FAILURE);
        }

        err = clEnqueueWriteBuffer(mCmd, mData, CL_TRUE, 0, dcopy.size() * sizeof(float), dcopy.data(), 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "failed to write input sequence to device!\n";
            exit(EXIT_FAILURE);
        }

        // set kernel arguments
        err = 0;
        err |= clSetKernelArg(mKrnl, 0, sizeof(cl_mem), &mParamsf);
        err |= clSetKernelArg(mKrnl, 1, sizeof(cl_mem), &mParamsi);
        err |= clSetKernelArg(mKrnl, 2, sizeof(cl_mem), &mData);
        err |= clSetKernelArg(mKrnl, 3, sizeof(cl_mem), &mParticle);
        err |= clSetKernelArg(mKrnl, 4, sizeof(cl_mem), &mResult);
        err |= clSetKernelArg(mKrnl, 5, localWindowSize * sizeof(float), NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "failed to set kernel arguments!\n";
            exit(EXIT_FAILURE);
        }
    }

    // release particles
    ~PSOCL()
    {
        // free ocl stuff
        clReleaseMemObject(mResult);
        clReleaseMemObject(mParticle);
        clReleaseMemObject(mData);
        clReleaseMemObject(mParamsf);
        clReleaseMemObject(mParamsi);
        clReleaseProgram(mProg);
        clReleaseKernel(mKrnl);
        clReleaseCommandQueue(mCmd);
        clReleaseContext(mCtx);

        // delete results mem
        delete[] mResults;

        // delete particles
        delete[] mParticles;
    }

    // init
    void init(const float lowerLimit,
              const float upperLimit,
              const float w  = 0.7f,
              const float cp = 1.4f,
              const float cg = 1.4f)
    {
        // params
        mW = w;
        mCP = cp;
        mCG = cg;
        const float diff = upperLimit - lowerLimit;

        // init particles
        for (uint i = 0; i < mParticleCount; ++i)
        {
            // for each particle
            OCLParticle<Dim> &par = mParticles[i];
            par.score = FLT_MAX;
            par.tmp = FLT_MAX;

            // init
            for (uint d = 0; d < Dim; ++d)
            {
                // x, v, best
                par.x[d] = mRnd.uniform() * diff + lowerLimit;
                par.v[d] = mRnd.uniform() * 2.0f * diff - diff;
                par.best[d] = mRnd.uniform() * diff + lowerLimit;
            }
        }

        // reset swarm
        mBestScore = FLT_MAX;
        memset(mBestPos, 0, Dim * sizeof(float));
    }

    // one optimization step
    float step()
    {
        for (uint i = 0; i < mParticleCount; ++i)
        {
            // for each particle
            OCLParticle<Dim> &par = mParticles[i];

            // write current particle
            int err = clEnqueueWriteBuffer(mCmd, mParticle, CL_TRUE, 0, Dim * sizeof(float), &(par.x), 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                std::cerr << "failed to write particle to device!\n";
                exit(EXIT_FAILURE);
            }

            // launch kernel
            err = clEnqueueNDRangeKernel(mCmd, mKrnl, 1, NULL, &mGlobalSize, &mWorkSize, 0, NULL, NULL);
            if (err)
            {
                std::cerr << "failed to execute kernel!\n";
                exit(EXIT_FAILURE);
            }

            // wait finish
            clFinish(mCmd);

            // read results
            err = clEnqueueReadBuffer(mCmd, mResult, CL_TRUE, 0, mDataNoWindowSize * sizeof(float), mResults, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                std::cerr << "failed to read results!\n";
                exit(EXIT_FAILURE);
            }

            // compute score
            float sum = 0.0f;
            for (uint r = 0; r < mDataNoWindowSize; ++r)
            {
                sum += mResults[r];
            }
            par.tmp = sum / mDataNoWindowSize;
        }

        for (uint i = 0; i < mParticleCount; ++i)
        {
            // for each particle
            OCLParticle<Dim> &par = mParticles[i];

            // update scores
            if (par.tmp < par.score)
            {
                par.score = par.tmp;
                memcpy(par.best, par.x, Dim * sizeof(float));

                // swarm
                if (par.tmp < mBestScore)
                {
                    mBestScore = par.tmp;
                    memcpy(mBestPos, par.x, Dim * sizeof(float));
                }
            }

            // update position x
            for (uint d = 0; d < Dim; ++d)
            {
                // uniform random values
                const float rp = mRnd.uniform(),
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
    const OCLParticle<Dim> *getParticles()
    {
        return mParticles;
    }

    // get current best
    const float *getBest()
    {
        return mBestPos;
    }

    float getScore()
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
            const OCLParticle<Dim> &par = mParticles[i];
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
            const OCLParticle<Dim> &par = mParticles[i];
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
    OCLParticle<Dim> *mParticles;
    XorShift<float> mRnd;
    float *mResults;

    // Params
    float mW,  // velocity inertia weight
          mCP, // personal best weight
          mCG; // group best weight

    // Swarm
    float mBestPos[Dim],
          mBestScore;

    // opencl stuff
    cl_context mCtx;
    cl_command_queue mCmd;
    cl_program mProg;
    cl_kernel mKrnl;
    cl_mem mParamsf,
           mParamsi,
           mData,
           mResult,
           mParticle;
    size_t mWorkSize,
           mGlobalSize;
    uint mDataNoWindowSize;
};


} // namespace

#endif
