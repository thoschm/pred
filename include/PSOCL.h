

#ifndef _PSOCL_H_
#define _PSOCL_H_


/////////////////////////////////
// INCLUDES
/////////////////////////////////
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>
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
          mPartX(NULL),
          mPartV(NULL),
          mPartBest(NULL),
          mPartScore(NULL),
          mPartTmp(NULL),
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
        mPartX = new float[Dim * particleCount];
        mPartV = new float[Dim * particleCount];
        mPartBest = new float[Dim * particleCount];
        mPartScore = new float[particleCount];
        mPartTmp = new float[particleCount];

        // init opencl platform and device
        int err;
        cl_device_id device_id;
        cl_platform_id platform_id;
        size_t workGroupSize,
               kernelWSize;
        char devName[100];
        cl_uint units,
                width;
        cl_ulong constMemSize,
                 locMemSize;

        // get platform
        err = clGetPlatformIDs(1, &platform_id, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "failed to get platform id!\n";
            exit(EXIT_FAILURE);
        }

        // get device
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
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
        std::cerr << buffer;
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
        mGroups = std::ceil(1.0f * mDataNoWindowSize / mWorkSize);
        mGlobalSize = mGroups * mWorkSize;
        std::cerr << "global size....: " << mGlobalSize << std::endl;
        std::cerr << "work groups....: " << mGroups << std::endl;

        // append dummy data to fill last work group
        std::vector<float> dcopy = data;
        dcopy.resize(mGlobalSize + (Window - 1) + targetAhead, 0.0f);
        std::cerr << "inflated data..: " << data.size() << " to " << dcopy.size() << std::endl;

        // create buffers
        const uint localWindowSize = mWorkSize + (Window - 1) + targetAhead;
        std::cerr << "local window sz: " << localWindowSize << std::endl;
        if ((localWindowSize + mWorkSize) * sizeof(float) > locMemSize)
        {
            std::cerr << "WARNING: local window size exceeds local memory capacity\n";
        }

        // alloc host mem for results
        mResults = new float[particleCount * mGroups];

        float pf[3];
        pf[0] = targetValue;
        pf[1] = targetSigma;
        pf[2] = minSigma;
        uint pi[7];
        pi[0] = targetAhead;
        pi[1] = Window;
        pi[2] = Nodes;
        pi[3] = localWindowSize;
        pi[4] = mDataNoWindowSize;
        pi[5] = Dim;
        pi[6] = mGroups;
        mParamsf  = clCreateBuffer(mCtx, CL_MEM_READ_ONLY, sizeof(pf), NULL, NULL);
        mParamsi  = clCreateBuffer(mCtx, CL_MEM_READ_ONLY, sizeof(pi), NULL, NULL);
        mData     = clCreateBuffer(mCtx, CL_MEM_READ_ONLY, dcopy.size() * sizeof(float), NULL, NULL);
        mParticle = clCreateBuffer(mCtx, CL_MEM_READ_ONLY, particleCount * Dim * sizeof(float), NULL, NULL);
        mResult   = clCreateBuffer(mCtx, CL_MEM_WRITE_ONLY, particleCount * mGroups * sizeof(float), NULL, NULL);
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
        err |= clSetKernelArg(mKrnl, 6, mWorkSize * sizeof(float), NULL);
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
        delete[] mPartX;
        delete[] mPartV;
        delete[] mPartBest;
        delete[] mPartScore;
        delete[] mPartTmp;
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
            mPartScore[i] = FLT_MAX;
            mPartTmp[i] = FLT_MAX;

            // init
            for (uint d = 0; d < Dim; ++d)
            {
                // x, v, best
                const uint idx = i * Dim + d;
                mPartX[idx] = mRnd.uniform() * diff + lowerLimit;
                mPartV[idx] = mRnd.uniform() * 2.0f * diff - diff;
                mPartBest[idx] = mRnd.uniform() * diff + lowerLimit;
            }
        }

        // reset swarm
        mBestScore = FLT_MAX;
        memset(mBestPos, 0, Dim * sizeof(float));
    }

    // one optimization step
    float step()
    {
        // write particles
        int err = clEnqueueWriteBuffer(mCmd, mParticle, CL_TRUE, 0, mParticleCount * Dim * sizeof(float), mPartX, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "failed to write particles to device!\n";
            exit(EXIT_FAILURE);
        }

        // launch kernels
        for (uint i = 0; i < mParticleCount; ++i)
        {
            // set particle id
            err = clSetKernelArg(mKrnl, 7, sizeof(uint), &i);
            if (err != CL_SUCCESS)
            {
                std::cerr << "failed to set particle id!\n" << err;
                exit(EXIT_FAILURE);
            }

            // launch kernel
            err = clEnqueueNDRangeKernel(mCmd, mKrnl, 1, NULL, &mGlobalSize, &mWorkSize, 0, NULL, NULL);
            if (err)
            {
                std::cerr << "failed to execute kernel!\n";
                exit(EXIT_FAILURE);
            }
        }

        // wait finish
        clFinish(mCmd);

        // read results
        err = clEnqueueReadBuffer(mCmd, mResult, CL_TRUE, 0, mParticleCount * mGroups * sizeof(float), mResults, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "failed to read results!\n";
            exit(EXIT_FAILURE);
        }

        for (uint i = 0; i < mParticleCount; ++i)
        {
            // compute score
            float sum = 0.0f;
            for (uint r = 0; r < mGroups; ++r)
            {
                const uint idx = i * mGroups + r;
                sum += mResults[idx];
            }
            mPartTmp[i] = sum / mDataNoWindowSize; // TODO: tmp not necessary

            // update scores
            if (mPartTmp[i] < mPartScore[i])
            {
                const uint idx = Dim * i;
                mPartScore[i] = mPartTmp[i];
                memcpy(&mPartBest[idx], &mPartX[idx], Dim * sizeof(float));

                // swarm
                if (mPartTmp[i] < mBestScore)
                {
                    mBestScore = mPartTmp[i];
                    memcpy(mBestPos, &mPartX[idx], Dim * sizeof(float));
                }
            }

            // update position x
            for (uint d = 0; d < Dim; ++d)
            {
                // index
                const uint idx = i * Dim + d;
                // uniform random values
                const float rp = mRnd.uniform(),
                            rg = mRnd.uniform();
                // update velocity and pos
                mPartV[idx] = mW * mPartV[idx] +
                              mCP * rp * (mPartBest[idx] - mPartX[idx]) +
                              mCG * rg * (mBestPos[d] - mPartX[idx]);
                mPartX[idx] += mPartV[idx];
            }
        }
        return mBestScore;
    }

    // get particle pointer
    const float *getParticles()
    {
        return mPartX;
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
            for (uint d = 0; d < Dim; ++d)
            {
                const uint idx = i * Dim + d;
                std::cerr << mPartX[idx] << " ";
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
            for (uint d = 0; d < Dim; ++d)
            {
                const uint idx = i * Dim + d;
                of << mPartX[idx] << " ";
            }
            of << std::endl;
        }
        of.close();
    }

protected:
    uint mParticleCount;
    float *mPartX,
          *mPartV,
          *mPartBest,
          *mPartScore,
          *mPartTmp;
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
    uint mDataNoWindowSize,
         mGroups;
};


} // namespace

#endif
