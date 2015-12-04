

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>
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
class PSOCL
{
    PSOCL(const PSOCL &other);
    PSOCL &operator=(const PSOCL &other);

public:
    // alloc particles
    PSOCL(const uint particleCount,
          const NumericalType targetValue,
          const NumericalType targetSigma,
          const NumericalType targetAhead,
          const NumericalType minSigma,
          const NumericalType *data,
          const uint dataSize) :
          mParticleCount(particleCount),
          mParticles(NULL),
          mW((NumericalType)0.0),
          mCP((NumericalType)0.0),
          mCG((NumericalType)0.0),
          mBestScore((NumericalType)FLT_MAX)
    {
        // reset best pos and create particles
        memset(mBestPos, 0, Dim * sizeof(NumericalType));
        mParticles = new Particle<NumericalType, Dim>[particleCount];

        // init opencl platform and device
        int err;
        cl_device_id device_id;
        cl_context context;
        cl_command_queue commands;
        cl_program program;
        cl_kernel kernel;
        cl_mem params, dbuf, part;
        cl_mem output;

        err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "failed to get device id!\n";
            exit(EXIT_FAILURE);
        }

        context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
        if (!context)
        {
            std::cerr << "failed to create opencl context!\n";
            exit(EXIT_FAILURE);
        }

        commands = clCreateCommandQueue(context, device_id, 0, &err);
        if (!commands)
        {
            std::cerr << "failed to create queue!\n";
            exit(EXIT_FAILURE);
        }

        std::ifstream cl_file("pso.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), std::istreambuf_iterator<char>());
        cl_file.close();
        program = clCreateProgramWithSource(context, 1, (const char **) &(cl_string.c_str()), NULL, &err);
        if (!program)
        {
            std::cerr << "loading kernel failed!\n";
            exit(EXIT_FAILURE);
        }

        err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            size_t len;
            char buffer[2048];
            std::cerr << "failed to build kernel!\n";
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            std::cerr << buffer << std::endl;
            exit(EXIT_FAILURE);
        }

        kernel = clCreateKernel(program, "pso", &err);
        if (!kernel || err != CL_SUCCESS)
        {
            std::cerr << "failed load kernel function!\n";
            exit(EXIT_FAILURE);
        }

        NumericalType p[5];
        p[0] = targetValue;
        p[1] = targetSigma;
        p[2] = targetAhead;
        p[3] = minSigma;
        p[4] = dataSize;
        params = clCreateBuffer(context, CL_MEM_READ_ONLY, 5u * sizeof(NumericalType), NULL, NULL);
        dbuf   = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize * sizeof(NumericalType), NULL, NULL);


//part   = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize * sizeof(NumericalType), NULL, NULL);
        if (!params || !dbuf)
        {
            std::cerr << "failed to allocate device memory!\n";
            exit(EXIT_FAILURE);
        }

        err = clEnqueueWriteBuffer(commands, params, CL_TRUE, 0, 5u * sizeof(NumericalType), p, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "failed write params!\n";
            exit(EXIT_FAILURE);
        }

        err = clEnqueueWriteBuffer(commands, dbuf, CL_TRUE, 0, dataSize * sizeof(NumericalType), data, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            std::cerr << "failed write input sequence to device!\n";
            exit(EXIT_FAILURE);
        }
/*

        err = 0;

        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);

        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);

        err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);

        if (err != CL_SUCCESS)

        {

            printf("Error: Failed to set kernel arguments! %d\n", err);

            exit(1);

        }



        // Get the maximum work group size for executing the kernel on the device

        //

        err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);

        if (err != CL_SUCCESS)

        {

            printf("Error: Failed to retrieve kernel work group info! %d\n", err);

            exit(1);

        }



        // Execute the kernel over the entire range of our 1d input data set

        // using the maximum number of work group items for this device

        //

        global = count;

        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);

        if (err)

        {

            printf("Error: Failed to execute kernel!\n");

            return EXIT_FAILURE;

        }



        // Wait for the command commands to get serviced before reading back results

        //

        clFinish(commands);



        // Read back the results from the device to verify the output

        //

        err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );

        if (err != CL_SUCCESS)

        {

            printf("Error: Failed to read output array! %d\n", err);

            exit(1);

        }



        // Validate our results

        //

        correct = 0;

        for(i = 0; i < count; i++)

        {

            if(results[i] == data[i] * data[i])

                correct++;

        }



        // Print a brief summary detailing the results

        //

        printf("Computed '%d/%d' correct values!\n", correct, count);



        // Shutdown and cleanup

        //

        clReleaseMemObject(input);

        clReleaseMemObject(output);

        clReleaseProgram(program);

        clReleaseKernel(kernel);

        clReleaseCommandQueue(commands);

        clReleaseContext(context);

*/
    }

    // release particles
    ~PSOCL()
    {


        // delete particles
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
            //par.tmp = scoreFunc(par.x, Dim, mPayload);
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
    XorShift<NumericalType> mRnd;

    // Params
    NumericalType mW,  // velocity inertia weight
                  mCP, // personal best weight
                  mCG; // group best weight

    // Swarm
    NumericalType mBestPos[Dim],
                  mBestScore;

    // opencl stuff

};




int main(int argc, char **argv)
{
   /* std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;


    try
    {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("kernel.cl");
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
        const uint size = 10000u;
        uint a = 1u,
             b = 2u;
        float c[size];

        // allocate device buffer to hold message
        cl::Buffer buffer1(context, CL_MEM_READ_ONLY, sizeof(uint));
        cl::Buffer buffer2(context, CL_MEM_READ_ONLY, sizeof(uint));
        cl::Buffer buffer3(context, CL_MEM_WRITE_ONLY, size * sizeof(float));

        queue.enqueueWriteBuffer(buffer1, CL_TRUE, 0, sizeof(uint), &a);
        queue.enqueueWriteBuffer(buffer2, CL_TRUE, 0, sizeof(uint), &b);

        // set message as kernel argument
        kernel.setArg(0, buffer1);
        kernel.setArg(1, buffer2);
        kernel.setArg(2, buffer3);

        // execute kernel
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(size));

        queue.enqueueReadBuffer(buffer3, CL_TRUE, 0, size * sizeof(float), &c);

        // wait for completion
        queue.finish();

        std::cout << c[0] << std::endl;
    }
    catch (cl::Error er)
    {
        printf("ERROR: %s(%d)\n", er.what(), er.err());
    }
*/

    return 0;
}
