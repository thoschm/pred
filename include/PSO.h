
#ifndef _PSO_H_
#define _PSO_H_


/////////////////////////////////
// INCLUDE
/////////////////////////////////
#include "XorShift.h"
#include <float.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <pthread.h>


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
// WORKER POOL
/////////////////////////////////
// TODO: prevent fast threads from waiting for slow threads, make a queue and let threads take chunks
class WorkerPool
{
    WorkerPool(const WorkerPool &other);
    WorkerPool &operator=(const WorkerPool &other);
public:
    struct Context
    {
        pthread_t tid;
        uint id;
        bool *alive;
        void (*fptr)(const uint, void *);
        pthread_barrier_t *barrier1,
                          *barrier2;
        void *payload;
    };

    WorkerPool() : mWorkers(0), mpl(NULL), mAlive(true)
    { }

    void create(uint workers, void (*extfunc)(const uint, void *), void *payload)
    {
        mWorkers = workers;
        // create barriers
        pthread_barrier_init(&mFirst, NULL, workers + 1u);
        pthread_barrier_init(&mSecond, NULL, workers + 1u);
        // create threads
        mpl = new Context[mWorkers];
        for (uint i = 0; i < mWorkers; ++i)
        {
            mpl[i].id = i;
            mpl[i].alive = &mAlive;
            mpl[i].fptr = extfunc;
            mpl[i].barrier1 = &mFirst;
            mpl[i].barrier2 = &mSecond;
            mpl[i].payload = payload;
            pthread_create(&mpl[i].tid, NULL, func, (void *)&mpl[i]);
        }
    }

    void startJob()
    {
        pthread_barrier_wait(&mFirst);
    }

    void waitDone()
    {
        pthread_barrier_wait(&mSecond);
    }

    ~WorkerPool()
    {
        // destroy threads
        mAlive = false;
        startJob();
        for (uint i = 0; i < mWorkers; ++i)
        {
            pthread_join(mpl[i].tid, NULL);
        }
        delete[] mpl;
        // destroy barriers
        pthread_barrier_destroy(&mFirst);
        pthread_barrier_destroy(&mSecond);
    }

private:
    static void *func(void *ptr)
    {
        Context *context = (Context *)ptr;
        for ( ; ; )
        {
            // barrier 1
            pthread_barrier_wait(context->barrier1);
            // check if alive
            if (*(context->alive) == false)
            {
                break;
            }
            // do work
            context->fptr(context->id, context->payload);
            // barrier 2
            pthread_barrier_wait(context->barrier2);
        }
        std::cerr << "thread " << context->id  << " terminated" << std::endl;
        return NULL;
    }

    uint mWorkers;
    Context *mpl;
    pthread_barrier_t mFirst,
                      mSecond;
    bool mAlive;
};


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
//#define PSO_MT
#define PSO_OMP
//#define THREADS 50u

template <typename NumericalType, int Dim>
class PSO
{
    PSO(const PSO &other);
    PSO &operator=(const PSO &other);

public:

#ifdef PSO_MT
    struct WorkerArgs
    {
        uint dim,
             range,
             partCnt,
             threads;
        NumericalType (*function)(const NumericalType *, const uint, const void *);
        const void *payload;
        Particle<NumericalType, Dim> *particles;
    };
#endif

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

#ifdef PSO_MT
        mArgs.dim = Dim;
        mArgs.range = particleCount / THREADS;
        mArgs.partCnt = particleCount;
        mArgs.threads = THREADS;
        mArgs.particles = mParticles;
        mArgs.function = scoreFunction;
        mArgs.payload = payload;
        mPool.create(THREADS, worker, (void *)&mArgs);
#endif
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
#ifdef PSO_MT
        mPool.startJob();
        mPool.waitDone();
#else
#ifdef PSO_OMP
#pragma omp parallel for
#endif
        for (uint i = 0; i < mParticleCount; ++i)
        {
            // for each particle
            Particle<NumericalType, Dim> &par = mParticles[i];

            // compute cost
            par.tmp = scoreFunc(par.x, Dim, mPayload);
        }
#endif

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

#ifdef PSO_MT
    // called by thread
    static void worker(const uint id, void *ptr)
    {
        WorkerArgs *args = (WorkerArgs *)ptr;

        // compute limits
        uint low = id * args->range,
             high = (id == args->threads - 1u) ? (args->partCnt) : ((id + 1u) * args->range);

        // process range
        for (uint i = low; i < high; ++i)
        {
            Particle<NumericalType, Dim> &par = args->particles[i];
            par.tmp = args->function(par.x, args->dim, args->payload);
        }
    }
#endif

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

#ifdef PSO_MT
    WorkerPool mPool;
    WorkerArgs mArgs;
#endif
};

}

#endif
