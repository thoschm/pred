
#ifndef _XOR_SHIFT_H_
#define _XOR_SHIFT_H_


/////////////////////////////////
// INCLUDE
/////////////////////////////////
#include <time.h>
#include <stdint.h>
#include <vector>
#include <cmath>


/////////////////////////////////
// TYPEDEFS
/////////////////////////////////
typedef unsigned int uint;


/////////////////////////////////
// NAMESPACE
/////////////////////////////////
namespace Predictor
{


///////////////////////////////
// simple random number gen.
///////////////////////////////
template <typename NumericalType>
class XorShift
{
private:
    XorShift(const XorShift &other);
    XorShift &operator=(const XorShift &other);

public:
    XorShift(uint64_t seed)
    {
        setSeed(seed);
    }
    XorShift()
    {
        setSeed(time(NULL));
    }
    void setSeed(uint64_t seed)
    {
        mSeed = (seed == 0) ? 0xdeadbeefu : seed;
    }
    uint64_t rand()
    {
        mSeed ^= mSeed << 13;
        mSeed ^= mSeed >> 7;
        mSeed ^= mSeed << 17;
        return mSeed;
    }
    NumericalType uniform()
    {
        return (rand() % 10000000u) * (NumericalType)0.0000001;
    }
private:
    uint64_t mSeed;
};


///////////////////////////////
// normal distribution gen
///////////////////////////////
template <typename NumericalType>
class NormalDistGenerator
{
private:
    NormalDistGenerator(const NormalDistGenerator &other);
    NormalDistGenerator &operator=(const NormalDistGenerator &other);

public:
    NormalDistGenerator(NumericalType mu    = (NumericalType)0.0,
                        NumericalType sigma = (NumericalType)1.0,
                        uint loops = 12u) : mMu(mu),
                                            mSigma(sigma),
                                            mLoops(loops),
                                            mScale(std::sqrt((NumericalType)12.0 / loops))
    { }

    void setMu(NumericalType val)
    {
        mMu = val;
    }

    void setSigma(NumericalType val)
    {
        mSigma = val;
    }

    void setLoops(uint val)
    {
        mLoops = val;
        mScale = std::sqrt((NumericalType)12.0 / val);
    }

    void setSeed(uint64_t seed)
    {
        mRnd.setSeed(seed);
    }

    NumericalType rand()
    {
        NumericalType sum = (NumericalType)0.0;
        for (uint i = 0; i < mLoops; ++i)
        {
            sum += mRnd.uniform();
        }
        return (sum - (NumericalType)0.5 * mLoops) * mScale * mSigma + mMu;
    }

private:
    XorShift<NumericalType> mRnd;
    NumericalType mMu, mSigma;
    uint mLoops;
    NumericalType mScale;
};


///////////////////////////////
// simple look up table
///////////////////////////////
template <typename Type>
class CircularLUT
{
private:
    CircularLUT(const CircularLUT &other);
    CircularLUT &operator=(const CircularLUT &other);

public:
    CircularLUT(uint cnt = 0) : mPos(0)
    {
        reserve(cnt);
    }

    void reserve(uint cnt)
    {
        mBuffer.reserve(cnt);
    }

    void clear()
    {
        mBuffer.clear();
    }

    void insert(const Type &value)
    {
        mBuffer.push_back(value);
    }

    Type next()
    {
        if (mPos == mBuffer.size())
        {
            mPos = 0;
        }
        return mBuffer[mPos++];
    }

protected:
    std::vector<Type> mBuffer;
    uint mPos;
};

}

#endif
