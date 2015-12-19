

#include <fstream>
#include <iostream>

typedef unsigned int uint;

int main(int argc, char **argv)
{
    const uint interval = 300u; // seconds
    uint nexttime = 0;
    float lastPrice = 0.0;
    std::ifstream ifs;
    ifs.open("price.txt", std::ios::in);
    std::ofstream ofs;
    ofs.open("chart.txt", std::ios::out);
    for ( ; ; )
    {
        uint stamp;
        float price, vol;
        ifs >> stamp
            >> price
            >> vol;
        if (ifs.eof())
        {
            break;
        }       
        
        if (stamp >= nexttime)
        {
            nexttime = stamp + interval;
            ofs << lastPrice << std::endl;
            std::cerr << stamp << " " << lastPrice << std::endl;
        }
        lastPrice = price;        
    }
    ifs.close();
    ofs.close();
    return 0;
}
