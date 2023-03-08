#include "matrixVecMultiplication.hpp"

void sequential_matVecMult(
    int _nF, int _n, int _nE, int _nV, int _nS, int _ndirs,int _nI,int _nR, int _nT,int N, int M,
    float *_vIN, float *_vOUT,
    uint32_t *_ICf, uint32_t *_ICv, uint16_t *_ICo, float *_ICl,
    uint32_t *_ECv,uint16_t *_ECo,
    uint32_t *_ISOv,
    float *_wmrSFP, float *_wmhSFP,float *_isoSFP
)
{

    float accumulator;
    int xIndex;

    /* IC */
    for(int segment = 0 ; segment < _n ; segment++)
    {
        for(int sample = 0; sample < _nS; sample++)
        {
            accumulator = 0.0f;
            for(int radii = 0 ; radii <  _nR; radii++ )
            {
                accumulator += _vIN[_ICf[segment] + radii] * _wmrSFP[ (radii*_ndirs*_nS) + (_ICo[segment] * _nS + sample)];
            }
            _vOUT[_ICv[segment]*_nS + sample] += _ICl[segment]* accumulator;
        }
    }

    /* EC */
    xIndex = _nR * _nF;
    for(int segment = 0; segment < _nE; segment++)
    {
        for(int sample = 0; sample < _nS ; sample++)
        {
            accumulator = 0.0f;
            for(int ec = 0; ec < _nT ; ec++)
            {
                accumulator += _vIN[xIndex + ec * _nE] * _wmhSFP[(ec*_ndirs*_nS) + (_ECo[segment] * _nS + sample)];
            }
            _vOUT[_ECv[segment]*_nS + sample] += accumulator;
        }
        xIndex++;
    }

    /* ISO */
    xIndex = _nR*_nF + _nT*_nE;
    for(int i = 0; i < _nV ; i++)
    {
        for(int sample = 0 ; sample < _nS ; sample++)
        {
            accumulator = 0.0f;
            for(int iso = 0; iso < _nI ; iso++)
            {
                accumulator += _vIN[xIndex + iso*_nV] * _isoSFP[iso*_nS + sample];
            }
            _vOUT[_ISOv[i]*_nS + sample] += accumulator;
        }
        xIndex++;
    }
}