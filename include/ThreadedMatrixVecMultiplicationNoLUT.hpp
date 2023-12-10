#pragma once

#include <cstdint>

void threaded_matVecMult_NoLUT(
    int _nF, int _n, int _nE, int _nV, int _nS, int _ndirs,
    float *_vIN, float *_vOUT,
    uint32_t *_ICf, uint32_t *_ICv, uint16_t *_ICo, float *_ICl,
    uint32_t *_ECv, uint16_t *_ECo,
    uint32_t *_ISOv,
    float *_wmrSFP, float *_wmhSFP, float *_isoSFP,
    uint32_t* _ICthreads, uint32_t* _ECthreads, uint32_t* _ISOthreads
);