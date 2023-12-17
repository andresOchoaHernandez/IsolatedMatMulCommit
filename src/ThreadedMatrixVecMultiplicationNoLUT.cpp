#include <pthread.h>

#include "ThreadedMatrixVecMultiplicationNoLUT.hpp"

#ifndef nTHREADS
    #define nTHREADS 1
#endif

#ifndef nIC
    #define nIC 1
#endif

#ifndef nEC
    #define nEC 0
#endif

#ifndef nISO
    #define nISO 0
#endif

/* global variables */
int         nF, n;
float      *x, *Y;
uint32_t    *ICthreads, *ISOthreads;
uint8_t     *ICthreadsT;
uint32_t    *ISOthreadsT;
uint32_t    *ICf, *ICv, *ISOv;
float       *ICl;


// ====================================================
// Compute a sub-block of the A*x MAtRIX-VECTOR product
// ====================================================
void* COMMIT_A__block( void *ptr )
{
    int      id = (long)ptr;
    float   x0;
    float   *xPtr;
    uint32_t *t_v, *t_vEnd, *t_f;
    float    *t_l;

    // intra-cellular compartments
    t_v    = ICv + ICthreads[id];
    t_vEnd = ICv + ICthreads[id+1];
    t_l    = ICl + ICthreads[id];
    t_f    = ICf + ICthreads[id];

    while( t_v != t_vEnd )
    {
        x0 = x[*t_f];
        if ( x0 != 0 )
            Y[*t_v] += (float)(*t_l) * x0;
        t_f++;
        t_v++;
        t_l++;
    }

#if nISO>=1
    // isotropic compartments
    t_v    = ISOv + ISOthreads[id];
    t_vEnd = ISOv + ISOthreads[id+1];
    xPtr   = x + nF + ISOthreads[id];

    while( t_v != t_vEnd )
    {
        x0 = *xPtr++;
        if ( x0 != 0 )
            Y[*t_v] += x0;
        t_v++;
    }
#endif

    pthread_exit( 0 );
}

void threaded_matVecMult_NoLUT(
    int _nF, int _n, int _nE, int _nV, int _nS, int _ndirs,
    float *_vIN, float *_vOUT,
    uint32_t *_ICf, uint32_t *_ICv, uint16_t *_ICo, float *_ICl,
    uint32_t *_ECv, uint16_t *_ECo,
    uint32_t *_ISOv,
    float *_wmrSFP, float *_wmhSFP, float *_isoSFP,
    uint32_t* _ICthreads, uint32_t* _ECthreads, uint32_t* _ISOthreads
)
{
    nF = _nF;
    n  = _n;

    x = _vIN;
    Y = _vOUT;

    ICf  = _ICf;
    ICv  = _ICv;
    ICl  = _ICl;
    ISOv = _ISOv;

    ICthreads  = _ICthreads;
    ISOthreads = _ISOthreads;

    // Run SEPARATE THREADS to perform the multiplication
    pthread_t threads[nTHREADS];
    int t;
    for(t=0; t<nTHREADS ; t++)
        pthread_create( &threads[t], NULL, COMMIT_A__block, (void *) (long int)t );
    for(t=0; t<nTHREADS ; t++)
        pthread_join( threads[t], NULL );
    return;
}