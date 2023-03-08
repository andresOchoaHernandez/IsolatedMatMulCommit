#include "matrixVecMultiplication.hpp"

CommitOriginalDataStructure::CommitOriginalDataStructure(int nF, int n, int nE, int nV, int nS, int ndirs,int nI,int nR, int nT,int threads):
_nF{nF},
_n{n},
_nE{nE},
_nV{nV},
_nS{nS},
_ndirs{ndirs},
_nI{nI},
_nR{nR},
_nT{nT},
M{_nV*_nS},
N{_nR*_nF + _nT*_nE + _nI*_nV},
icf(_n),
icv(_n),
ico(_n),
icl(_n),
ecv(_nE),
eco(_nE),
isov(_nV),
wmrSFP(_nR*_ndirs*_nS),
wmhSFP(_nT*_ndirs*_nS),
isoSFP(_nI*_nS),
_threads(threads),
icThreads(_threads + 1),
ecThreads(_threads + 1),
isoThreads(_threads + 1)
{}

void CommitOriginalDataStructure::loadDataset(std::string& inputPath,std::string& outputPath){}