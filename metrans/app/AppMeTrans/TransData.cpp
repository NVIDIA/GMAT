#include "TransData.h"
#include "TransDataConverter.h"

void TransData::Free() {
    if (dpVideoFrame && pDec) {
        pDec->UnlockFrame(&dpVideoFrame, 1);
    } else if (dpVideoFrame && pConverter) {
        pConverter->Recycle(dpVideoFrame);
    }
}
