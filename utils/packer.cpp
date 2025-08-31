#include "packer.h"

void pack(const int* src, int* dst, const int m, const int n) {    
    int n_dst = (n + 8 - 1) / 8;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n_dst; j++) {
            int chunk = 0;
            for (int k = 0; k < 8; k++) {
                if (j * 8 + k < n) {
                    chunk |= (src[i * n + j * 8 + k] & 0x0F) << (k * 4);
                }
            }
            dst[i * n_dst + j] = chunk;
        }
    }
    return;
}

void unpack(const int* src, int* dst, const int m, const int n) {
    int n_src = (n + 8 - 1) / 8;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int chunk = src[i * n_src + j / 8];
            int nibble = (chunk >> ((j % 8) * 4)) & 0x0F;
            dst[i * n + j] = (nibble & 0x8) ? (nibble | ~0x0F) : nibble;
        }
    }
    return;
}