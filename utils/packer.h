#pragma once

#include <stdint.h>

void pack(const int* src, int* dst, const int m, const int n);
void unpack(const int* src, int* dst, const int m, const int n);