//
// Created by Andrew Astakhov on 09.04.24.
//

#ifndef CMMCORE_CMMCORE_CUTILS_H
#define CMMCORE_CMMCORE_CUTILS_H
#include "stdio.h"
#include "stdlib.h"


/*
 * Macro Usage
 * -----------
 * typedef ArrayType(double);
 * This will generate a type named `TdoubleArray` and functions :
 * `initTdoubleArray`, `insertTdoubleArray`, `freeTdoubleArray` */
#define ArrayType(type) typedef struct {type *array; size_t used;size_t size;} T##type##Array; void initT##type##Array(T##type##Array *a, size_t initialSize) {a->array = malloc(initialSize * sizeof(type));a->used = 0;a->size = initialSize; } void insertT##type##Array(T##type##Array *a, type element) {if (a->used == a->size) { a->size *= 2;a->array = realloc(a->array, a->size * sizeof(type));} a->array[a->used++] = element ;} void freeT##type##Array(T##type##Array *a){ free ( a->array ); a->array = NULL; a->used = a->size = 0; }
ArrayType(double );
ArrayType(int );
ArrayType(long );
#endif // CMMCORE_CMMCORE_CUTILS_H

