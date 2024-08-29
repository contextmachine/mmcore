#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int compare(const void *a, const void *b) {
    double diff = *(double*)a - *(double*)b;
    if (diff > 0) return 1;
    else if (diff < 0) return -1;
    else return 0;
}

double* uniqueSortedEps(double* array, int size, double eps, int* new_size) {
    if (size == 0) {
        *new_size = 0;
        return NULL;
    }

    // Sort the array
    qsort(array, size, sizeof(double), compare);

    // Allocate memory for the output array, at most it will be of the same size as the input array
    double* result = (double*)malloc(size * sizeof(double));
    if (!result) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE);
    }

    // Initialize the unique array with the first element
    result[0] = array[0];
    int unique_count = 1;

    // Iterate through the sorted array and add unique elements to the result
    for (int i = 1; i < size; ++i) {
        if (fabs(array[i] - array[i-1]) > eps) {
            result[unique_count++] = array[i];
        }
    }

    // Resize the result array to the actual number of unique elements
    result = (double*)realloc(result, unique_count * sizeof(double));
    if (!result) {
        perror("Failed to reallocate memory");
        exit(EXIT_FAILURE);
    }

    // Set the new size of the array
    *new_size = unique_count;
    return result;
}
int compare2(const void *a, const void *b) {
    double diff = (*(double*)a - *(double*)b);
    if (diff > 0) return 1;
    else if (diff < 0) return -1;
    else return 0;
}

// Function to sort and remove duplicates
double* uniqueSorted(double *arr, int n, int *new_len) {
    // If the array is empty or has one element, return it as is
    if (n == 0) {
        *new_len = 0;
        return NULL;
    }
    if (n == 1) {
        *new_len = 1;
        double *result = (double*)malloc(sizeof(double));
        result[0] = arr[0];
        return result;
    }

    // Sort the array
    qsort(arr, n, sizeof(double), compare2);

    // Allocate memory for the unique sorted array
    double *unique_arr = (double*)malloc(n * sizeof(double));
    int j = 0;

    // Add the first element
    unique_arr[j++] = arr[0];

    // Iterate over the sorted array and add only unique elements
    for (int i = 1; i < n; i++) {
        if (arr[i] != arr[i-1]) {
            unique_arr[j++] = arr[i];
        }
    }

    // Resize the memory to match the number of unique elements
    unique_arr = (double*)realloc(unique_arr, j * sizeof(double));
    *new_len = j;

    return unique_arr;
}


int compare3(const void *a, const void *b) {
    int diff = (*(int*)a - *(int*)b);
    if (diff > 0) return 1;
    else if (diff < 0) return -1;
    else return 0;
}

// Function to sort and remove duplicates
int* uniqueSortedInt(int *arr, int n, int *new_len) {
    // If the array is empty or has one element, return it as is
    if (n == 0) {
        *new_len = 0;
        return NULL;
    }
    if (n == 1) {
        *new_len = 1;
        int *result = (int*)malloc(sizeof(int));
        result[0] = arr[0];
        return result;
    }

    // Sort the array
    qsort(arr, n, sizeof(int), compare3);

    // Allocate memory for the unique sorted array
    int *unique_arr = (int*)malloc(n * sizeof(int));
    int j = 0;

    // Add the first element
    unique_arr[j++] = arr[0];

    // Iterate over the sorted array and add only unique elements
    for (int i = 1; i < n; i++) {
        if (arr[i] != arr[i-1]) {
            unique_arr[j++] = arr[i];
        }
    }

    // Resize the memory to match the number of unique elements
    unique_arr = (int*)realloc(unique_arr, j * sizeof(int));
    *new_len = j;

    return unique_arr;
}

void uniqueSortedIntEmplace(int *arr, int n, int *new_len,int *unique_arr ) {
    // If the array is empty or has one element, return it as is
    if (n == 0) {
        *new_len = 0;
        return;
    }
    if (n == 1) {
        *new_len = 1;
        int *result = (int*)malloc(sizeof(int));
        result[0] = arr[0];
        return ;
    }

    // Sort the array
    qsort(arr, n, sizeof(int), compare3);

    // Allocate memory for the unique sorted array
    
    int j = 0;

    // Add the first element
    unique_arr[j++] = arr[0];

    // Iterate over the sorted array and add only unique elements
    for (int i = 1; i < n; i++) {
        if (arr[i] != arr[i-1]) {
            unique_arr[j++] = arr[i];
        }
    }

    // Resize the memory to match the number of unique elements
    unique_arr = (int*)realloc(unique_arr, j * sizeof(int));
    *new_len = j;

   
}
