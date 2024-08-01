#include <stdio.h>

// Function to swap two elements
void swap(double* a, double* b) {
    double t = *a;
    *a = *b;
    *b = t;
}

// Partition function
int partition(double* arr, int low, int high) {
    double pivot = arr[high]; // pivot
    int i = (low - 1); // Index of smaller element

    for ( int j = low; j <= high - 1; j++) {
        // If current element is smaller than the pivot
        if (arr[j] < pivot) {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

// QuickSort function
void quickSort(double* arr, int low, int high) {
    if (low < high) {
        // pi is partitioning index, arr[p] is now at right place
         int pi = partition(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

void swap_int(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int partitionArgSort(double* arr, int* indices, int low, int high) {
    double pivot = arr[indices[high]]; // pivot
    int i = (low - 1); // Index of smaller element

    for (int j = low; j <= high - 1; j++) {
        // If current element is smaller than or equal to the pivot
        if (arr[indices[j]] <= pivot ) {
           i++; // increment index of smaller element
            swap_int(&indices[i], &indices[j]);

        }
    }
    swap_int(&indices[i + 1], &indices[high]);
    return (i + 1);
}

void argSort(double* arr, int* indices, int low, int high) {
    if (low < high) {
        // pi is partitioning index, arr[pi] is now at right place
        int pi = partitionArgSort(arr, indices, low, high);

        // Separately sort elements before
        // partition and after partition
        argSort(arr, indices, low, pi - 1);
        argSort(arr, indices, pi + 1, high);
    }
}

/*
void printArray(double arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

int main() {
    double arr[] = {10, 7, 8, 9, 1, 5};
    int n = (int) sizeof(arr) / sizeof(arr[0]);
    printf("Unsorted array: \n");
    printArray(arr, n);

    quickSort(arr, 0, n - 1);

    printf("Sorted array: \n");
    printArray(arr, n);
    return 0;
}
*/