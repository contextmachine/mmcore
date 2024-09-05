#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef _MATRIX_H
#define _MATRIX_H
// Structure definition for the matrix
typedef struct {
    double *data; // Pointer to hold the matrix elements
    int rows;     // Number of rows
    int columns;  // Number of columns
} matrix;

// Macro to access matrix elements (row-major order)
#define MAT(m, i, j) (m.data[(i) * (m.columns) + (j)])
#define MAT_PTR(m, i, j) (m->data[(i) * (m->columns) + (j)])

inline double get_item(matrix *m, int i, int j) {
    return MAT_PTR(m, i, j);
}
inline void set_item(matrix *m, int i, int j, double val) {
    m->data[(i) * (m->columns) + (j)]=val;

}
// Function to set matrix values from a one-dimensional array
void set_matrix_from_array(matrix *m, double *arr) {
    int total_size = m->rows * m->columns;
    for (int i = 0; i < total_size; i++) {
        m->data[i] = arr[i]; // Copy element from 1D array to matrix
    }
}

// Function to set matrix values from a two-dimensional array
#ifndef _WIN32
// For non-Windows platforms, use VLAs
void set_matrix_from_array2d(matrix *m, double arr[m->rows][m->columns]) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->columns; j++) {
            MAT_PTR(m, i, j) = arr[i][j]; // Copy element from 2D array to matrix
        }
    }
}
#else
// For Windows, use dynamically allocated 2D array (double **)
void set_matrix_from_array2d(matrix *m, double **arr) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->columns; j++) {
            MAT_PTR(m, i, j) = arr[i][j]; // Copy element from 2D array to matrix
        }
    }
}
#endif //_WIN32

// Function to create and initialize a matrix
matrix create_matrix(int rows, int columns) {
    matrix m;
    m.rows = rows;
    m.columns = columns;
    m.data = (double *)malloc(rows * columns * sizeof(double));
    return m;
}
// Function to release the memory allocated for the matrix
void free_matrix(matrix *m) {
    if (m->data != NULL) {
        free(m->data);  // Free the dynamically allocated memory for the matrix data
        m->data = NULL; // Set the pointer to NULL to avoid dangling pointers
    }
}
// Function to print a matrix
void print_matrix(matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->columns; j++) {
            printf("%8.4f ", MAT_PTR(m, i, j));
        }
        printf("\n");
    }
}



// Function to perform LU decomposition
int LU_decomposition(matrix *A, int *P) {
    int i, j, k;
    for (i = 0; i < A->rows; i++) {
        P[i] = i; // Initialize permutation matrix
    }

    for (i = 0; i < A->rows; i++) {
        double maxA = 0.0;
        int imax = i;

        // Find pivot
        for (k = i; k < A->rows; k++) {
            if (fabs(MAT_PTR(A, k, i)) > maxA) {
                maxA = fabs(MAT_PTR(A, k, i));
                imax = k;
            }
        }

        // Pivot if necessary
        if (imax != i) {
            // Swap rows in P
            int tmp = P[i];
            P[i] = P[imax];
            P[imax] = tmp;

            // Swap rows in A
            for (k = 0; k < A->columns; k++) {
                double tmpVal = MAT_PTR(A, i, k);
                MAT_PTR(A, i, k) = MAT_PTR(A, imax, k);
                MAT_PTR(A, imax, k) = tmpVal;
            }
        }

        // Perform elimination
        for (j = i + 1; j < A->rows; j++) {
            MAT_PTR(A, j, i) /= MAT_PTR(A, i, i);

            for (k = i + 1; k < A->columns; k++) {
                MAT_PTR(A, j, k) -= MAT_PTR(A, j, i) * MAT_PTR(A, i, k);
            }
        }
    }

    return 1; // LU decomposition succeeded
}

// Function to solve Ax = b using LU decomposition
void LU_solve(matrix *A, int *P, double *b, double *x) {
    #ifdef _WIN32
    // Use malloc for Windows
    double *y = (double *)malloc(A->rows * sizeof(double));
    if (y == NULL) {
        // Handle allocation failure
        return;
    }
    #else
    // Use VLAs for non-Windows
    double y[A->rows];
    #endif

    int i, j;

    // Forward substitution to solve Ly = Pb
    for (i = 0; i < A->rows; i++) {
        y[i] = b[P[i]];

        for (j = 0; j < i; j++) {
            y[i] -= MAT_PTR(A, i, j) * y[j];
        }
    }

    // Backward substitution to solve Ux = y
    for (i = A->rows - 1; i >= 0; i--) {
        x[i] = y[i];

        for (j = i + 1; j < A->columns; j++) {
            x[i] -= MAT_PTR(A, i, j) * x[j];
        }

        x[i] /= MAT_PTR(A, i, i);
    }
    #ifdef _WIN32
    // Free dynamically allocated memory on Windows
    free(y);
    #endif
}

// Function to calculate the inverse of a matrix
int invert_matrix(matrix *A, matrix *inverse) {
    #ifdef _WIN32
    double *b = (double *)malloc(A->rows * sizeof(double));
    double *x = (double *)malloc(A->rows * sizeof(double));
    int *P = (int *)malloc(A->rows * sizeof(int));
    if (b == NULL || x == NULL || P == NULL) {
        // Handle allocation failure
        return 0;
    }
    #else
    double b[A->rows], x[A->rows];
    int P[A->rows];
    #endif
    int i, j;

    // Perform LU decomposition
    if (!LU_decomposition(A, P)) {
        return 0; // Matrix is singular
    }

    // Solve Ax = b for each column of the identity matrix
    for (i = 0; i < A->columns; i++) {
        // Set b to be the ith column of the identity matrix
        for (j = 0; j < A->rows; j++) {
            b[j] = (i == j) ? 1.0 : 0.0;
        }

        // Solve the system using LU decomposition
        LU_solve(A, P, b, x);

        // Store the result as the ith column of the inverse matrix
        for (j = 0; j < A->rows; j++) {
            MAT_PTR(inverse, j, i) = x[j];
        }
    }
    #ifdef _WIN32
    free(b);
    free(x);
    free(P);
    #endif

    return 1;
}





#endif //_MATRIX_H