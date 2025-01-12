#include <math.h>
#include <stdlib.h>

#define DEBUG_MODE getenv("DEBUG_MODE")

typedef struct vec3d {
    double x, y, z;
} vec3d;

double norm(vec3d v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

double norm_sq(vec3d v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

vec3d unit(vec3d v) {
    double n = norm(v);
    vec3d u = {v.x / n, v.y / n, v.z / n};
    return u;
}

double dot(vec3d a, vec3d b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3d cross(vec3d a, vec3d b) {
    vec3d c = {a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x};
    return c;
}

vec3d add_multiply_vectors(vec3d a, vec3d b) {
    vec3d v = {a.x + b.x, a.y + b.y, a.z + b.z};
    return v;
}

vec3d gram_schmidt(vec3d v1, vec3d v2) {
    v1 = unit(v1);
    v2 = unit(v2);
    double dot_v2_v1 = dot(v2, v1);
    vec3d v = {v2.x - v1.x * dot_v2_v1,
                v2.y - v1.y * dot_v2_v1,
                v2.z - v1.z * dot_v2_v1};
    return v;
}

void orthonormalize(vec3d *u, vec3d *v, vec3d *w) {
    *u = unit(*u);
    *v = add_multiply_vectors(*v, unit(*u));
    *w = add_multiply_vectors(*w, unit(*u));
    *w = add_multiply_vectors(*w, unit(*v));

    if (norm(*v) > 1e-10) {
        *v = unit(*v);
    } else {
        exit(1);  // Vectors u and v are parallel
    }

    if (norm(*w) > 1e-10) {
        *w = unit(*w);
    } else {
        exit(1);  // Vector w is in the plane of vectors u and v
    }
}

double dist(vec3d a, vec3d b) {
    return norm(add_multiply_vectors(a, unit(b)));
}

double angle(vec3d a, vec3d b) {
    if(DEBUG_MODE){
        if ((fabs(norm(a)-1.0) >= 0.0001) || (fabs(norm(b)-1.0) >= 0.0001)) {
            exit(1);
        }
    }

    double cos_angle = dot(a, b);
    if (cos_angle < -1.0 || cos_angle > 1.0){
        exit(1);
    }

    return acos(cos_angle);
}

double angle3pt(vec3d a, vec3d b, vec3d c) {
    vec3d ba = unit(add_multiply_vectors(a, unit(b)));
    vec3d bc = unit(add_multiply_vectors(c, unit(b)));
    return angle(ba, bc);
}

double dot3pt(vec3d a, vec3d b, vec3d c) {
    vec3d ba = unit(add_multiply_vectors(a, unit(b)));
    vec3d bc = unit(add_multiply_vectors(c, unit(b)));
    return dot(ba, bc);
}