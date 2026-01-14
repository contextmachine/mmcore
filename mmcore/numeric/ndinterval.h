#ifndef __INTERVAL_H__
#define __INTERVAL_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdio.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

typedef struct {
    double l;
    double u;
} interval;

/**
 * ADD OPERATIONS
*/
static inline interval interval_add(interval i1, interval i2){
    return (interval) {i1.l + i2.l, i1.u + i2.u};
}
static inline void interval_inplace_add(interval* i1, interval i2) {
    i1->l += i2.l;
    i1->u += i2.u;
    return;
}
static inline interval interval_add_scalar(interval i, double s) {
    return (interval) {i.l + s, i.u + s};
}
static inline void interval_inplace_add_scalar(interval* i, double s) {
    i->l += s;
    i->u += s;
    return;
}
static inline interval interval_scalar_add(double s, interval i) {
    return (interval) {i.l + s, i.u + s};
}
static inline void interval_inplace_scalar_add(double s, interval* i) {
    i->l += s;
    i->u += s;
    return;
}

/**
 * SUBTRACT OPERATIONS
*/
static inline interval interval_subtract(interval i1, interval i2){
    return (interval) {i1.l - i2.u, i1.u - i2.l};
}
static inline void interval_inplace_subtract(interval* i1, interval i2) {
    i1->l -= i2.u;
    i1->u -= i2.l;
    return;
}
static inline interval interval_subtract_scalar(interval i, double s) {
    return (interval) {i.l - s, i.u - s};
}
static inline void interval_inplace_subtract_scalar(interval* i, double s) {
    i->l -= s;
    i->u -= s;
    return;
}
static inline interval interval_scalar_subtract(double s, interval i) {
    return (interval) {s - i.u, s - i.l};
}
static inline void interval_inplace_scalar_subtract(double s, interval* i) {
    i->l -= s;
    i->u -= s;
    return;
}

/**
 * MULTIPLY OPERATIONS
*/
static inline interval interval_multiply(interval i1, interval i2) {
    double _1 = i1.l*i2.l;
    double _2 = i1.l*i2.u;
    double _3 = i1.u*i2.l;
    double _4 = i1.u*i2.u;
    return (interval) {
        fmin(fmin(_1, _2), fmin(_3, _4)),
        fmax(fmax(_1, _2), fmax(_3, _4))
    };
} 
static inline void interval_inplace_multiply(interval* i1, interval i2) {
    double _1 = i1->l*i2.l;
    double _2 = i1->l*i2.u;
    double _3 = i1->u*i2.l;
    double _4 = i1->u*i2.u;
    i1->l = fmin(fmin(_1, _2), fmin(_3, _4));
    i1->u = fmax(fmax(_1, _2), fmax(_3, _4));
    return;
}
static inline interval interval_multiply_scalar(interval i, double s) {
    if (s >= 0) { return (interval) { i.l*s, i.u*s }; }
    else        { return (interval) { i.u*s, i.l*s }; }
}
static inline void interval_inplace_multiply_scalar(interval* i, double s) {
    if (s >= 0) { 
        i->l *= s; 
        i->u *= s;
    } else { 
        double ltmp = i->l;
        i->l = i->u * s; 
        i->u = ltmp * s;
    }
    return;
}
static inline interval interval_scalar_multiply(double s, interval i) {
    if (s >= 0) { return (interval) { i.l*s, i.u*s }; }
    else        { return (interval) { i.u*s, i.l*s }; }
}
static inline void interval_inplace_scalar_multiply(double s, interval* i) {
    if (s >= 0) { 
        i->l *= s; 
        i->u *= s;
    } else { 
        double ltmp = i->l;
        i->l = i->u * s; 
        i->u = ltmp * s;
    }
    return;
}

/**
 * DIVIDE OPERATIONS
*/
static inline interval interval_inverse(interval i) {
    if ((i.l > 0 && i.u > 0) || (i.l < 0 && i.u < 0)) {
        return (interval) { 1/i.u, 1/i.l };
    // } else if (i.l != 0 && i.u == 0) {
    //     return (interval) { -INFINITY, 1/i.l };
    // } else if (i.l == 0 && i.l != 0) {
    //     return (interval) { 1/i.u, INFINITY };
    } else {
        return (interval) { -INFINITY, INFINITY };
    }
}
static inline interval interval_divide(interval i1, interval i2) {
    return interval_multiply(i1, interval_inverse(i2));
} 
static inline void interval_inplace_divide(interval* i1, interval i2) {
    interval res = interval_divide(*i1, i2);
    i1->l = res.l;
    i1->u = res.u;
    return;
}
static inline interval interval_divide_scalar(interval i, double s) {
    return interval_multiply_scalar(i, (1/s));
}
static inline void interval_inplace_divide_scalar(interval* i, double s) {
    interval res = interval_multiply_scalar(*i, (1/s));
    i->l = res.l;
    i->u = res.u;
    return;
}
static inline interval interval_scalar_divide(double s, interval i) {
    return interval_scalar_multiply(s, interval_inverse(i));
}
static inline void interval_inplace_scalar_divide(double s, interval* i) {
    interval res = interval_scalar_multiply(s, interval_inverse(*i));
    i->l = res.l;
    i->u = res.u;
    return;
}

/**
 * POWER OPERATIONS
*/
static inline interval interval_square (interval i){
    interval ret;
    double lp = pow(i.l,2);
    double up = pow(i.u,2);
    if (i.l <= 0 && i.u >= 0) {
        ret.l = 0;
    } else {
        ret.l = fmin(lp, up);
    }
    ret.u = fmax(lp, up);
    return ret;
}
static inline interval interval_power_scalar(interval i, double s){
    if (s < 0) {
        return (interval) interval_inverse(interval_power_scalar(i,-s));
    }
    if (i.l > 0 && i.u > 0) {
        return (interval) { pow(i.l,s), pow(i.u,s) };
    }
    int p = round(s);
    if (p % 2) {
        // odd power
        return (interval) { pow(i.l,p), pow(i.u,p) }; 
    } else {
        // even power
        interval ret;
        double lp = pow(i.l,p);
        double up = pow(i.u,p);
        if (i.l <= 0 && i.u >= 0) {
            ret.l = 0;
        } else {
            ret.l = fmin(lp, up);
        }
        ret.u = fmax(lp, up);
        return ret;
    }
}
static inline void interval_inplace_power_scalar(interval* i, double s){
    interval res = interval_power_scalar(*i, s);
    i->l = res.l;
    i->u = res.u;
    return;
}

/**
 * UNARY OPERATIONS
*/

static inline int interval_nonzero (interval i) {
    return !(i.l == 0 && i.u == 0);
}

static inline interval interval_negative(interval i) {
    return (interval) { -i.u, -i.l };
}

// #define sign(x) ((x > 0) - (x < 0))

// static inline interval interval_sin(interval i){
//     double diff = i.u - i.l;
//     if (diff >= 2*M_PI) {
//         return (interval) { -1, 1 };
//     }
//     int div = (int) (i.u / (2*M_PI));
//     i.l -= div*2*M_PI - M_PI; i.u -= div*2*M_PI - M_PI;

//     interval ret;

//     if ((i.l <=  -M_PI/2 && i.u >= -M_PI/2) ||
//         (i.l <= 3*M_PI/2 && i.u >= 3*M_PI/2)) {
//         ret.l = -1;
//     } else {
//         ret.l = fmin(sin(i.l), sin(i.u));
//     }

//     if ((i.l <= -3*M_PI/2 && i.u >= -3*M_PI/2) ||
//         (i.l <=    M_PI/2 && i.u >=    M_PI/2)) {
//         ret.u = 1;
//     } else {
//         ret.u = fmax(sin(i.l), sin(i.u));
//     }

//     return ret;
// }

static inline interval interval_sin(interval i) {
    double diff = i.u - i.l;
    if (diff <= M_PI) {
        double cl = cos(i.l);
        double cu = cos(i.u);
        if (cl >= 0 && cu >= 0) {
            return (interval) { sin(i.l), sin(i.u) }; 
        } else if (cl <= 0 && cu <= 0) {
            return (interval) { sin(i.u), sin(i.l) };
        } else if (cl >= 0 && cu <= 0) {
            return (interval) { fmin(sin(i.l), sin(i.u)), 1 };
        } else if (cl <= 0 && cu >= 0) {
            return (interval) { -1, fmax(sin(i.l), sin(i.u)) };
        }
    }
    if (diff <= 2*M_PI) {
        double cl = cos(i.l);
        double cu = cos(i.u);
        if (cl >= 0 && cu >= 0) {
            return (interval) { -1, 1 }; 
        } else if (cl <= 0 && cu <= 0) {
            return (interval) { -1, 1 };
        } else if (cl >= 0 && cu <= 0) {
            return (interval) { fmin(sin(i.l), sin(i.u)), 1 };
        } else if (cl <= 0 && cu >= 0) {
            return (interval) { -1, fmax(sin(i.l), sin(i.u)) };
        }
    }
    return (interval) { -1, 1 };
}

static inline interval interval_cos(interval i){
    return interval_sin((interval){ i.l+M_PI/2, i.u+M_PI/2 });
}
static inline interval interval_tan(interval i){
    int div = (int) ((i.u + M_PI_2) / (M_PI));
    i.l -= div*M_PI; i.u -= div*M_PI;

    if (i.l < -M_PI_2) {
        return (interval) { -INFINITY, INFINITY };
    }

    return (interval){ tan(i.l), tan(i.u) };
}
static inline interval interval_arctan(interval i) {
    return (interval){ atan(i.l), atan(i.u) };
}
static inline interval interval_tanh(interval i) {
    return (interval) { tanh(i.l), tanh(i.u) };
}
static inline interval interval_exp(interval i){
    return (interval){ exp(i.l), exp(i.u) };
}
static inline interval interval_sqrt(interval i){
    if (i.l < 0) {
        return (interval) { -INFINITY, INFINITY };
    }
    return (interval){ sqrt(i.l), sqrt(i.u) };
}

static inline double interval_norm(interval i){
    return (i.u - i.l);
}

/**
 * SET OPERATIONS
*/

static inline interval interval_union(interval i1, interval i2){
    return (interval) { fmin(i1.l, i2.l), fmax(i1.u, i2.u) };
}
static inline interval interval_intersection(interval i1, interval i2){
    double rl = fmax(i1.l, i2.l);
    double ru = fmin(i1.u, i2.u);
    if (rl > ru) {
        return (interval) { NAN, NAN };
    }
    return (interval) { rl, ru };
}

static inline interval interval_minimum (interval i1, interval i2) {
    return (interval) { fmin(i1.l, i2.l), fmin(i1.u, i2.u) };
}

static inline interval interval_maximum (interval i1, interval i2) {
    return (interval) { fmax(i1.l, i2.l), fmax(i1.u, i2.u) };
}


/**
 * UTILITY
*/

static inline int interval_equal(interval i1, interval i2){
    // return (fabs(i1.l - i2.l) < 1e-10 && fabs(i1.u - i2.u) < 1e-10);
    return (i1.l == i2.l && i1.u == i2.u);
}
static inline int interval_not_equal(interval i1, interval i2){
    return !interval_equal(i1, i2);
}

// true if i1 \subseteq i2
static inline int interval_subseteq(interval i1, interval i2) {
    return (i1.l >= i2.l && i1.u <= i2.u);
}
// true if i1 \supseteq i2
static inline int interval_supseteq(interval i1, interval i2) {
    return (i2.l >= i1.l && i2.u <= i1.u);
}

// true if i1 \subset i2
static inline int interval_subset(interval i1, interval i2) {
    return (i1.l > i2.l && i1.u < i2.u);
}
// true if i1 \supset i2
static inline int interval_supset(interval i1, interval i2) {
    return (i2.l > i1.l && i2.u < i1.u);
}


#ifdef __cplusplus
}
#endif

#endif
