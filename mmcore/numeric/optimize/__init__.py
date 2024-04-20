import numpy as np
import numpy as np


def ON_FindLocalMinimum(f, farg, ax, bx, cx, rel_stepsize_tol, abs_stepsize_tol, max_it, t_addr):
    d = e = 0.0
    t_addr[0] = bx
    _f = lambda x: f(farg, x)

    if max_it < 2:
        print("max_it must be => 2")
        return 0
    if not ((0.0 < rel_stepsize_tol < 1.0) and abs_stepsize_tol > 0.0):
        print("rel_stepsize_tol must be strictly between 0.0 and 1.0 and abs_stepsize_tol must be > 0")
        return 0

    a, b = (ax, cx) if ax < cx else (cx, ax)
    x = w = v = bx
    rc = _f(x)
    if rc:
        if rc < 0: print("ON_FindLocalMinimum() f() failed to evaluate.")
        t_addr[0] = x
        return 1 if rc > 0 else 0

    fw = fv = fx = rc
    dw = dv = dx = rc

    for _ in range(max_it):
        xm = 0.5 * (a + b)
        tol1 = rel_stepsize_tol * abs(x) + abs_stepsize_tol
        tol2 = 2.0 * tol1
        if abs(x - xm) <= (tol2 - 0.5 * (b - a)):
            t_addr[0] = x
            return 1
        if abs(e) > tol1:
            d1 = 2.0 * (b - a)
            d2 = d1
            if dw != dx: d1 = (w - x) * dx / (dx - dw)
            if dv != dx: d2 = (v - x) * dx / (dx - dv)
            u1 = x + d1
            u2 = x + d2
            ok1 = (a - u1) * (u1 - b) > 0.0 and dx * d1 <= 0.0
            ok2 = (a - u2) * (u2 - b) > 0.0 and dx * d2 <= 0.0
            olde = e
            e = d
            if ok1 or ok2:
                d = d1 if ok1 and (ok2 and abs(d1) < abs(d2) or not ok2) else d2
                if abs(d) <= abs(0.5 * olde):
                    u = x + d
                    if u - a < tol2 or b - u < tol2:
                        d = tol1 if xm >= x else -tol1
                else:
                    d = 0.5 * (e=(dx if dx >= 0.0 else a-x if a-x > b-x else b-x))
            else:
                    d = 0.5 * (e=(dx if dx >= 0.0 else a-x if a-x > b-x else b-x))
        else:
                d = 0.5 * (e=(dx if dx >= 0.0 else a-x if a-x > b-x else b-x))

                u = (x + d if abs(d) >= tol1 else (x + tol1 if d >= 0.0 else x - tol1))

                fu = _f(u)
                if fu > rc:
                    t_addr[0] = x
                    return 1

                if fu <= fx:
                    if u >= x:
                        a = x
                    else:
                        b = x
                    v, fv, dv = w, fw, dw
                    w, fw, dw = x, fx, dx
                    x, fx, dx = u, fu, du
                else:
                    if u < x:
                        a = u
                    else:
                        b = u
                    if fu <= fw or w == x:
                        v, fv, dv = w, fw, dw
                        w, fw, dw = u, fu, du
                    elif fu < fv or v == x or v == w:
                        v, fv, dv = u, fu, du

            t_addr[0] = x
            print("ON_FindLocalMinimum() failed to converge")
            return 2


class ON_LocalZero1:
            def __init__(self):
                self.m_t0 = np.nan
                self.m_t1 = np.nan
                self.m_f_tolerance = 0.0
                self.m_t_tolerance = 0.0
                self.m_k = None
                self.m_k_count = 0
                self.m_s0 = np.nan
                self.m_f0 = np.nan
                self.m_s1 = np.nan
                self.m_f1 = np.nan

            def BracketZero(self, s0, f0, s1, f1, level):

                if ((f0 <= 0.0 and f1 >= 0.0) or (f0 >= 0.0 and f1 <= 0.0)) or abs(f0) <= self.m_f_tolerance or abs(
                        f1) <= self.m_f_tolerance:
                    self.m_t0 = s0
                    self.m_t1 = s1
                    return True

                if level <= 8:
                    s = 0.5 * s0 + s1
                    iterated_func = lambda s: self.Evaluate(s)
                    f, d = iterated_func(s), iterated_func(s)

                    if (s0 < s and s < s1) and iterated_func:
                        if f * d >= 0.0:
                            if self.BracketZero(s0, f0, s, f, level):
                                self.m_s0 = s0
                                self.m_f0 = f0
                                self.m_s1 = s
                                self.m_f1 = f
                                return True
                            if self.BracketZero(s, f, s1, f1, level):
                                self.m_s0 = s
                                self.m_f0 = f
                                self.m_s1 = s1
                                self.m_f1 = f1
                                return True
                        elif self.BracketZero(s, f, s1, f1, level):
                            self.m_s0 = s
                            self.m_f0 = f
                            self.m_s1 = s1
                            self.m_f1 = f1
                            return True
                        elif self.BracketZero(s0, f0, s, f, level):
                            self.m_s0 = s0
                            self.m_f0 = f0
                            self.m_s1 = s
                            self.m_f1 = f
                            return True
                    return False

            def FindZero(self, t):
                if not t:
                    print("Illegal input - m_t0 and m_t1 are not valid.")
                    return False
                elif not t:
                    self.m_s0 = self.m_s1 = self.m_t1
                elif self.m_t0 >= self.m_t1:
                    self.m_s0 = self.m_t0
                    self.m_s1 = self.m_t1
                else:
                    self.m_s0 = self.m_t1
                    self.m_s1 = s0
                t = self.m_s0
                return True

            def NewtonRaphson(self, s0, f0, s1, f1, maxit):
                if abs(f0) <= self.m_f_tolerance and abs(f0) <= abs(f1):
                    return s0

                if abs(f1) <= self.m_f_tolerance:
                    return s1

                if f0 > f1:
                    x, s0, s1 = s0, s1, x
                    x, f0, f1 = f0, f1, x

                s = 0.5 * (s0 + s1)
                iterated_func = lambda s: self.Evaluate(s)

            f, d = iterated_func(s), iterated_func(s)

            if abs(f) <= self.m_f_tolerance:
                return s

            if f1 <= 0.0:
                return s0 if abs(f0) <= abs(f1) else s1

            ds, prevds = abs(s1 - s0), 0.0

            while maxit:
                if (f + (s0 - s) * d) * (f + (s1 - s) * d) > 0.0 or abs(2 * f) > abs(prevds * d):
                    prevds = ds
                    ds = 0.5 * (s1 - s0)
                    s = s0 + ds
                    if s == s0:
                        return s0 if abs(f0) < abs(f1) else s1
                else:
                    prevds = ds
                    ds = -f / d
                    x = s
                    s += ds
                    if s == x:
                        return s0 if abs(f0) < abs(f1) else s1

                fs, ds = f, d = iterated_func(s)
                if abs(fs) <= self.m_f_tolerance:
                    return s

                if f < 0.0:
                    f0, s0 = f, s
                else:
                    f1, s1 = f, s

                if abs(s1 - s0) <= self.m_t_tolerance:
                    return s0 if abs(f0) <= abs(f1) else s1
            return s0 if abs(f0) <= abs(f1) else s1

def ON_LocalZero1_FindZero(t):
    m_s0, m_s1, m_f0, m_f1, m_f_tolerance = 0.0, 0.0, 0.0, 0.0, 0.001
    m_t0, m_t1 = 0.0, 0.0  # Assuming that m_t0 and m_t1 are initially zero

    # def ON_ERROR(message): print(message)  # Define ON_ERROR function if necessary
    # def Evaluate(s0, f0, dummy1, dummy2): pass  # Define Evaluate function if necessary
    # def BracketZero(s0, f0, s1, f1): pass  # Define BracketZero function if necessary
    # def NewtonRaphson(s0, f0, s1, f1, maxit, t): pass  # Define NewtonRaphson function if necessary
    # def BracketSpan(s0, f0, s1, f1): pass  # Define BracketSpan function if necessary

    if not np.isfinite(m_t0):
        if not np.isfinite(m_t1):
            ON_ERROR("Illegal input - m_t0 and m_t1 are not valid.")
            return False
        m_s0 = m_s1 = m_t1
    elif not np.isfinite(m_t1):
        m_s0 = m_s1 = m_t0
    elif m_t0 <= m_t1:
        m_s0 = m_t0
        m_s1 = m_t1
    elif m_t1 < m_t0:
        m_s0 = m_t1
        m_s1 = m_t0
    else:
        ON_ERROR("Illegal input - m_t0 and m_t1 are not valid.")
        return False

    if m_s0 == m_s1:
        if Evaluate(m_s0, m_f0, None, 1):
            m_f1 = m_f0
            if np.abs(m_f0) <= m_f_tolerance:
                t = m_s0
                return True
            ON_ERROR(
                "Illegal input - m_t0 = m_t1 and the function value is not zero at m_t0."
            )
            return False
        ON_ERROR("Evaluation failed.")
        return False

    if not Evaluate(m_s0, m_f0, None, 1):
        ON_ERROR("Evaluation failed at m_s0.")
        return False

    if not Evaluate(m_s1, m_f1, None, -1):
        ON_ERROR("Evaluation failed at m_s1.")
        return False

    if not BracketZero(m_s0, m_f0, m_s1, m_f1):
        ON_ERROR("Unable to bracket a zero of the function.")
        return False

    if np.abs(m_f0) <= m_f_tolerance and np.abs(m_f0) <= np.abs(m_f1):
        t = m_s0
        return True

    if np.abs(m_f1) <= m_f_tolerance:
        t = m_s1
        return True

    if not BracketSpan(m_s0, m_f0, m_s1, m_f1):
        ON_ERROR(
            "Unable to bracket the function in a span of m_k[]. m_k[] may be invalid."
        )
        return False

    if not NewtonRaphson(m_s0, m_f0, m_s1, m_f1, 128, t):
        ON_ERROR("Newton-Raphson failed to converge. Is your function C2?")
        return False

    return True
def ON_LocalZero1_NewtonRaphson(s0, f0, s1, f1, maxit):
    # private function - input must satisfy

    s, f, d, x, ds, prevds = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    m_f_tolerance = 0.001
    m_t_tolerance = 0.001

    if np.abs(f0) <= m_f_tolerance and np.abs(f0) <= np.abs(f1):
        # |f(s0)| <= user specified stopping tolerance
        return s0
    if np.abs(f1) <= m_f_tolerance:
        # |f(s1)| <= user specified stopping tolerance
        return s1

    if f0 > 0.0:
        x = s0
        s0 = s1
        s1 = x
        x = f0
        f0 = f1
        f1 = x

    s = 0.5 * (s0 + s1)

    # Placeholder for Evaluate function
    f, d = [0.0, 0.0]

    if np.abs(f) <= m_f_tolerance:
        # |f(s)| <= user specified stopping tolerance
        return s

    if f1 <= 0.0:
        return s0 if (np.abs(f0) <= np.abs(f1)) else s1

    ds = np.abs(s1 - s0)
    prevds = 0.0

    while maxit:
        maxit -= 1
        if (f + (s0 - s) * d) * (f + (s1 - s) * d) > 0.0 or np.abs(2.0 * f) > np.abs(
            prevds * d
        ):
            # bisect
            prevds = ds
            ds = 0.5 * (s1 - s0)
            s = s0 + ds
            if s == s0:
                if np.abs(f1) < np.abs(f0):
                    s = s1
                return s
        else:
            # Newton iterate
            prevds = ds
            ds = -f / d
            x = s
            s += ds
            if s == x:
                if np.abs(f0) < np.abs(f):
                    f = f0
                    s = s0
                if np.abs(f1) < np.abs(f):
                    s = s1
                return s

        # Placeholder for Evaluate function
        f, d = [0.0, 0.0]

        if np.abs(f) <= m_f_tolerance:
            # |f(s)| <= user specified stopping tolerance
            if np.abs(f0) < np.abs(f):
                f = f0
                return s0

            if np.abs(f1) < np.abs(f):
                return s1

        if f < 0.0:
            f0 = f  # f0 needed for emergency bailout
            s0 = s
        else:  # f > 0.0
            f1 = f  # f1 needed for emergency bailout
            s1 = s

        if np.abs(s1 - s0) <= m_t_tolerance:
            # a root has been bracketed to an interval that is small enough
            return s0 if (np.abs(f0) <= np.abs(f1)) else s1

    return s0 if (np.abs(f0) <= np.abs(f1)) else s1  # emergency bailout
