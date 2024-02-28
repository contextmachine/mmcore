def bisection1d(f, step=0.00001, start=-1, stop=3):
    # Smaller step values produce more accurate and precise results

    sign = f(start) > 0
    x = start
    roots = []
    while x <= stop:
        value = f(x)
        if value == 0:
            # We hit a root
            roots.append((x, value))

        elif (value > 0) != sign:
            # We passed a root
            roots.append((x, value))

        # Update our sign
        sign = value > 0
        x += step
    return roots
