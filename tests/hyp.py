def llll(hp,step=600):
    a = ClosestPoint(hp.b, hp.side_d)(x0=0.5, bounds=((0, 1),))
    uva = unit([hp.side_a.a, hp.side_a.b, hp.side_a.c])
    l = Linear.from_two_points(hp.a, np.asarray(a.pt).flatten())
    uvb = unit([l.a, l.b, l.c])

    dt = np.dot(uva, uvb)
    lll = []
    stp = (1 / dt)*step
    for pt in hp.side_a.divide_distance(stp).T:
        lll.append(Linear.from_two_points(pt, np.asarray(ClosestPoint(pt, hp.side_d)(x0=0.5, bounds=((0, 1),)).pt).flatten()))
    return lll
