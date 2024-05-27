# pypy performance
Since we can support pypy3 on this branch you can 
```bash
pypy3 -m pip install --user --force-reinstall git+https://github.com/contextmachine/mmcore.git@tiny
```
At this point, don't expect much performance improvement as we are still using a lot of numpy.
We will move away from this in the future, but not to the detriment of performance on the cpython version.

Tests so far show that the performance of mmcore on pypy lags behind mmcore on cpython. 
It's important to note that I haven't run any tests that reveal the advantages of jit, rather the opposite,
I've mostly done fairly simple tests, each of which involved calling a single procedure once. 
Let's be honest, if you're a staunch pypy supporter, you'll say these are just disgusting tests.

```
#cpython
(0.0, 0.0018742084503173828)
(0.0, 0.03659391403198242)
19.799999999999986 19.8
(0.0, 0.5192592144012451)
0.23 0.23
(0.0, 0.019153118133544922)


#pypy
(0.0, 0.014207839965820312)
(0.0, 0.2798471450805664)
19.8 19.8
(0.0, 2.7021310329437256)
0.23000000000031934 0.23
(0.0, 0.087615966796875)


```
Without going into the details of the test, we take the integral of the NURBS spline (length over parameter) several times, 
and then solve the inverse problem of parameter estimation over length. 
>I can say that pypy is about **7x** slower at the first estimation, \
> however with each call the distance decreases \
> and by the last step we have a difference of **4x**. 

This makes me see prospects in using the library with pypy and develop this direction.

