# PDE Basic Usage

1. Ensure the necessary libraries are imported. These include numpy, enum and functools.
    ```python
    from mmcore.geom.pde import PDE, PDE2D, PDEMethodEnum, Offset
    ```
2. Define the mathematical function you wish to calculate the PDE for.

    ```python
    # Define the mathematical function
    def func(t):
        return t**2
    
    ```

3. Instantiate a PDE object, passing in the defined function as a parameter.

    ```python
    # Create a PDE object
    pde = PDE(func)
    ```

4. Evaluate the PDE at various points in time using the __call__ method of the instantiated PDE object. For example:
   result = pde(1.0)

    ```python
    # Evaluate the PDE
    result = pde(1.0) 
    ```
5. If you wish to calculate the offset at a specific point along the curve, use the Offset class, passing in the
   function and offset distance.
    ```python
    # Compute the offset
    offset = Offset(func, 1.0)
    offset_result = offset(1.0)
    ```

6. For advanced use, refer to the provided class and method documentation. Remember, numerical methods provide
   approximate solutions, and their accuracy depends on factors like the step size and the specific function being
   evaluated.

The `PDE` and `Offset` classes utilize numerical methods for derivative estimation. While these numerical methods
provide a good approximation, they are not always perfectly accurate. Use with caution for precise calculations. Be sure
to choose a suitable step-size `h` for your use case.
