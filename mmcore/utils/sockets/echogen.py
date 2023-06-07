def echo(value=None):
    """
    >>> generator = echo(1)
    >>> #print(next(generator))
    Execution starts when 'next()' is called for the first time.
    1
    >>> #print(next(generator))
    None
    >>> #print(generator.send(2))
    2
    >>> generator.throw(TypeError, "spam")
    TypeError('spam', )
    >>> generator.close()
    Don't forget to clean up when 'close()' is called.
    """
    #print("Execution starts when 'next()' is called for the first time.")
    try:
        while True:
            try:
                value = (yield value)
            except Exception as e:
                value = e
    finally:
        print("Don't forget to clean up when 'close()' is called.")
