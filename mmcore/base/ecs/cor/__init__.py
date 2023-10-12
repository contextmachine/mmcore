class AbstractHandler(object):
    """Parent class of all concrete handlers"""
    next_handler: 'AbstractHandler'

    def __init__(self, *next_handlers):
        super().__init__()
        self.next_handlers = next_handlers

    def handle(self, *args, **kwargs):
        """It calls the processRequest through given request"""

        handled = self.processRequest(*args, **kwargs)

        if handled is None:
            return self.next_handler.handle(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.handle(*args, **kwargs)

    def processRequest(self, *args, **kwargs):
        """throws a NotImplementedError"""

        raise NotImplementedError('First implement it !')


class SecondConcreteHandler(AbstractHandler):
    """Concrete Handler # 2: Child class of AbstractHandler"""

    def processRequest(self, request):
        '''return True if the request is handled'''

        if 'e' < request <= 'l':
            print("This is {} handling request '{}'".format(self.__class__.__name__, request))

            return True


class ThirdConcreteHandler(AbstractHandler):
    """Concrete Handler # 3: Child class of AbstractHandler"""

    def processRequest(self, request):
        '''return True if the request is handled'''

        if 'l' < request <= 'z':
            print("This is {} handling request '{}'".format(self.__class__.__name__, request))

            return True


class DefaultHandler(AbstractHandler):
    """Default Handler: child class from AbstractHandler"""

    def processRequest(self, request):
        """Gives the message that the request is not handled and returns true"""

        print("This is {} telling you that request '{}' has no handler right now.".format(self.__class__.__name__,

                                                                                          request))

        return True
