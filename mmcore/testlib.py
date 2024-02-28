import abc
import json
import rich


class CaseFileTest:
    def __init__(self): ...

    @abc.abstractmethod
    def case(self, inputs): ...

    @abc.abstractmethod
    def check(self, outputs, result) -> bool: ...

    def __call__(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        l = []
        for item in data:
            inpt = item['inputs']
            res = item['result']
            out = self.case(inpt)
            valid = self.check(out, res)
            rich.print(
                {
                    'status': valid,
                    'inputs': inpt,
                    "result": {
                        'correct': res,
                        'prediction': out
                    }

                }
            )

            l.append(valid)

        return l
