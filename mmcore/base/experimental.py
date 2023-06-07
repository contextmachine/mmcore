import json

import rich
class ST:
    INDENT = 2
    ignore = ['tt', '_tt', "ignore"]

    def __init__(self, **kw):
        self._tt = "   "
        self.tt = str(self._tt)
        super().__init__()
        self.__dict__ |= kw
        self.__dict__["type"] = self.__class__.__name__
        self.uuid = hex(id(self))

    def __str__(self):

        lst = []
        for k, v in self.__dict__.items():
            if isinstance(v, ST):
                v.tt += self.tt
            if isinstance(v, str):
                v = f'"{v}"'
            if not k in self.ignore:
                lst.append(f'{self._tt}"{k}": {v.__str__()}')
        vls = f",\n{self.tt}".join(lst)
        res = "{" + f"\n{self.tt}{vls}\n{self.tt}" + "}"
        self.tt = str(self._tt)
        return res


    def pretty(self, **kwargs):
        rich.print_json(self.__str__())

s=ST(a=5,b=5, name="s1")
s2=ST(a=11,b="rrrr",name="s2")
s3=ST(first=s2,second=s,name="s3")

s5=ST(first=s,second=s,name="s5")
s4=ST(first=s2,second=s5,name="s4")

s7=ST(first=s4,bar=22,name="s7")
s6=ST(first=s7,foo=s2,name="s6")


s21=ST(a=11,b="rrrr",name="s21",previous=s2)
s31=ST(first=s21,second=s5, previous=s3, name="s31")
s51=ST(first=s7, second=s4, previous=s5, sname="s51")
s41=ST(first=s21,second=s51,previous=s4,name="s41")
s71=ST(first=s41,bar=22,previous=s7,name="s71")
s61=ST(first=s71,foo=s21,previous=s6,name="s61")

print(s61)
json.loads(s7.__str__())
