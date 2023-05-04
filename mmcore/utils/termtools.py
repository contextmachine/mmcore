from enum import IntEnum
from termcolor import RESET


class TermAttrs(IntEnum):
    bold = 1
    dark = 2
    underline = 4
    blink = 5
    reverse = 7
    concealed = 8


class TermColors(IntEnum):
    black = 30
    grey = 30
    red = 31
    green = 32
    yellow = 33
    blue = 34
    magenta = 35
    cyan = 36
    light_grey = 37
    dark_grey = 90
    light_red = 91
    light_green = 92
    light_yellow = 93
    light_blue = 94
    light_magenta = 95
    light_cyan = 96
    white = 97


def mmprint(s, *args, **kwargs):
    print(f"{PREF} {s}", *args, **kwargs)


class ColorStr(str):
    color: TermColors = TermColors.light_grey
    term_attrs: list[TermAttrs] = [TermAttrs.bold]

    def __new__(cls, s, color=None, term_attrs=()):
        inst = str.__new__(cls, s)
        if color is not None:
            inst.color = color
        if term_attrs is not None:
            inst.term_attrs = term_attrs
        return inst

    def __repr__(self):
        return str.__repr__(self)

    def __str__(self):
        _attrs = "".join([f"\033[{attr}m" for attr in self.term_attrs])
        return f"\033[{self.color}m{_attrs}{str.__str__(self)}{RESET}"


PREF = f"[{ColorStr('mmcore', color=TermColors.magenta, term_attrs=[TermAttrs.blink, TermAttrs.bold])}]"


class MMColorStr(ColorStr):
    color: TermColors = TermColors.light_grey
    term_attrs: list[TermAttrs] = [TermAttrs.blink]
    real_len=11
    def __str__(self):
        return f"{PREF} {super().__str__()}{RESET}"

    def __add__(self, other):
        return self.__str__()+other.__str__()