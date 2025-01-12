from __future__ import annotations

from mmcore.api import Base


class Color(Base):
    """
    The Color class wraps all of the information that defines a simple color.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> Color:
        return Color()

    @classmethod
    def create(cls, red: int, green: int, blue: int, opacity: int) -> Color:
        """
        Creates a new color.
        red : The red component of the color. The value can be 0 to 255.
        green : The green component of the color. The value can be 0 to 255.
        blue : The blue component of the color. The value can be 0 to 255.
        opacity : The opacity of the color. The value can be 0 to 255.
        Returns the newly created color or null if the creation failed.
        """
        return Color()

    def get_color(self) -> tuple[bool, int, int, int, int]:
        """
        Gets all of the information defining this color.
        red : The red component of the color. The value can be 0 to 255.
        green : The green component of the color. The value can be 0 to 255.
        blue : The blue component of the color. The value can be 0 to 255.
        opacity : The opacity of the color. The value can be 0 to 255. A value of 255 indicates
        it is completely opaque.
        Returns true if getting the color information was successful.
        """
        return (bool(), int(), int(), int(), int())

    def set_color(self, red: int, green: int, blue: int, opacity: int) -> bool:
        """
        Sets all of the color information.
        red : The red component of the color. The value can be 0 to 255.
        green : The green component of the color. The value can be 0 to 255.
        blue : The blue component of the color. The value can be 0 to 255.
        opacity : The opacity of the color. The value can be 0 to 255. A value of 255 indicates
        it is completely opaque. Depending on where the color is used, the opacity
        value may be ignored.
        Returns true if setting the color information was successful.
        """
        return bool()

    @property
    def red(self) -> int:
        """
        Gets and sets the red component of the color. The value can be 0 to 255.
        """
        return int()

    @red.setter
    def red(self, value: int):
        """
        Gets and sets the red component of the color. The value can be 0 to 255.
        """
        pass

    @property
    def green(self) -> int:
        """
        Gets and sets the green component of the color. The value can be 0 to 255.
        """
        return int()

    @green.setter
    def green(self, value: int):
        """
        Gets and sets the green component of the color. The value can be 0 to 255.
        """
        pass

    @property
    def blue(self) -> int:
        """
        Gets and sets the blue component of the color. The value can be 0 to 255.
        """
        return int()

    @blue.setter
    def blue(self, value: int):
        """
        Gets and sets the blue component of the color. The value can be 0 to 255.
        """
        pass

    @property
    def opacity(self) -> int:
        """
        Gets and sets the opacity of the color. The value can be 0 to 255.
        """
        return int()

    @opacity.setter
    def opacity(self, value: int):
        """
        Gets and sets the opacity of the color. The value can be 0 to 255.
        """
        pass
