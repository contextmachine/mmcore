ENABLE_WARNINGS = True


def disable_warnings():
    global ENABLE_WARNINGS
    ENABLE_WARNINGS = False


def enable_warnings():
    global ENABLE_WARNINGS
    ENABLE_WARNINGS = True


def props_update_support_warning(cls):
    if ENABLE_WARNINGS:
        raise UserWarning(f"{cls} instance support this property by default . The user argument will be ignored!")
