TOKEN = "$"


def islink(s):
    if isinstance(s, str):
        if s.startswith(TOKEN):
            return True
    return False


def clear_token(s: str):
    return s[len(TOKEN):]


def make_link(s: str):
    return f'{TOKEN}{s}'
