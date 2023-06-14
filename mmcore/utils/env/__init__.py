import dotenv


def load_dotenv_from_path(filename, override=False, raise_error_if_not_found=False, **kwargs):
    return dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv(filename=filename,
                                                             raise_error_if_not_found=raise_error_if_not_found,
                                                             usecwd=True), override=override, **kwargs)


def load_dotenv_from_stream(stream, override=False, **kwargs):
    return dotenv.load_dotenv(stream=stream, override=override, **kwargs)
