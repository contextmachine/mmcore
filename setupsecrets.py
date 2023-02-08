# Copyright (c) CONTEXTMACHINE.
# Author : Andrew Astakhov (sth-v), aa@contextmachine.ru, https://github.com/sth-v
# Default secret link https://github.com/contextmachine/secrets.git available for contextmachine
# (https://github.com/contextmachine) members only!!!
# Do not worry if you do not have access to contextmachine secrets.
# All services used by mmcore can be run locally, see general documentation for details.


import base64
import json
import os
import pprint

__all__ = ["SecretsManager"]

import shutil
from dotenv import load_dotenv, find_dotenv, dotenv_values

import subprocess

from types import TracebackType

from typing import Any, ContextManager, Type

# For use custom secrets repositories, set SECRETS_LINK environment variable, it will be cloned with git.
# You can subclass SecretsManager and override "__setup_repo__" , "__setup_env__", or/and "additional_actions",
# for creating a custom pipeline, etc. custom auth, kubernetes connection, ...
# You also can fork this repo for mores customisation.


class SecretsManager(ContextManager):
    secrets_link = "https://github.com/contextmachine/secrets.git" if os.getenv("SECRETS_LINK") is None else \
        os.getenv("SECRETS_LINK")

    env_file_name = f'{os.getcwd()}/{"env.json" if os.getenv("SECRETS_ENV_FILE_NAME") is None else os.getenv("SECRETS_ENV_FILE_NAME")}'
    update = True if os.getenv("SECRETS_UPDATE") is None else os.getenv("SECRETS_UPDATE")
    logging = True if os.getenv("SECRETS_LOGGING") is None else os.getenv("SECRETS_LOGGING")
    update_environ = True
    additional_actions = [
        lambda data: os.environ.__setitem__("CADEX_LICENSE", base64.b64decode(os.getenv("CADEX_LICENSE")).decode())
        ]
    stringify_spec = dict()

    def log(self, msg, *args, **kwargs):

        if self.logging:
            print(msg, *args, **kwargs)

    def __init__(self, extend_additional_actions=(), update=True, update_stringify_spec=None, **kwargs):

        super().__init__()
        self.update = update
        self.log("Starting setup secrets ...")
        if update_stringify_spec is None:
            update_stringify_spec = dict()
        for name in dir(self):
            if not (name[0] == '_'):
                self.__dict__[name] = self.__getattribute__(name)
        self.additional_actions.extend(extend_additional_actions)
        self.stringify_spec.update(update_stringify_spec)
        self.kwargs = kwargs
        self.__dict__ |= kwargs

        self.log(pprint.pformat(self.__dict__) + "\n")

    def __exit__(self, __exc_type: Type[BaseException] | None, __exc_value: BaseException | None,
                 __traceback: TracebackType | None) -> bool | None:
        return False

    def __setup_repo__(self) -> None:
        """

        @return:  None
        @rtype: NoneType
        @summary: Override this method if you want to create a custom secret delivery pipeline.
        By default, a closed repository is used with the env file[1],
        which is referenced by the secrets_link attribute; it will be cloned with git.

        @note: This scheme is successful because it allows you to work with your git infrastructure
        and does not require additional authentication layers.
        The method will also work when using a local git repository, and when using Kubernetes[777]
        [1] Can be set by env_file_name attribute, "env.json" by default.
        [777] Kubernetes support is in progress, but will be available as soon as possible.
        """
        # This check is in the body of the method
        # that you can completely override what you need without changing __enter__

        # Yes, this may be excessive.
        # But I prefer to check this case to avoid having to routinely delete the directory in some cases.

        shutil.rmtree(f"{self.repo_name}", ignore_errors=True)

        proc = subprocess.Popen(["git", "clone", self.secrets_link])
        proc.wait()
        shutil.move(f"{os.getenv('WORKDIR')}/{self.repo_name}/{self.env_file_name}", self.env_file_name)
        shutil.rmtree(f"{os.getenv('WORKDIR')}/{self.repo_name}", ignore_errors=True)

    @property
    def USE_REPO(self):
        return bool(int(os.getenv("USE_REPO")))

    @property
    def USE_DOTENV(self):
        return bool(int(os.getenv("USE_DOTENV")))

    @USE_DOTENV.setter
    def USE_DOTENV(self, v):
        os.environ['USE_DOTENV'] = str(int(v))
        os.environ['USE_DOTENV'] = str(int(v))

    @USE_REPO.setter
    def USE_REPO(self, v):
        os.environ['USE_REPO'] = str(int(v))

    def __setup_dotenv__(self, usecwd=True, verbose=True, override=True, **kwargs):
        try:
            load_dotenv(find_dotenv(usecwd=usecwd,
                                    raise_error_if_not_found=True,
                                    filename=self.env_file_name if self.env_file_name.split(".")[
                                                                       -1] == "env" else ".env"),
                        verbose=verbose,
                        override=override,
                        **kwargs
                        )

        except IOError as err:
            print(f"dotenv miss env file:\n\t{err}")
        return dotenv_values()

    def __enter__(self) -> dict[str, Any]:
        """

        @return: Dictionary with secrets.
        @rtype: dict[str, Any]
        @summary: In SecretsManager, __enter__ structure like this:
            1. Call SecretsManager.__setup_repo__(self) -> None
            2. Call SecretsManager.__setup_env__(self) -> dict[str, Any] and return env dictionary.
            3. Call additional_actions in "for" loop like if this is:
                ```python
                for action in additional_actions:
                    action(<env dictionary>)
                ```
                Then every action can (must!) mutate env dictionary, you can set action order.
            4. If update_environ is True, `os.environ` should be updated from env dictionary.
                The Conversion value to str is determined by ```__stringify__``` method.
        @todo: __stringify__ documentation and examples.
        @todo: Kubernetes full support.

        """

        if self.USE_REPO:
            if self.update or not os.path.isfile(self.env_file_name):
                self.__setup_repo__()
            data = self.__setup_env__()
        elif self.USE_DOTENV:
            data = self.__setup_dotenv__()
        else:
            data = {}
        self.try_update_environ(data)
        self.try_additional_actions(data)
        if self.logging:
            print("Success!")
        return data

    _update = True

    def try_additional_actions(self, data):
        for action in self.additional_actions:
            action(data)

    def try_update_environ(self, data):
        if self.update_environ:
            for k in data.keys():
                os.environ[k] = self.__stringify__(k, data[k])

    @property
    def update(self):
        return self._update

    @update.setter
    def update(self, value):
        self._update = value

    @property
    def repo_name(self):
        return self.secrets_link.split("/")[-1].replace(".git", "")

    def __stringify__(self, key: str, val: Any) -> str:
        if self.stringify_spec.get(key) is not None:
            return self.stringify_spec.get(key)(str(val))
        else:
            return str(val)

    def __setup_env__(self) -> dict[str, Any]:
        """
            @return: Dictionary with secrets.
            @attention: This method will be called lasted additional_actions applications.
            It means that data can be changed after returned to this method.
            @rtype: dict[str, Any]

            @summary: Override this method if you want to return a custom secret dict.
            By default, will be open and return the secret env file[1] in a current root[2].
            [1] Name can be set by env_file_name attribute, "env.json" by default.
            [2] Current root always equals "__file__" attribute value.

            """

        with open(self.env_file_name, "r") as f:
            data = json.load(f)

        return data
