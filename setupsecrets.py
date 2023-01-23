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

from types import TracebackType

from typing import Any, ContextManager, Type

# For use custom secrets repositories, set SECRETS_LINK environment variable, it will be cloned with git.
# You can subclass SecretsManager and override "__setup_repo__" , "__setup_env__", or/and "additional_actions",
# for creating a custom pipeline, etc. custom auth, kubernetes connection, ...
# You also can fork this repo for mores customisation.

USE_REPO = True


class SecretsManager(ContextManager):
    secrets_link = "https://github.com/contextmachine/secrets.git" if os.getenv("SECRETS_LINK") is None else \
        os.getenv("SECRETS_LINK")

    env_file_name = "env.json" if os.getenv("SECRETS_ENV_FILE_NAME") is None else os.getenv("SECRETS_ENV_FILE_NAME")
    update = True if os.getenv("SECRETS_UPDATE") is None else os.getenv("SECRETS_UPDATE")
    logging = True if os.getenv("SECRETS_LOGGING") is None else os.getenv("SECRETS_LOGGING")
    update_environ = True if os.getenv("SECRETS_UPDATE_ENVIRON") is None else os.getenv("SECRETS_UPDATE_ENVIRON")
    additional_actions = [
        lambda data: data.__setitem__("CADEX_LICENSE", base64.b64decode(data["CADEX_LICENSE"]).decode())
        ]
    stringify_spec = dict()

    def __init__(self, extend_additional_actions=(), update_stringify_spec=None, **kwargs):

        super().__init__()
        if update_stringify_spec is None:
            update_stringify_spec = dict()
        for name in dir(self):
            if not (name[0] == '_'):
                self.__dict__[name] = self.__getattribute__(name)
        self.additional_actions.extend(extend_additional_actions)
        self.stringify_spec.update(update_stringify_spec)
        self.__dict__ |= kwargs
        if self.logging:
            pprint.pprint(self.__dict__)
            print("\n")

    def __exit__(self, __exc_type: Type[BaseException] | None, __exc_value: BaseException | None,
                 __traceback: TracebackType | None) -> bool | None:
        return False

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
        if self.logging:
            print("Starting setup secrets ...")

        data = self.__setup_env__()

        for action in self.additional_actions:
            action(data)
        if self.update_environ:
            for k in data.keys():
                os.environ[k] = self.__stringify__(k, data[k])

        if self.logging:
            print("Success!")
        return data

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
