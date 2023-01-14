# Copyright (c) CONTEXTMACHINE.
# Author : Andrew Astakhov (sth-v), aa@contextmachine.ru, https://github.com/sth-v
# Default secret link https://github.com/contextmachine/secrets.git available for contextmachine
# (https://github.com/contextmachine) members only!!!
# Do not worry if you do not have access to contextmachine secrets.
# All services used by mmcore can be run locally, see general documentation for details.


import base64
import os
import pprint
import shutil
import subprocess

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

        self.__setup_repo__()
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
        if USE_REPO:
            if self.update or not os.path.isfile(self.env_file_name):
                # Yes, this may be excessive.
                # But I prefer to check this case to avoid having to routinely delete the directory in some cases.
                shutil.rmtree(f"{self.repo_name}", ignore_errors=True)

                proc = subprocess.Popen(["git", "clone", self.secrets_link])
                proc.wait()
                shutil.move(f"{self.repo_name}/{self.env_file_name}", self.env_file_name)
                shutil.rmtree(f"{self.repo_name}", ignore_errors=True)

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

        return {
            "REDISHOST": "c-c9q1muil9vsf3ol4p3di.rw.mdb.yandexcloud.net",
            "REDISPORT": 6380,
            "REDISPASSWORD": "caMbuj-tabxy1-pikkij",
            "AWS_ACCESS_KEY_ID": "***REMOVED***",
            "AWS_SECRET_ACCESS_KEY": "YCN4wjUY7RY25f_EAG9aL-muke8LTo0tgCCcBhN4",
            "DEFAULT_REGION": "ru_center_1",
            "RHINO_COMPUTE_URL": "79.143.24.246",
            "RHINO_COMPUTE_PORT": 8080,
            "RHINO_COMPUTE_API_KEY": "84407047-8380-441c-9c76-a07ca394b88e",
            "RHINO_COMPUTE_GH_DEPLOY_PATH": "c:/users/administrator/compute-deploy",
            "CADEX_LICENSE": "QENVU1RPTUVSPWZhdGFoaTI3MDdAaGVtcHlsLmNvbSBAQ1VTVE9NRVJfQVBQPUVWQUwgQFBST0RVQ1Q9QU5ZIEBPUz1BTlkgQENPTkZJR1VSQVRJT049QU5ZIEBWRVJTSU9OPUFOWSBAU0VSSUFMX05VTUJFUj1ETUNILTM0WEItRFVYUS1NM1NTIEBMQU5HVUFHRT1QWVRIT04gQFVQR1JBREVfRVhQSVJBVElPTj0yMDIyMTIwNSBAVVNFX0VYUElSQVRJT049MjAyMjEyMDUgQFZFTkRPUj1odHRwOi8vd3d3LmNhZGV4Y2hhbmdlci5jb20gQFdBUk5JTkc9VGhpcyBsaWNlbnNlIGtleSBtYXkgbm90IGJlIGRpc2Nsb3NlZCB0byBhbnkgdGhpcmQgcGFydHkgd2l0aG91dCBwcmlvciB3cml0dGVuIHBlcm1pc3Npb24gZnJvbSB2ZW5kb3IuIFRLTFFJOUJHMUVHMlVYRE9VT0xVVjdLUDlOV1k0NDBZTzE0TkJHM1NBSlU0R044UUNUMTdITENRTzg0VENMN0gyQjRUN05NSE5XMUowQUFDSkJaR0NDTjBLUlREU0dBVjhVT0ZIUURLM0U0MzRFVjQ1QUtIMjdIM0ZSNzFLNUlUNEhPOUNGQThCRDE5NVA2TThEVTVNT0xZNkM3MUs1SVQ0SE9aN1VBNDJFNkVQTU0wNkdWQlVCUjNVQzdESVo3U0dPNkJIQ1M0MERQM0xWOVM4REcxSUFDOExCRURGWjdTR082VkJTSkNCN0hOQ0szMDhHSkJFUlhEVlFVNktFTko0QlBWSUFNQzlWVTM2SkYyRzBFRkdNUU9OVlAzNUpQTkhXRlU3UUoyOVQxNjdTQk9EREk4VlRWSTIwUkJKTklSNkFGNjI0V0c5VzZRMTAzU0RCTzlJVUFQSEJTSElYV0Q0S0Y0OU1OWldQVDBZTDc2OTM3VUdKS0FFQ0o3RUE2TjZMS1pGTUQ0UUNNWktBNFBQTzg5SUFGUTBHS0QwRVUyMk9CSTNNQTFBNkZRREpHMkNPWExZTUhLRkNWQzRMRzZRMkpTWUJHMUpQVEI0MUhOMkVJQVNISjFEUEwyQTBXNlQ3SVRGRk9BQVQ4NEtFMlRTQklQUzFIVElNU1I5UUFCNjk2RTQ1WFU2SVVBNk9JM0EwMzNFRVBTOE83TTJRNVFGRTZPNUNONTlGMjg1TjdIM0VXNzU5S0hDQVY2SUg3NDUxSDZFVlU2OFQ3NTlKR1VHRERHMDFJSTFBMDYzQkFWRlJJODI0MjZaTUdYQzZaOEdGWE8wSFdMWEdGMTc2Q01EQUFXVkRJVVpaOFI1NzAxMkRQSVVMM1JIWlNPTExSWVYxVVFVVDFVUzk=",
            "MAINTAINER": "CONTEXTMACHINE"
            }
