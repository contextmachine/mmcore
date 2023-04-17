import dataclasses
import os

import cxm

from mmcore import load_dotenv_from_path

load_dotenv_from_path(".env")
import requests

STORAGE = bool(os.getenv("STORAGE"))
BUCKET = bool(os.getenv("BUCKET"))
VERBOSE = bool(os.getenv("CXM_VERBOSE"))
API = os.getenv("CXM_VIEWER_API_ENDPOINT")
if VERBOSE:
    print(f"using viewer api endpoint: {API}")

API = "https://api.contextmachine.online"

sess = cxm.S3Session(bucket=BUCKET)


@dataclasses.dataclass
class SceneView:
    name: str
    includes: list[str] = ()
    endpoint: str = API
    _api_data = None

    def asdict(self):

        return {
            "includes": self.includes,
            "metadata": {
                "type": "scene",
                "name": self.name,
                "default_view": "ortho"
            },
            "chart": {
                "headers": [
                    "string"
                ]
            }
        }

    def __post_init__(self):
        try:
            self.update()
        except requests.exceptions.JSONDecodeError:
            self.commit()
        except KeyError:
            self.commit()

    def update(self):
        self._api_data = requests.get(f"{self.endpoint}/scenes/{self.name}").json()
        self.name = self._api_data["metadata"]["name"]
        self.includes = self._api_data["includes"]

    def commit(self):
        self._api_data = requests.put(f"{self.endpoint}/scenes/create", json=self.asdict()).json()[self.name]
        self.name = self._api_data["metadata"]["name"]
        self.includes = self._api_data["includes"]


class ViewerApi:
    item_prefix: str = "tmpj"

    def __init__(self, endpoint: str = API):
        self.endpoint = endpoint
        self.streams = []

    def put_scene(self, name, includes):
        return SceneView(name=name, includes=includes, endpoint=self.endpoint)

    def get_scene(self, name):
        return SceneView(name=name, endpoint=self.endpoint)

    def scenes(self):
        return [SceneView(name=scene, endpoint=self.endpoint) for scene in
                requests.get(f"{self.endpoint}/scenes").json()]

    def items(self):
        return [item_name for item_name in requests.get(f"{self.endpoint}/get_keys").json()]

    def del_item(self, name):
        return sess.delete_object(Key=f"{self.item_prefix}/{name}")
    def put_item(self, name, data: str | bytes):
        """

        @param name: name of object without prefixes. eg: test-object
        @param data: str or bytes (json)
        @return: dict: regular response for you s3 server as python dict
        """
        return sess.put_object(Key=f"{self.item_prefix}/{name}", Body=data)

    def get_item(self, name):
        return requests.get(f"{self.endpoint}/get_part/{name}").json()

    def include_stream(self, name, create_scene=False):
        requests.post(f"{self.endpoint}/include_stream/{name}").json()

        self.streams.append(name)
        if create_scene:
            scene_name = f"stream-{name}"
            self.create_scene(scene_name, [name])

            return SceneView(name=scene_name, endpoint=self.endpoint)

    def exclude_stream(self, name):
        requests.post(f"{self.endpoint}/exclude_stream/{name}").json()
        self.streams.remove(name)
