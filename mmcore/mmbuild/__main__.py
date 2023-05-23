import dataclasses
import json
import os
import sys

import yaml
import dotenv
import rich
import jinja2
from mmcore.utils.termtools import ColorStr, TermColors, MMColorStr
DIRECTORY = "/".join(__file__.split("/")[:-1])
dotenv.load_dotenv(dotenv_path="~/env/.env")
import mmcore
from mmcore.base import DictSchema
def stringify_kube_env(name, val):
    return f"""- name: {name}
    value: {val}"""


class AppSpec:
    def __init__(self, build_dir=f"{os.getcwd()}/.mm", title=f"Generate with mmcore {mmcore.__version__()}"):
        self.build_dir = build_dir
        self.title=title
        self._jinja_env = jinja2.Environment()
        # language=Jinja2
        self.kube_template = jinja2.Template("""{{title}}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{app.name}}
  labels:
    app: {{app.name}}
spec:
  replicas: {{app.kube.replicas}}
  selector:
    matchLabels:
      app: {{app.name}}
  template:
    metadata:
      name: {{app.name}}
      labels:
        app: {{app.name}}
    spec:
      containers:
        - name: {{app.name}}-container
          # language=Jinja2
          env:  {% for key, value in env.items() %}
            - name: {{key}}
              value: {{value}} {% endfor %}
          image: {{app.registry}}/{{app.name}}
          imagePullPolicy: {{app.kube.policies.pull}}
      restartPolicy: {{app.kube.policies.restart}}

---
apiVersion: v1
kind: Service
metadata:
  name: {{app.name}}
spec:
  selector:
    app: {{app.name}}
  ports:
    - protocol: TCP
      port: {{app.port}}
      targetPort: {{app.port}}
  type: {{app.kube.type}}

        """)
        # language=Bash
        self.build_template = jinja2.Template("""{{title}}
docker buildx build --tag {{app.registry}}/{{app.name}} --platform linux/arm64 --platform linux/amd64 --push .
        """)

        # language=Bash
        self.dockerfile_template = jinja2.Template("""{{title}}
FROM python
WORKDIR /app
COPY . .
RUN {{app.dockerfile.run}}
EXPOSE {{app.port}}
ENTRYPOINT {{stringify(app.dockerfile.entrypoint)}}
        """)

    @property
    def env(self):
        return dotenv.dotenv_values(dotenv_path=f"{os.getenv('HOME')}/env/.env")

    @property
    def app(self):
        with open("config.yaml", "r") as config_stream:
            return DictSchema(yaml.load(config_stream, yaml.BaseLoader)).get_init_default().app
            # print("\n\nconfig:\n")
            # rich.print_json(json.dumps(data))

    @property
    def build(self):
        return self.build_template.render(app=self.app, env=self.env, title=self.title)

    @property
    def dockerfile(self):
        return self.dockerfile_template.render(app=self.app, env=self.env, stringify=json.dumps, title=self.title)

    @property
    def kube(self):
        return self.kube_template.render(app=self.app, env=self.env, stringify=stringify_kube_env, title=self.title)

    @property
    def spec(self):
        return {
            "app": dataclasses.asdict(self.app),
            "env": self.env
        }

    def dump(self, to_stdout=False, print_spec=True):
        if print_spec:
            rich.print_json(json.dumps(self.spec))

        if to_stdout:
            with open(f"{self.build_dir}/spec.json", "w") as sys.stdout:
                sys.stdout.write(json.dumps(self.spec, indent=2))
            with open(f"{self.build_dir}/Dockerfile", "w") as sys.stdout:
                sys.stdout.write(self.dockerfile)
            with open(f"{self.build_dir}/kube.yaml", "w") as sys.stdout:
                sys.stdout.write(self.kube)
            with open(f"{self.build_dir}/build.sh", "w") as sys.stdout:
                sys.stdout.write(self.build)

        else:
            import shutil
            shutil.rmtree(self.build_dir, ignore_errors=True)
            os.mkdir(self.build_dir)
            with open(f"{self.build_dir}/spec.json", "w") as specfile:
                specfile.write(json.dumps(self.spec, indent=2))
            with open(f"{self.build_dir}/Dockerfile", "w") as dockerfile:
                dockerfile.write(self.dockerfile)
            with open(f"{self.build_dir}/kube.yaml", "w") as kube:
                kube.write(self.kube)
            with open(f"{self.build_dir}/build.sh", "w") as build:
                build.write(self.build)



if __name__=="__main__":
    app_spec=AppSpec()
    app_spec.dump()