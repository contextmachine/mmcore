# Installation

Before we begin, ensure that the `mmcore` library is installed.

## Docker

Preferred method of installation is docker.

```bash
docker pull ghcr.io/contextmachine/mmcore:main
```

You can use dev-container with `mmcore` during development. Or build images of your own applications and services for
production.

## Poetry

The second fine way of installing it, assuming you're using poetry.
To do this, add this line to your `pyproject.toml`.

```toml
[tool.poetry.dependencies]
...
mmcore = { path = "../mmcore" }
```

And then just install using poetry.

```bash
poetry install
```

## Pip

Also, you can install it using pip.

```bash
python3 -m pip install git+https://github.com/contextmachine/mmcore.git

```

> Use a new environment. `mmcore` includes some C extensions that have dependencies on libraries such as `openmp`,
> duplication of which in the environment can cause troubles.


