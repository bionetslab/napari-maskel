# Installation

napari-maskel requires Python 3.14+.

## Via napari's plugin manager

With napari already installed and running, open **Plugins → Install/Uninstall Plugins**, search for "napari-maskel", and click **Install**.

## From PyPI

```sh
pip install napari-maskel
```

```sh
uv add napari-maskel
```

## From GitHub (latest unreleased)

```sh
pip install git+https://github.com/bionetslab/napari-maskel.git
```

```sh
uv add git+https://github.com/bionetslab/napari-maskel.git
```

## From a local clone (development)

```sh
git clone https://github.com/bionetslab/napari-maskel.git
cd napari-maskel
uv sync --extra dev   # + test tools
napari
```

To test against an unreleased [maskel](https://bionetslab.github.io/maskel/) checkout instead of its PyPI release, point `uv` at it locally:

```sh
uv sync --extra dev && uv pip install -e ../maskel   # adjust the path
```

or add a local, untracked `uv.toml` with a `[tool.uv.sources]` override for `maskel`.
