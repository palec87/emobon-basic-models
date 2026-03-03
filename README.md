# [Package name]

:: short description::

## Installation

### Using uv (recommended)

```bash
uv add mgnify-methods
```

### Using pip

```bash
pip install mgnify-methods
```

### From source

```bash
git clone https://github.com/palec87/mgnify-methods.git
cd mgnify-methods
uv sync
```

## Development

This project uses `uv` for dependency management and has a `Makefile` for common development tasks.

### Setup development environment

```bash
make dev-install
```

### Run tests

```bash
make test
```

### Run linters

```bash
make lint
```

### Format code

```bash
make format
```

### Build documentation

```bash
make docs
```

### Build package

```bash
make build
```

### Available Make targets

Run `make help` to see all available targets.

## Documentation

Documentation is available at [ReadTheDocs](https://mgnify-methods.readthedocs.io/) (once deployed).

## License

See [LICENSE](LICENSE) file for details.

