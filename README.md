# emobon-models

Basic models for EMOBON metadata and abundance tables.

## Installation

### Using uv (recommended)

```bash
uv add emobon-models
```

### Using pip

```bash
pip install emobon-models
```

### From source

```bash
git clone https://github.com/palec87/emobon-models.git
cd emobon-models
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

Documentation is available at ReadTheDocs (once deployed).

## License

See [LICENSE](LICENSE) file for details.

