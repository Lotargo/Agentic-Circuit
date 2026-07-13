# MVP validation

This branch exists to validate the accumulated main-branch implementation through GitHub Actions.

Checks expected:

- Python package installation and pytest
- TypeScript typecheck and production build
- Docker Compose configuration validation
- Python and TypeScript Docker image builds
- HTTP smoke test through TS gateway to Python engine with a mocked OpenAI-compatible provider
