import numpy as np
import pytest

from numbench.backends import BackendUnavailable, get_backend, list_backends


def all_backend_names() -> list[str]:
    return [b.name for b in list_backends()]


def load_backend(name: str):
    """Return the backend or skip with the explicit reason it cannot run here."""
    try:
        return get_backend(name)
    except BackendUnavailable as exc:
        pytest.skip(f"backend {name} not runnable on this machine: {exc}")


@pytest.fixture(scope="session")
def reference():
    return get_backend("numpy64")


@pytest.fixture(params=all_backend_names())
def backend(request):
    return load_backend(request.param)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
