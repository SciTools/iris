import sys
import types

import pytest


@pytest.fixture(autouse=True, scope="module")
def fake_stratify():
    try:
        import stratify
    except:
        fake = types.ModuleType("iris.experimental.stratify")
        fake.interpolate = lambda *a, **k: None
        sys.modules["iris.experimental.stratify"] = fake
        yield
    finally:
        # Remove fake after tests in this module complete
        sys.modules.pop("iris.experimental.stratify", None)
