import os
import tempfile

# Configure env BEFORE app/config import so paths point at a throwaway dir and
# watsonx is treated as unconfigured.
_tmp = tempfile.mkdtemp(prefix="ragtest_")
os.environ["UPLOAD_FOLDER"] = os.path.join(_tmp, "uploads")
os.environ["VECTOR_DB"] = os.path.join(_tmp, "vector_store")
os.environ["FLASK_SECRET_KEY"] = "test-secret"
for _k in ("IBM_API_KEY", "IBM_URL", "IBM_PROJECT_ID", "IBM_Project_id"):
    os.environ.pop(_k, None)
os.makedirs(os.environ["UPLOAD_FOLDER"], exist_ok=True)

import pytest  # noqa: E402


@pytest.fixture
def appmod():
    import app as appmodule
    appmodule._histories.clear()
    return appmodule


@pytest.fixture
def client(appmod):
    appmod.app.config.update(TESTING=True)
    return appmod.app.test_client()
