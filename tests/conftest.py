import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="session")
def data():
    from quanara.data_loading import load_all
    return load_all()


@pytest.fixture(scope="session")
def analytic(data):
    from quanara.integration import build_analytic
    return build_analytic(data)


@pytest.fixture()
def tmp_registry(tmp_path, monkeypatch):
    """Isolated model/scenario folders so tests never touch the reference artefacts."""
    from quanara import config, training
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "ACTIVE_MODEL_FILE", tmp_path / "models" / "active_model.json")
    monkeypatch.setattr(config, "SCENARIOS_DIR", tmp_path / "scenarios")
    monkeypatch.setattr(training, "ABLATIONS_DIR", tmp_path / "ablations")
    return tmp_path


@pytest.fixture(autouse=True)
def _protect_active_model():
    """Tests must never leave the app's real active model changed (it is shared state the user sees)."""
    from quanara import config
    f = config.ACTIVE_MODEL_FILE
    before = f.read_text() if f.exists() else None
    yield
    after = f.read_text() if f.exists() else None
    if after != before:
        if before is None:
            f.unlink(missing_ok=True)
        else:
            f.write_text(before)
        pytest.fail(f"test changed the real active model: {before} -> {after} (restored)")
