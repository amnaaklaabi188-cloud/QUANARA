"""The package rename leaves every saved model usable: each pickled pipeline loads unchanged through the
legacy-name alias and reproduces its saved holdout predictions exactly."""
import hashlib
import sys

import numpy as np
import pandas as pd
import pytest

from quanara import registry
from quanara.integration import model_frame, training_rows

MODELS = registry.list_models()


def test_alias_returns_the_current_modules():
    import quanara.models as qm
    if MODELS:
        registry.load_model(MODELS[0]["model_id"])
    from quanara._legacy import LEGACY
    for old in LEGACY:  # both earlier package names resolve to the very same module objects
        __import__(f"{old}.models")
        assert sys.modules[f"{old}.models"] is qm
    assert qm.__spec__.name == "quanara.models"


@pytest.mark.skipif(not MODELS, reason="reference models not trained")
def test_every_saved_pipeline_reproduces_its_holdout_predictions(analytic):
    frames = {}
    for m in MODELS:
        md = registry.load_metadata(m["model_id"])
        key = (md["scope"], md["mode"])
        if key not in frames:
            frames[key] = training_rows(model_frame(analytic, *key))
        rows = frames[key]
        saved = registry.load_table(m["model_id"], "predictions_holdout")
        hold = rows[rows["Date"] >= pd.Timestamp(md["periods"]["holdout"][0])].reset_index(drop=True)
        assert len(hold) == len(saved), m["model_id"]
        pred = registry.load_model(m["model_id"], "eval").predict(hold)
        assert np.allclose(pred, saved["y_pred"].to_numpy(), rtol=1e-6, atol=1e-6), m["model_id"]


@pytest.mark.skipif(not MODELS, reason="reference models not trained")
def test_artifacts_are_not_rewritten_by_loading():
    d = registry.model_dir(MODELS[0]["model_id"]) / "model_final.joblib"
    before = hashlib.sha256(d.read_bytes()).hexdigest()
    registry.load_model(MODELS[0]["model_id"])
    assert hashlib.sha256(d.read_bytes()).hexdigest() == before
