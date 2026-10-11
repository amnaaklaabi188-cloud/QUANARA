"""QuanAra — aviation-to-hospitality demand forecasting for Abu Dhabi.

Package layout
--------------
config        paths, constants, modelling calendar
data_loading  locate and read the five competition workbooks (read-only)
schema        expected schemas and validation
cleaning      suppression handling, numeric coercion, frequency decoding
dictionary    field-definition table (Data_Dictionary.pdf + audit annotations)
audit         data-quality audit report
integration   reproducible analytical datasets (daily grain) with a build log
features      feature catalogue and feature construction (shared by training and simulator)
correlation   correlation statistics for the EDA lab
models        model factory (baseline, Ridge, RF, CatBoost, XGBoost, SVR)
training      chronological holdout + expanding-window CV training
evaluation    metrics and ablation study
registry      saving / loading / activating trained pipelines
scenario      What-if simulation engine
copilot       grounded explanations (rule-based; optional LLM)
charts        Plotly figure builders
"""

__version__ = "1.0.0"

from . import _legacy as _legacy

_legacy.install()  # saved model files keep loading after the package rename
