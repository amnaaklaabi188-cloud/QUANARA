# Future research roadmap

These directions are proposed; none is an implemented or benchmarked capability.

| Direction | Scientific motivation | Validation strategy |
|---|---|---|
| AI-Assisted Forecast Benchmarking | Test whether foundation models, future GPT-family systems and specialized forecasting architectures add predictive information beyond conventional baselines. | Use identical targets, horizons, chronological splits, information availability and metrics; report cost, uncertainty and reproducibility. No GPT model has been benchmarked here. |
| Weather-Aware Forecasting | Temperature, rainfall, visibility, severe weather and disruptions may affect operations and demand. | Use timestamped historical weather and forecasts available at each origin; compare with baseline via temporal ablations and disruption-period evaluation. |
| Events and Seasonal Intelligence | Holidays, conferences, major events and seasonal patterns may explain demand changes. | Use events announced before each forecast origin, rolling evaluation and ablations; test generalization to unseen events and years. |
| Uncertainty-Aware Decision Support | Decisions need calibrated risk estimates as well as point forecasts. | Evaluate interval coverage, interval width, probabilistic scoring and robustness under plausible input shifts on untouched temporal holdouts. |
