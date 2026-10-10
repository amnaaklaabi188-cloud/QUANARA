# Observed-data methodology

## Definitions and dictionary verification

`Guests` is the recorded total guests for a date and, for international data, nationality. It is not a new-arrival count or an independently verified guest-night measure. `New Arrivals` identifies check-ins. `Same-Day Guests` refers to guests departing the same day. Nationality describes hotel guests; departure country describes a route origin. These populations cannot be joined as equivalent individuals or markets without an independently justified mapping.

`Total PAX` includes passengers beyond those terminating in Abu Dhabi. `Total P2P` identifies point-to-point passengers. Transfers and transit are distinct. Seat capacity is a supply measure; load factor is a ratio, not an additive count. The script never sums load factors or weekly frequencies into passenger counts.

The supplied PDF describes labelled test data, largely suppressed flight fields and monthly records. The actual test workbooks omit `Guests`; actual flight records include dates within months from 2023 onward. The dictionary is useful for definitions but its example coverage and availability claims do not override observed workbook structure. Original files remain untouched. Literal `*` tokens, blank cells and invalid numeric entries remain missing rather than becoming zero. Spreadsheet display formatting does not prove a literal asterisk is stored.

## Audit

Every workbook sheet is inventoried with file SHA-256, dimensions, date coverage, target presence, exact duplicate count and candidate-key duplicate count. Hotel keys are date or date–nationality; flight candidate keys include date, departure country/city, arrival city, airline and destination. A zero duplicate count proves uniqueness under the tested keys, not real-world completeness.

Column quality reports record stored types, blanks, literal suppression markers and numeric conversion coverage. Monthly coverage reports distinguish observed dates from calendar days and absent hotel date–nationality keys. Consistency checks flag arrivals above guests, P2P above PAX, capacity exceedance, passenger partition discrepancies and load-factor ratio discrepancies. These flags require interpretation; capacity exceedance can occur for reasons discussed in the dictionary.

## Feature inventory

No original feature-engineering implementation or model inputs were supplied. The model feature inventory is therefore header-only. Offline analysis uses observed `New Arrivals`, `Same-Day Guests`, cabin passenger counts, seat capacities and passenger totals. These analytical columns are not asserted to be trained model features.

Monthly aviation calculations sum reported count/capacity rows across all origins. The flight workbook has mixed reporting regimes: 2022 appears monthly, followed by dated route records. Monthly sums are exploratory observed-row totals; complete aviation coverage and a consistent measurement definition require provider confirmation. No daily disaggregation of 2022 monthly values is performed.

Hotel monthly targets sum daily observed `Guests`. This measures guest-presence days in supplied records; it is neither distinct visitors nor verified overnight stays. International totals include observed nationality rows only. Absent nationality rows are not imputed. Months with missing calendar dates are excluded from monthly analysis; absent nationality keys are reported separately and limit completeness. No nationality-specific aviation correlations are computed.

## Pearson and Spearman

Daily associations are computed separately for domestic totals and each international nationality, using only labelled training records and pairwise-valid numeric observations. At monthly resolution, count/capacity features are compared separately with domestic and international observed hotel totals for intersecting calendar months. Constant features and pairs with fewer than three observations are excluded. Unlabelled test records never enter target correlations.

Outputs include Pearson r, Spearman rho, pairwise n, missing pairs, direction, target, period, scope and aggregation. Pearson quantifies linear association; Spearman uses ranks to quantify monotonic association. Both range from −1 to +1. Naive p-values are included explicitly as **independent-sample calculations**, not reliable significance evidence for autocorrelated, seasonal, trending time series. Multiple comparisons are exploratory and uncorrected. Pooling nationality records in scatter plots is descriptive only; inferential daily correlations remain within nationality.

Contemporaneous arrivals and same-day counts are not automatically available at a forecasting origin. Their relation to `Guests` may partly reflect definitional overlap. Any future forecasting pipeline must specify release timing and horizon, fit transformations on training data, lag unavailable variables, and evaluate chronologically. Correlation does not prove causation, useful future prediction, or causal scenario effects.

## Models, diagnostics and scenarios

No training pipeline, artifact or prediction record is available. MAE, RMSE, WMAPE, R², cross-validation, model ranking, residual diagnostics, predictions and simulator examples cannot be reconstructed. Model-only figure types are deliberately absent. Hyperparameters, training periods and active models must be recovered from source and artifact metadata before reporting them.

For later model recovery, evaluate identical target scopes, granularities and held-out periods. WMAPE requires a nonzero sum of absolute observed outcomes. Record sample counts, temporal splits, preprocessing fitting boundaries and artifact hashes. Scenario sensitivity must compare actual baseline and changed inputs through the same eligible pipeline, and zero change must reproduce baseline. Such sensitivity is not a causal effect estimate.
