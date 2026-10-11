# QuanAra — Data Audit Report
_Generated 2026-10-10T15:43:05 from the supplied files only._

## 1. File inventory (originals unchanged; SHA-256)
| dataset | file | bytes | sha256 |
|---|---|---|---|
| intl_train | data international_train.xlsx | 1935490 | beaa3c39d3f9bf891667fdb50a2584e56a3b02dc3c459b17ea312022415fad44 |
| intl_test | data international_test.xlsx | 255860 | 9e0379f5798c8b861bf11b2694ed911c3f8dd4530713ee94523a4b3f8fbbb931 |
| dom_train | data domestic_train.xlsx | 51710 | 8aade22d507d92ded353de395c6ea413536fb0a0741efc2c4653f6ea804a5208 |
| dom_test | data domestic_test.xlsx | 14356 | 3eb97c30bf25a84cfe7ffb2df76af8fed57785c2bc4c16a229974e7e9b513e37 |
| flight | flight_data.xlsx | 15722937 | 5520c0a95126636ee90084209c8e4110bd0b0b095d64cc9dbac37356ae00a9f8 |
| dictionary | Data_Dictionary.pdf | 338446 | a059e4fbec0079fe5722ff007883a8a031c1989827e439b732fe64ffcc0c2c67 |

## 2. Hotel datasets

| dataset | rows | cols | start | end | dates | missing days | dup keys | target (Guests) | nationalities | NA<10 rows | Same-Day suppressed % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| intl_train | 58,622 | 6 | 2022-01-01 | 2025-07-31 | 1308 | 0 | 0 | yes | 45 | 5202 | 59.7 |
| intl_test | 9,202 | 5 | 2025-08-01 | 2026-02-28 | 212 | 0 | 0 | **absent** | 45 | 0 | 47.8 |
| dom_train | 1,308 | 5 | 2022-01-01 | 2025-07-31 | 1308 | 0 | 0 | yes | — | 0 | 0.0 |
| dom_test | 212 | 4 | 2025-08-01 | 2026-02-28 | 212 | 0 | 0 | **absent** | — | 0 | 0.0 |

- **intl_train** columns: `Date, Guests, New Arrivals, Same-Day Guests, Nationality, Residence (groups)`; missing: {'Date': 0, 'Guests': 0, 'New Arrivals': 262, 'Same-Day Guests': 35026, 'Nationality': 0, 'Residence (groups)': 0}
- **intl_test** columns: `Date, New Arrivals, Same-Day Guests, Nationality, Residence (groups)`; missing: {'Date': 0, 'New Arrivals': 2, 'Same-Day Guests': 4402, 'Nationality': 0, 'Residence (groups)': 0}
- **dom_train** columns: `Date, Guests, New Arrivals, Same-Day Guests, Residence (groups)`; missing: {'Date': 0, 'Guests': 0, 'New Arrivals': 0, 'Same-Day Guests': 0, 'Residence (groups)': 0}
- **dom_test** columns: `Date, New Arrivals, Same-Day Guests, Residence (groups)`; missing: {'Date': 0, 'New Arrivals': 0, 'Same-Day Guests': 0, 'Residence (groups)': 0}

- International Guests range 10–6804; implied average length of stay ΣGuests/ΣNew Arrivals = 3.61 (intl), 2.46 (domestic).
- Missing nationality-day combinations: {'ARMENIA': 24, 'AZERBAIJAN': 42, 'CZECHIA': 27, 'DENMARK': 42, 'FINLAND': 54, 'ISRAEL': 4, 'NORWAY': 44, 'QATAR': 1}

## 3. Flight operations dataset

- 117,608 rows × 26 columns; 2022-01-01 → 2026-02-28; duplicate keys: 0; negative values: 0.
- **Grain detected from data:** {"daily": {"rows": 116395, "start": "2023-01-01", "end": "2026-02-28", "distinct_dates": 1155}, "monthly": {"rows": 1213, "start": "2022-01-01", "end": "2022-12-01", "distinct_dates": 12}}
- 33 departure countries, 124 cities, 51 airlines, 250 routes (daily data). Arrival City = ['Abu Dhabi'], Destination = ['AUH'].
- Average Weekly Frequency decodes to daily flights (AWF × days_in_month / 7); max decode error 8.9e-16. Distribution of daily flights per route-day: {1: 74315, 2: 25245, 3: 8190, 4: 7544, 5: 790, 6: 309, 8: 2}.
- Identities: PAX = P2P + Transfer + Transit on 100.00% of rows; Seats = Σcabin capacity on 100.00% of daily rows; P2P = Σcabin P2P on only 28.37%; Transfer = Σcabin transfer on 58.59%.
- Load Factor equals PAX/Seats (max abs diff 1.1e-16); > 100% on 8.49% of daily rows (max 107.6%) and 1.07% of monthly rows (max 185.7%).
- Total Transit non-zero rows: daily 3, monthly 4. P2P share of daily PAX 50.6%, transfer share 49.4%.
- Missing values, monthly rows: {'Average Weekly Frequency': 1213, 'Business Class P2P Count': 1213, 'Business Class Seat Capacity': 1213, 'Economy Class P2P Count': 1213, 'Economy Class Seat Capacity': 1213, 'First Class P2P Count': 1213, 'First Class Seat Capacity': 1213, 'Transfer Business Class Count': 1213, 'Transfer Economy Class Count': 1213, 'Transfer First Class Count': 1213, 'Transit Business Class Count': 1213, 'Transit Economy Class Count': 1213, 'Transit First Count': 1213, 'flights': 1213, 'flights_decode_error': 1213}
- Missing values, daily rows: {'Total PAX Excluding Infant': 1, 'Transfer Business Class Count': 1, 'Transfer Economy Class Count': 1, 'Transfer First Class Count': 1}

## 4. Market matching (Nationality ↔ Departure Country Name)

- 33 of 45 nationalities match a departure country by name, covering 92.18% of international guest-days.
- Matched: ARMENIA, AUSTRIA, AZERBAIJAN, BAHRAIN, BELGIUM, CANADA, CHINA, EGYPT, FRANCE, GERMANY, INDIA, IRELAND, ISRAEL, ITALY, JAPAN, JORDAN, KAZAKHSTAN, KUWAIT, LEBANON, NETHERLANDS, OMAN, PHILIPPINES, POLAND, QATAR, RUSSIAN FEDERATION, SAUDI ARABIA, SOUTH KOREA, SPAIN, SWITZERLAND, TURKEY, UNITED KINGDOM, UNITED STATES OF AMERICA, UZBEKISTAN
- Nationalities without a departure country: AUSTRALIA, BRAZIL, CZECHIA, DENMARK, FINLAND, MEXICO, MOROCCO, NORWAY, PAKISTAN, ROMANIA, SOUTH AFRICA, SWEDEN
- Departure countries without a nationality: none

## 5. Data Dictionary vs. delivered files

- Flight data is documented as monthly; it is monthly only for 2022 and DAILY from 2023-01-01.
- Average Weekly Frequency is documented as mostly '*'; it is populated on 100% of daily rows and decodes exactly to daily flight counts.
- Cabin-class P2P/seat/transfer columns are documented as predominantly '*'; they are empty only on 2022 monthly rows.
- Suppression is documented as '*'; the .xlsx files contain empty cells instead (zero literal '*' cells).
- Load Factor is documented as a percentage string; it is stored as a numeric fraction.
- Dates are documented as strings; they are stored as Excel datetimes.
- International test coverage is documented as 'Aug 2025 – Aug 2025'; the file covers 2025-08-01 → 2026-02-28 (212 days).
- Guests is documented as the target in the test files; both test files do not contain a Guests column.
- The dictionary states 32 unique columns; the five files contain 31 distinct column names (Date is shared).

## 6. Leakage register
| feature | risk | decision |
|---|---|---|
| Guests (same day / recent lags) | target leakage | Never a feature. Short lags (t-1, t-7) are unavailable for the test period and for any What-if date after 2025-07-31, so they are not offered. Only the 364-day seasonal lag is used. |
| New Arrivals | accounting component of Guests (subset, same day) | Competition mode only (the official test files provide it). Never used in What-if mode. |
| Same-Day Guests | same-day hotel observation | Competition mode only. |
| Total PAX / Load Factor / P2P share | deterministic identities (PAX = P2P + Transfer + Transit; LF = PAX / Seats) | Allowed, but the simulator recomputes all of them jointly so they never contradict each other. |
| Rolling / lagged aviation features | look-ahead if windows were centred | All windows are trailing (end at day t). The scaler/encoder is fitted on training folds only (inside the sklearn Pipeline). |
| Holdout period | temporal leakage via random K-fold | Only chronological holdout and expanding-window CV are used. |

## 7. Warnings

- **[critical] Official test files contain no observed Guests** — Out-of-sample metrics can only be computed on held-out TRAIN dates (chronological holdout). Test-period outputs are forecasts that cannot be scored here.
- **[critical] Flight data changes grain in 2023** — 2022 is monthly (1213 rows); daily data starts 2023-01-01. Daily models therefore use 2023-03-01 → 2025-07-31 (after a 56-day feature warm-up).
- **[warning] Test extract filtered differently from train** — 5,202 intl-train rows have New Arrivals < 10; the intl-test file has 0. Competition-mode market models are trained on rows with New Arrivals ≥ 10 to match the test population.
- **[warning] Some nationality-day combinations are missing** — 8 nationalities have days without a row (international Guests minimum is 10); suppression of small counts is possible but not confirmed. Missing rows are not imputed.
- **[warning] Departure country ≠ nationality** — 33 of 45 nationalities share a name with a departure country (92.2% of international guest-days). The name match is a modelling assumption, not an identity.
- **[info] Load factor above 100%** — 8.5% of daily route rows exceed 100% (max 107.6%); kept as documented (overbooking / infant counting).
- **[info] Total Transit is effectively zero in daily data** — Non-zero on only 3 of 116,395 daily rows (<0.001% of PAX); transit features carry no usable information and transit controls are disabled.
- **[info] Same-Day Guests suppression is ambiguous** — 59.7% of intl-train rows are '*' (zero, suppressed or n/a — indistinguishable).

## 8. Field-definition table
| column | source | definition | dtype_units | missing_meaning | role | known_whatif | known_competition | scenario_editable | audit_finding |
|---|---|---|---|---|---|---|---|---|---|
| Date | intl train/test, dom train/test | Calendar date to which the guest counts apply (documented as string YYYY/M/D). International: one row per nationality per date. Domestic: exactly one row per date. | date (stored as Excel datetime) | never missing | key | True | True | False | Stored as real Excel datetimes, not strings. Daily, no gaps: train 2022-01-01→2025-07-31 (1,308 days), test 2025-08-01→2026-02-28 (212 days). |
| Guests | intl train, dom train | TARGET VARIABLE. Total number of guests recorded for the nationality (international) or in total (domestic) on the given date. | integer, persons recorded on the date | never missing in train; column ABSENT from both test files | target | False | False | False | Absent from both official test files (no observed target to score against). International minimum is exactly 10, and 238 nationality-day combinations (8 nationalities) have no row; suppression of small counts is a possible but unconfirmed explanation. Because New Arrivals (check-ins) are a subset that excludes continuing stays, Guests counts everyone recorded on the date; summed over days it approximates guest-nights (same-day visitors are included). |
| New Arrivals | intl train/test, dom train/test | Guests newly arrived (check-in) on the date, as opposed to continuing stays. Always a subset of Guests. | integer, persons | empty cell = suppressed/unavailable (262 intl-train rows, 2 intl-test rows) | predictor (competition mode only) | False | True | False | Verified ≤ Guests on every train row. Strong accounting link to Guests → allowed only in Competition mode, where the test files supply it. Test rows never have New Arrivals < 10, while 8.9% of intl-train rows do: the test extract was filtered differently (distribution shift). |
| Same-Day Guests | intl train/test, dom train/test | Guests who visited and departed on the same day without an overnight stay. International: '*' indicates zero, suppressed, or not applicable. Domestic: documented as always numeric. | integer, persons | empty cell = '*' = zero OR suppressed OR n/a (cannot be distinguished) | predictor (competition mode only) | False | True | False | Intl: 59.7% of train rows and 47.8% of test rows suppressed. Domestic: never missing (matches dictionary). Verified ≤ Guests and ≤ New Arrivals. |
| Nationality | intl train/test | Passport nationality / country of origin of the guests, uppercase ISO-style names. 45 distinct nationalities. | categorical (45 levels) | never missing | key / market identifier | True | True | False | 45 nationalities, identical sets in train and test. 33 share a name with a flight 'Departure Country Name'; nationality is NOT the same concept as departure country. |
| Residence (groups) | intl train/test, dom train/test | High-level residence group: 'International' (non-resident visitor) in the international files, 'Domestic' (resident / in-country) in the domestic files. | categorical (constant per file) | never missing | constant (excluded) | True | True | False | Constant within each file → carries no predictive information; used only to label scope. |
| Date | flight | First day of the reporting month (e.g. '1/1/2022 12:00:00 AM' = January 2022) — documented as MONTHLY. | date | never missing | key | True | True | False | Granularity differs from the dictionary: 2022 = monthly (1 date per month, the 1st; 1,213 rows); 2023-01-01→2026-02-28 = DAILY (1,155 distinct days; 116,395 rows). |
| Departure Country Name | flight | Full name of the country from which the flight departs. | categorical (33 levels) | never missing | market identifier / aggregation key | True | True | True | 33 countries; all 33 match a guest nationality by name. A departure country is where the flight originates, not the passengers' nationality (e.g. UK departures carry many non-UK nationals and 72% transfer traffic). |
| Departure City | flight | City of origin; some countries have several departure cities. | categorical (124 levels in daily data) | never missing | derived predictor (count of active cities) | True | True | False | Too granular to join to hotel data (no city in hotel files). Used via a derived 'active departure cities' count. |
| Arrival City | flight | Destination city; 'Abu Dhabi' in all observed records. | categorical | never missing | constant (excluded) | True | True | False | Verified constant = 'Abu Dhabi' on all 117,608 rows. |
| Airline Name | flight | Operating airline; 'Unknown' for a few unidentified records. | categorical (51 levels in daily data) | 'Unknown' label (4 rows) | derived predictor (airline count, Etihad seat share) | True | True | False | Cannot be a direct feature at daily-aggregate grain (many airlines per day). Represented by active-airline count and Etihad Airways seat share. |
| Average Weekly Frequency | flight | Average number of flights per week on the route during the reporting month. Dictionary: 'Populated with * in most observed rows … may not be available'. | float, flights/week | empty on all 1,213 monthly (2022) rows | predictor (decoded to daily flights) | True | True | True | Contradicts the dictionary: populated on 100% of daily rows. Value = flights_that_day × 7 / days_in_month exactly (max decode error 9e-16), giving an integer daily flight count (1–8). Frequency is therefore a validated feature for daily data. |
| Business Class P2P Count | flight | Point-to-point Business Class passengers (Abu Dhabi is their final destination). * = not available. | integer, passengers | empty on all 2022 monthly rows | excluded (approximate) | True | True | False | Populated on all daily rows. Cabin P2P counts sum to Total P2P on only 28.4% of rows (shortfall 1–28 pax, consistent with infants not being assigned a cabin). Totals are used instead. |
| Business Class Seat Capacity | flight | Business Class seats available on the route for the month. * = n/a (e.g. no Business cabin). | integer, seats | empty on all 2022 monthly rows | derived predictor (business seat share) | True | True | False | Daily rows: cabin capacities sum exactly to Total Seats on 100% of rows. 58.6% of daily rows have Business capacity > 0. |
| Economy Class P2P Count | flight | Point-to-point Economy Class passengers for the route. | integer, passengers | empty on all 2022 monthly rows | excluded (approximate) | True | True | False | See Business Class P2P Count (cabin P2P ≠ Total P2P on 71.6% of rows). |
| Economy Class Seat Capacity | flight | Economy Class seats available on the route. | integer, seats | empty on all 2022 monthly rows | component of Total Seats | True | True | False | Exact component of Total Seats on daily rows. |
| First Class P2P Count | flight | Point-to-point First Class passengers; predominantly * (few routes offer First). | integer, passengers | empty on all 2022 monthly rows; zero where no First cabin | excluded (approximate) | True | True | False | Daily rows are populated (zeros, not '*', where there is no First cabin). |
| First Class Seat Capacity | flight | First Class seats available; predominantly * as most airlines do not operate First. | integer, seats | empty on all 2022 monthly rows | component of Total Seats | True | True | False | First capacity > 0 on only 5.4% of daily rows. |
| Load Factor | flight | Share of seat capacity occupied: Total PAX / Total Seats. Documented as a percentage string; values above 100% possible (overbooking / infant counting). | ratio (stored as a fraction, e.g. 0.4702 = 47.02%) | never missing | predictor (recomputed after aggregation) | True | True | True | Stored as a numeric fraction, not a % string. Equals Total PAX / Total Seats to 1e-16 on every row. > 100% on 8.5% of daily rows (max 107.6%) and 1.1% of 2022 rows (max 185.7%). Aggregates are recomputed as ΣPAX / ΣSeats — never averaged. |
| Total P2P | flight | Point-to-point passengers across all cabins: journey originates at the departure city and terminates at Abu Dhabi. | integer, passengers | never missing | key predictor | True | True | True | The aviation quantity closest to potential hotel demand — but P2P includes residents returning home, VFR travellers and day visitors, so P2P ≠ hotel guests. 50.6% of daily PAX are P2P. |
| Total PAX | flight | All passengers on board across classes, including P2P and transfer/transit. | integer, passengers | never missing | predictor | True | True | True | Identity verified on 100% of rows: Total PAX = Total P2P + Total Transfer + Total Transit. |
| Total PAX Excluding Infant | flight | Total passengers minus lap infants; typically 1–5 below Total PAX. | integer, passengers | 1 daily row missing | excluded (near-duplicate of Total PAX) | True | True | False | Median 2 below Total PAX (max 31); never above it. Redundant with Total PAX. |
| Total Seats | flight | Total seat capacity offered across all cabins; the Load Factor denominator. | integer, seats | never missing (never zero) | key predictor (scenario lever) | True | True | True | Exactly equals the sum of cabin capacities on daily rows. Mean 224 seats per decoded flight. |
| Total Transfer | flight | Passengers who arrived on this route and transferred to another flight at Abu Dhabi (change of aircraft / flight number). | integer, passengers | never missing | predictor (distinguished from P2P) | True | True | True | 49.4% of daily PAX. Transfer passengers connect onward and are not expected to stay in Abu Dhabi hotels (unless on a stopover — not identifiable in the data). |
| Total Transit | flight | Passengers in transit through Abu Dhabi (same aircraft / simpler connection). Mostly 0. | integer, passengers | never missing | excluded (near-zero variance) | True | True | False | Zero on all but 3 of 116,395 daily rows (164 passengers, <0.001% of PAX) and non-zero on 4 monthly 2022 rows. No usable information → transit controls are disabled in the simulator. |
| Transfer Business Class Count | flight | Business Class passengers who transferred at Abu Dhabi. * = not available. | integer, passengers | empty on 2022 rows + 1 daily row | excluded (approximate) | True | True | False | Cabin transfer counts sum to Total Transfer on 58.6% of rows only. |
| Transfer Economy Class Count | flight | Economy Class passengers who transferred at Abu Dhabi. * = not available. | integer, passengers | empty on 2022 rows + 1 daily row | excluded (approximate) | True | True | False | See Transfer Business Class Count. |
| Transfer First Class Count | flight | First Class passengers who transferred at Abu Dhabi. * = not available. | integer, passengers | empty on 2022 rows + 1 daily row | excluded (approximate) | True | True | False | See Transfer Business Class Count. |
| Transit Business Class Count | flight | Business Class passengers in transit at Abu Dhabi. * = not available. | integer, passengers | empty on 2022 rows | excluded (inconsistent) | True | True | False | Cabin transit counts are > 0 on 3.6% of daily rows although Total Transit is zero on all but 3 rows — internally inconsistent; excluded. |
| Transit Economy Class Count | flight | Economy Class passengers in transit at Abu Dhabi. * = not available. | integer, passengers | empty on 2022 rows | excluded (inconsistent) | True | True | False | See Transit Business Class Count. |
| Transit First Count | flight | First Class passengers in transit (note the abbreviated name vs. the 'Class Count' pattern). * = not available. | integer, passengers | empty on 2022 rows | excluded (inconsistent) | True | True | False | Column name confirmed exactly as the dictionary warns ('Transit First Count'). |
| Destination | flight | IATA code of the arrival airport; 'AUH' in all observed records. | categorical | never missing | constant (excluded) | True | True | False | Verified constant = 'AUH' on all rows. |
