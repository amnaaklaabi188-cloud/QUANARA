"""Audit authorized local inputs; never copy workbooks into public outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/quanara-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/quanara-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def numeric_clean(values):
    """Suppressed and blank observations remain missing; never impute zero."""
    text = values.astype(str).str.strip()
    return pd.to_numeric(values.mask(text.isin(['*', ''])), errors='coerce')


def correlation_stats(x, y):
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return None
    return pearsonr(x[valid], y[valid]), spearmanr(x[valid], y[valid]), int(valid.sum())


def aviation_fields(columns):
    return [c for c in columns if c.endswith(('Count', 'Capacity')) or c.startswith('Total ')]


def run(inputs, output):
    results = output / 'docs/results'
    figures = output / 'docs/figures'
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    inventory, columns, correlations, frames = [], [], [], {}
    expected = {'data domestic_train.xlsx', 'data domestic_test.xlsx', 'data international_train.xlsx', 'data international_test.xlsx', 'flight_data.xlsx'}
    paths = sorted(inputs.rglob('*.xlsx'))
    names = [p.name for p in paths]
    if set(names) != expected or len(names) != len(expected):
        raise ValueError('Supply exactly one of each of the five documented workbook filenames; no additional workbooks.')
    for path in paths:
        book = pd.ExcelFile(path)
        if book.sheet_names != ['Export']:
            raise ValueError(f'Unexpected sheets in {path.name}: {book.sheet_names}')
        for sheet in book.sheet_names:
            frame = pd.read_excel(book, sheet_name=sheet, keep_default_na=False)
            frames[path.name] = frame
            dates = pd.to_datetime(frame['Date'], errors='coerce')
            keys = ['Date', 'Nationality'] if 'Nationality' in frame else ['Date']
            if 'Departure City' in frame:
                keys = ['Date', 'Departure Country Name', 'Departure City', 'Arrival City', 'Airline Name', 'Destination']
            inventory.append(dict(file=path.name, sheet=sheet, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(frame), columns=len(frame.columns), date_min=str(dates.min().date()), date_max=str(dates.max().date()), invalid_dates=int(dates.isna().sum()), target_available='Guests' in frame, duplicate_rows=int(frame.duplicated().sum()), duplicate_keys=int(frame.duplicated(keys).sum()), key_columns=';'.join(keys), nationalities=frame['Nationality'].nunique() if 'Nationality' in frame else None))
            for col in frame:
                raw = frame[col]
                suppressed = raw.astype(str).str.strip().eq('*')
                missing = raw.isna() | raw.astype(str).str.strip().eq('')
                clean = raw.mask(suppressed | missing)
                numeric = pd.to_numeric(clean, errors='coerce')
                columns.append(dict(file=path.name, column=col, dtype=str(raw.dtype), missing=int(missing.sum()), suppressed=int(suppressed.sum()), numeric_values=int(numeric.notna().sum()), unique_values=int(clean.nunique())))
            if 'Guests' not in frame:
                continue
            for scope, group in ([('domestic daily total', frame)] if 'Nationality' not in frame else [(f'international daily nationality: {n}', g) for n, g in frame.groupby('Nationality')]):
                for feature in ['New Arrivals', 'Same-Day Guests']:
                    x = numeric_clean(group[feature])
                    y = pd.to_numeric(group['Guests'], errors='coerce')
                    valid = x.notna() & y.notna()
                    n = int(valid.sum())
                    if n < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
                        continue
                    p = pearsonr(x[valid], y[valid]); s = spearmanr(x[valid], y[valid])
                    correlations.append(dict(feature=feature, pearson_r=p.statistic, spearman_rho=s.statistic, valid_samples=n, missing_observations=len(group)-n, pearson_p_naive=p.pvalue, spearman_p_naive=s.pvalue, direction='positive' if p.statistic>0 else 'negative', target='Guests', period_start=str(pd.to_datetime(group.Date).min().date()), period_end=str(pd.to_datetime(group.Date).max().date()), scope=scope, aggregation='daily', interpretation='contemporaneous observed association; not a forecast feature; serial dependence invalidates naive independent-sample inference'))
    for pdf in sorted(inputs.rglob('Data_Dictionary.pdf')):
        inventory.append(dict(file=pdf.name, sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(), sheet=None, rows=None, columns=None))
    coverage = []
    for name, frame in frames.items():
        dates = pd.to_datetime(frame.Date)
        for month, group in frame.groupby(dates.dt.to_period('M')):
            n_dates = pd.to_datetime(group.Date).nunique()
            n_nationalities = frame.Nationality.nunique() if 'Nationality' in frame else 1
            coverage.append(dict(file=name, month=str(month), rows=len(group), observed_dates=n_dates, calendar_days=month.days_in_month, expected_hotel_rows=month.days_in_month*n_nationalities if 'Departure City' not in frame else None, missing_hotel_keys=month.days_in_month*n_nationalities-len(group) if 'Nationality' in frame else (month.days_in_month-n_dates if 'Departure City' not in frame else None), reporting_note='2022 monthly-looking; 2023 onward dated-route records; completeness not certified' if 'Departure City' in frame else 'daily hotel records; absent rows never imputed'))
    pd.DataFrame(coverage).to_csv(results/'dataset_coverage.csv',index=False)
    fig, axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for name in frames:
        c=pd.DataFrame(coverage).loc[lambda x:x.file==name]
        dates=pd.to_datetime(c.month+'-01')
        if name=='flight_data.xlsx': axes[0].plot(dates,c.rows,label=name,color='#369653')
        else: axes[1].plot(dates,c.missing_hotel_keys,label=name)
    axes[0].set_ylabel('Reported flight rows/month'); axes[0].set_title('Observed input coverage — reporting regimes and absent hotel keys')
    axes[1].set_ylabel('Absent expected hotel keys'); axes[1].set_xlabel('Calendar month'); axes[0].legend(); axes[1].legend()
    fig.tight_layout(); fig.savefig(figures/'dataset_coverage.png',dpi=180); plt.close(fig)
    quality = []
    for name, frame in frames.items():
        if 'Guests' in frame:
            guests = numeric_clean(frame.Guests)
            arrivals = numeric_clean(frame['New Arrivals'])
            quality.append(dict(file=name, check='new_arrivals_exceed_guests', violations=int((arrivals>guests).sum())))
        if 'Total PAX' in frame:
            for label, violations in [('p2p_exceeds_pax',frame['Total P2P']>frame['Total PAX']), ('pax_exceeds_seats',frame['Total PAX']>frame['Total Seats']), ('load_factor_above_one',numeric_clean(frame['Load Factor'])>1), ('passenger_partition_mismatch',frame['Total P2P']+frame['Total Transfer']+frame['Total Transit']!=frame['Total PAX'])]:
                quality.append(dict(file=name,check=label,violations=int(violations.sum())))
            ratio=frame['Total PAX']/frame['Total Seats'].replace(0,np.nan)
            quality.append(dict(file=name,check='load_factor_ratio_mismatch_tolerance_1e-6',violations=int((abs(numeric_clean(frame['Load Factor'])-ratio)>1e-6).sum())))
    pd.DataFrame(quality).to_csv(results/'quality_checks.csv',index=False)
    pd.DataFrame(inventory).to_csv(results/'dataset_inventory.csv', index=False)
    pd.DataFrame(columns).to_csv(results/'column_quality.csv', index=False)
    pd.DataFrame(correlations).to_csv(results/'feature_correlations.csv', index=False)
    # Monthly observed-row sums, all-origin totals; no nationality-to-departure-country mapping.
    flight = frames['flight_data.xlsx'].copy()
    flight['Month'] = pd.to_datetime(flight.Date).dt.to_period('M')
    aviation_columns = aviation_fields(flight.columns)
    for col in aviation_columns:
        flight[col] = numeric_clean(flight[col])
    aviation = flight.groupby('Month')[aviation_columns].sum(min_count=1)
    monthly_corr = []
    for name, frame in frames.items():
        if 'Guests' not in frame: continue
        frame = frame.copy()
        frame['Month'] = pd.to_datetime(frame.Date).dt.to_period('M')
        guests = frame.groupby('Month').Guests.sum(min_count=1).rename('Guests')
        daily_coverage = frame.groupby('Month').Date.nunique()
        complete = daily_coverage.index[daily_coverage.to_numpy() == np.array([m.days_in_month for m in daily_coverage.index])]
        joined = aviation.join(guests, how='inner').loc[lambda x: x.index.isin(complete)]
        scope = 'international observed-nationality monthly total' if 'international' in name else 'domestic monthly total'
        for feature in aviation_columns:
            pair = joined[[feature,'Guests']].dropna()
            stats = correlation_stats(pair[feature], pair.Guests)
            if stats is None: continue
            p, s, _ = stats
            monthly_corr.append(dict(feature=feature, pearson_r=p.statistic,spearman_rho=s.statistic,valid_samples=len(pair), missing_observations=len(joined)-len(pair),pearson_p_naive=p.pvalue,spearman_p_naive=s.pvalue,direction='positive' if p.statistic>0 else 'negative',target='sum of daily Guests (guest-presence days; not arrivals or verified guest nights)',period_start=str(joined.index.min()),period_end=str(joined.index.max()),scope=scope,aggregation='complete hotel calendar month; observed flight rows summed',interpretation='contemporaneous; mixed flight reporting regimes (monthly 2022, dated routes from 2023); observed-row sums only, coverage not certified; unadjusted trend/seasonality; no forecast or causal claim'))
        fig, axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        dates=joined.index.to_timestamp()
        axes[0].plot(dates,joined.Guests,color='#369653'); axes[0].set_ylabel('Sum of daily Guests'); axes[0].set_title(scope+' — observed months; absent nationality keys not imputed')
        axes[1].plot(dates,joined['Total P2P'],label='P2P passengers'); axes[1].plot(dates,joined['Total PAX'],label='All passengers'); axes[1].set_ylabel('Monthly sum of reported passengers'); axes[1].set_xlabel('Month'); axes[1].legend()
        fig.tight_layout(); fig.savefig(figures/('international_trends.png' if 'international' in name else 'domestic_trends.png'),dpi=180); plt.close(fig)
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        for ax, feature in zip(axes,['Total P2P','Total Seats']):
            ax.scatter(joined[feature],joined.Guests,color='#369653')
            ax.set_xlabel(feature+' (reported monthly sum)'); ax.set_ylabel('Sum of observed daily Guests')
        fig.suptitle(f'{scope} — {joined.index.min()}–{joined.index.max()}; contemporaneous')
        fig.tight_layout(); fig.savefig(figures/f'{"international" if "international" in name else "domestic"}_aviation_scatter.png',dpi=180); plt.close(fig)
        for method in ['pearson','spearman']:
            eligible = joined.loc[:, joined.nunique(dropna=True)>1]
            matrix=eligible.corr(method=method)
            fig,ax=plt.subplots(figsize=(17,15)); im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r')
            ax.set_xticks(range(len(matrix)),matrix.columns,rotation=70,ha='right'); ax.set_yticks(range(len(matrix)),matrix.columns)
            for i in range(len(matrix)):
                for j in range(len(matrix)): ax.text(j,i,f'{matrix.iloc[i,j]:.2f}',ha='center',va='center',fontsize=5)
            ax.set_title(f'{method.title()} — {scope}\nObserved months {joined.index.min()}–{joined.index.max()}; incomplete nationality coverage'); fig.colorbar(im,ax=ax,label='Correlation'); fig.tight_layout()
            fig.savefig(figures/f'{"international" if "international" in name else "domestic"}_{method}.png',dpi=180); plt.close(fig)
    pd.concat([pd.DataFrame(correlations),pd.DataFrame(monthly_corr)],ignore_index=True).to_csv(results/'feature_correlations.csv',index=False)
    plot_quality(frames, pd.DataFrame(columns), figures)
    pd.DataFrame(columns=['feature','data_source','definition','unit','transformation','aggregation','temporal_alignment','missing_treatment','model_usage','leakage']).to_csv(results/'feature_inventory.csv',index=False)
    pd.DataFrame(columns=['model','target','scope','MAE','RMSE','WMAPE','R2','train_samples','evaluation_samples','evaluation_period','artifact']).to_csv(results/'model_benchmarking.csv',index=False)
    print(json.dumps(inventory,indent=2)); print('Correlation rows:',len(correlations)+len(monthly_corr))


def plot_quality(frames, quality, figures):
    labels = quality.file + ': ' + quality.column
    missing = (quality.missing + quality.suppressed).to_numpy()
    selected = missing > 0
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(labels[selected], missing[selected], color='#369653')
    ax.set_xlabel('Missing or suppressed observations (count)')
    ax.set_title('Input quality — all supplied workbooks')
    fig.tight_layout(); fig.savefig(figures/'missing_data.png', dpi=180); plt.close(fig)
    for name, frame in frames.items():
        if 'Guests' not in frame: continue
        scope = 'international nationality-date records' if 'Nationality' in frame else 'domestic daily records'
        prefix = 'international' if 'Nationality' in frame else 'domestic'
        target = numeric_clean(frame.Guests)
        period = f'{pd.to_datetime(frame.Date).min().date()}–{pd.to_datetime(frame.Date).max().date()}'
        fig, ax = plt.subplots(figsize=(8,5)); ax.hist(target.dropna(), bins=40, color='#369653')
        ax.set_xlabel('Guests per nationality-date' if prefix=='international' else 'Guests per date'); ax.set_ylabel('Observed records')
        ax.set_title(scope+' — '+period+' distribution')
        fig.tight_layout(); fig.savefig(figures/f'{prefix}_distribution.png',dpi=180); plt.close(fig)
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        for ax, feature in zip(axes,['New Arrivals','Same-Day Guests']):
            x=numeric_clean(frame[feature]); valid=x.notna() & target.notna()
            ax.scatter(x[valid],target[valid],s=3,alpha=.15,color='#369653',rasterized=True)
            ax.set_xlabel(feature+' (guests)'); ax.set_ylabel('Guests'); ax.set_title(f'{feature}; n={valid.sum():,}')
        fig.suptitle(scope+' — '+period+'\nContemporaneous observed records; no forecast inference')
        fig.tight_layout(); fig.savefig(figures/f'{prefix}_scatter.png',dpi=180); plt.close(fig)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--inputs',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(args.inputs,args.output)
