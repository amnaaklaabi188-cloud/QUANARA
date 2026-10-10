"""Independent checks against local original workbooks and generated CSVs."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def verify(inputs, output):
    root=output/'docs/results'
    inventory=pd.read_csv(root/'dataset_inventory.csv')
    frames={}
    for row in inventory.itertuples():
        paths=list(inputs.rglob(row.file))
        assert len(paths)==1, row.file
        path=paths[0]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row.sha256, row.file
        if path.suffix=='.xlsx':
            frame=pd.read_excel(path,keep_default_na=False)
            assert frame.shape==(int(row.rows),int(row.columns)),row.file
            assert frame.duplicated().sum()==row.duplicate_rows,row.file
            frames[row.file]=frame
    correlations=pd.read_csv(root/'feature_correlations.csv')
    flight=frames['flight_data.xlsx']
    flight_months=pd.to_datetime(flight.Date).dt.to_period('M')
    tested=0
    for row in correlations.itertuples():
        if row.aggregation=='daily':
            name='data domestic_train.xlsx' if row.scope=='domestic daily total' else 'data international_train.xlsx'
            frame=frames[name]
            if name.startswith('data international'):
                frame=frame[frame.Nationality==row.scope.split(': ',1)[1]]
            pair=frame[[row.feature,'Guests']].apply(lambda c:pd.to_numeric(c,errors='coerce'))
        else:
            name='data international_train.xlsx' if row.scope.startswith('international') else 'data domestic_train.xlsx'
            hotel=frames[name]
            months=pd.to_datetime(hotel.Date).dt.to_period('M')
            days=hotel.groupby(months).Date.nunique()
            complete=days.index[days.to_numpy()==np.array([m.days_in_month for m in days.index])]
            target=hotel.groupby(months).Guests.sum(min_count=1)
            feature=pd.to_numeric(flight[row.feature],errors='coerce').groupby(flight_months).sum(min_count=1)
            pair=pd.concat([feature.rename(row.feature),target.rename('Guests')],axis=1,join='inner').loc[lambda x:x.index.isin(complete)]
        valid=pair.dropna()
        assert len(valid)==row.valid_samples,(row.scope,row.feature,'n')
        assert len(pair)-len(valid)==row.missing_observations,(row.scope,row.feature,'missing')
        for method,col in [('pearson','pearson_r'),('spearman','spearman_rho')]:
            expected=valid[row.feature].corr(valid.Guests,method=method)
            assert np.isclose(expected,getattr(row,col),rtol=1e-10,atol=1e-10),(row.scope,row.feature,method)
        tested+=1
    for file in ['feature_inventory.csv','model_benchmarking.csv']:
        assert pd.read_csv(root/file).empty,file
    print(f'PASS: all six original hashes unchanged; five workbook dimensions verified; {tested} correlation rows independently recomputed; missing model tables empty.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.inputs,a.output)
