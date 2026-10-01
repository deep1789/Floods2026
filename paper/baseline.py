import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as H
df=pd.read_csv('../CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv',encoding='utf-8-sig',parse_dates=['date']).sort_values(['location','date']).reset_index(drop=True)
Q='river_discharge_m3s';P='precipitation_mm';S='soil_moisture_0_100cm_m3m3'
g=df.groupby('location')
df['lq']=np.log1p(df[Q]); df['lp']=np.log1p(df[P])
for k in (1,2,3,7): df[f'lq_l{k}']=g.lq.shift(k)
for k in (1,2,3): df[f'lp_l{k}']=g.lp.shift(k)
for w in (3,7,14,30): df[f'P_s{w}']=g[P].transform(lambda x:x.rolling(w).sum())
df['dlq']=df.lq-df.lq_l1
df['S_d7']=df[S]-g[S].shift(7)
df['sin']=np.sin(2*np.pi*df.date.dt.dayofyear/365.25);df['cos']=np.cos(2*np.pi*df.date.dt.dayofyear/365.25)
df['loc']=df.location.astype('category').cat.codes
feats=['lq','lq_l1','lq_l2','lq_l3','lq_l7','lp','lp_l1','lp_l2','lp_l3','P_s3','P_s7','P_s14','P_s30',S,'S_d7','temperature_mean_c','relative_humidity_mean_pct','sin','cos','dlq','elevation_m','loc']
def nse(o,p): return 1-((o-p)**2).sum()/((o-o.mean())**2).sum()
rows=[]
for h in (1,3,7):
    df['y']=g.lq.shift(-h); df['yq']=g[Q].shift(-h)
    d=df.dropna(subset=feats+['y']); tr=d[d.date<'2025-09-01'];te=d[d.date>='2025-09-01']
    # residual target: y - lq (predict change in log Q)
    m=H(max_iter=400,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,random_state=0,categorical_features=[feats.index('loc')])
    m.fit(tr[feats],tr.y-tr.lq); pred=np.expm1(te.lq+m.predict(te[feats])).clip(lower=0)
    per=np.expm1(te.lq)
    for l,x in te.assign(pred=pred,per=per).groupby('location'):
        rows.append([h,l,nse(x.yq,x.per),nse(x.yq,x.pred),nse(np.log1p(x.yq),np.log1p(x.per)),nse(np.log1p(x.yq),np.log1p(x.pred))])
r=pd.DataFrame(rows,columns=['h','location','NSE_persist','NSE_hgb','logNSE_persist','logNSE_hgb'])
r.to_csv('baseline_results.csv',index=False)
print(r.round(3).to_string()); print(r.groupby('h').median(numeric_only=True).round(3))
# feature importance via permutation for h=3? skip
