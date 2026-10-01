import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
df=pd.read_csv('../CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv',encoding='utf-8-sig',parse_dates=['date'])
df=df.sort_values(['location','date'])
Q='river_discharge_m3s';P='precipitation_mm';S='soil_moisture_0_100cm_m3m3'
locs=df.groupby('location').elevation_m.first().sort_values().index.tolist()
# monsoon share
df['m']=df.date.dt.month
ms=df.groupby('location').apply(lambda x: pd.Series({'monsoon_P_share':x[x.m.between(6,9)][P].sum()/x[P].sum(),'monsoon_Q_share':x[x.m.between(6,9)][Q].sum()/x[Q].sum(),'Q_cv':x[Q].std()/x[Q].mean(),'Q_p99_over_med':x[Q].quantile(.99)/max(x[Q].median(),1e-3),'Q_p95':x[Q].quantile(.95),'P_p99':x[P].quantile(.99),'rain_ne_precip':(x[P]-x['rain_mm']).abs().gt(.05).mean(),'Q_zero':(x[Q]==0).mean()}))
print(ms.round(3).to_string())
# lag xcorr
res={}
for l in locs:
    x=df[df.location==l].set_index('date')
    lp=np.log1p(x[P]);lq=np.log1p(x[Q]); dq=lq.diff()
    res[l]=[lp.corr(lq.shift(-k)) for k in range(0,8)]
lagdf=pd.DataFrame(res,index=range(8)).T; print(lagdf.round(2).to_string())
# 3d precip sum vs Q
# events
for l,d0,d1 in [('Khokana','2024-09-24','2024-10-03'),('Devghat','2024-09-24','2024-10-03'),('Rasuwagadhi','2025-07-04','2025-07-12'),('Bahrabise','2025-07-04','2025-07-12'),('Rasuwagadhi','2024-08-12','2024-08-20')]:
    x=df[(df.location==l)&(df.date.between(d0,d1))][['date',P,S,'temperature_mean_c',Q]]
    print(l);print(x.to_string(index=False))
# top discharge date per location
print(df.loc[df.groupby('location')[Q].idxmax(),['location','date',Q,P]].to_string())
# annual maxima per year
print(df.groupby(['location',df.date.dt.year])[Q].max().unstack().round(1).to_string())
# cross-site corr of log Q
piv=df.pivot(index='date',columns='location',values=Q); print(np.log1p(piv).corr().round(2).to_string())
pp=df.pivot(index='date',columns='location',values=P); print(np.log1p(pp).corr().round(2).to_string())
# stationarity-ish / autocorr
for l in locs:
    x=np.log1p(piv[l]); print(l, round(x.autocorr(1),3), round(x.autocorr(7),3))
# persistence NSE baseline h=1,3,7 on test last year (2025-09..2026-08)
def nse(o,p): return 1-((o-p)**2).sum()/((o-o.mean())**2).sum()
out=[]
for l in locs:
    q=piv[l]; te=q['2025-09-01':]
    out.append([l]+[nse(te.iloc[h:],q.shift(h)['2025-09-01':].iloc[h:]) for h in (1,3,7)])
print(pd.DataFrame(out,columns=['loc','h1','h3','h7']).round(3).to_string())
ms.to_csv('prelim_site_stats.csv'); lagdf.to_csv('prelim_lag_corr.csv')
# ---------- FIGURES
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
# Fig: study area scatter (lon/lat) w/ elevation
s=df.groupby('location').first()
fig,ax=plt.subplots(figsize=(7,3.6)); sc=ax.scatter(s.longitude,s.latitude,c=s.elevation_m,s=140,cmap='viridis',edgecolor='k')
for n,r in s.iterrows(): ax.annotate(n,(r.longitude,r.latitude),xytext=(4,5),textcoords='offset points',fontsize=7)
plt.colorbar(sc,label='Elevation (m)'); ax.set_xlabel('Longitude (°E)');ax.set_ylabel('Latitude (°N)');ax.set_title('Ten monitoring locations'); fig.tight_layout(); fig.savefig('figures/fig1_locations.png',dpi=160); plt.close()
# Fig: discharge time series log
fig,axs=plt.subplots(5,2,figsize=(10,9),sharex=True)
for ax,l in zip(axs.T.ravel(),locs):
    x=piv[l]; ax.semilogy(x.index,x.clip(lower=.01),lw=.6); ax.set_title(f'{l} ({s.loc[l,"river"]})',fontsize=8)
fig.suptitle('Modelled daily discharge (m³/s, log scale)'); fig.tight_layout(); fig.savefig('figures/fig2_discharge.png',dpi=160); plt.close()
# Fig: lag correlation heat
fig,ax=plt.subplots(figsize=(6,4)); im=ax.imshow(lagdf.values,cmap='magma',aspect='auto',vmin=0,vmax=lagdf.values.max())
ax.set_yticks(range(len(locs)));ax.set_yticklabels(lagdf.index,fontsize=7);ax.set_xlabel('Lag k (days): corr[log(1+P_t), log(1+Q_{t+k})]');plt.colorbar(im);fig.tight_layout();fig.savefig('figures/fig3_lagcorr.png',dpi=160);plt.close()
# Fig: seasonal climatology Q and P
fig,axs=plt.subplots(1,2,figsize=(9,3.2))
for l in locs:
    x=df[df.location==l]; c=x.groupby('m')[Q].mean(); axs[0].plot(c.index,c/c.mean(),label=l,lw=1)
    axs[1].plot(x.groupby('m')[P].sum()/ (x[P].sum()),lw=1)
axs[0].set_title('Mean Q by month / annual mean');axs[1].set_title('Monthly share of precipitation');axs[0].legend(fontsize=5);axs[0].set_xlabel('Month');axs[1].set_xlabel('Month')
fig.tight_layout();fig.savefig('figures/fig4_seasonality.png',dpi=160);plt.close()
# Fig: Sep 2024 event
fig,axs=plt.subplots(2,1,figsize=(7,4.5),sharex=True)
for l in ['Khokana','Devghat','Chatara']:
    x=df[(df.location==l)&df.date.between('2024-09-15','2024-10-15')].set_index('date')
    axs[0].plot(x[Q]/x[Q].iloc[0:5].mean(),label=l); axs[1].bar(x.index+pd.Timedelta(hours=0),x[P],alpha=.4,label=l)
axs[0].set_ylabel('Q / Q(first 5 d)');axs[1].set_ylabel('P (mm/day)');axs[0].legend();axs[0].set_title('27–29 Sep 2024 event');import matplotlib.dates as md;axs[1].xaxis.set_major_formatter(md.DateFormatter('%d %b'));axs[1].xaxis.set_major_locator(md.DayLocator(interval=5));fig.tight_layout();fig.savefig('figures/fig5_event2024.png',dpi=160);plt.close()
# Fig: scale mismatch: Q mean vs known? just boxplot log
fig,ax=plt.subplots(figsize=(7,3.2)); ax.boxplot([np.log10(piv[l].clip(lower=1e-2)) for l in locs],tick_labels=locs,showfliers=False); plt.xticks(rotation=45,ha='right',fontsize=7); ax.set_ylabel('log10 Q (m³/s)'); fig.tight_layout(); fig.savefig('figures/fig6_qbox.png',dpi=160); plt.close()
