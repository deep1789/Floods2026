import pandas as pd, numpy as np
from floodlab import data as _d
df=_d.load().sort_values(['location','date'])
Q='river_discharge_m3s';P='precipitation_mm';S='soil_moisture_0_100cm_m3m3'
def md(d,idx=False):
    d=d.reset_index() if idx else d
    h='| '+' | '.join(map(str,d.columns))+' |\n|'+'|'.join(['---']*len(d.columns))+'|\n'
    return h+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in d.values)+'\n'
w=lambda n,t:open(f'tables/{n}.md','w').write(t)
order=df.groupby('location').elevation_m.first().sort_values().index
# T1 sites
s=df.groupby('location').agg(River=('river','first'),Basin=('basin','first'),DHM=('dhm_station','first'),Lat=('latitude','first'),Lon=('longitude','first'),Elev=('elevation_m','first'),Qmean=(Q,'mean'),Qmed=(Q,'median'),Qmax=(Q,'max')).loc[order]
s=s.round({'Lat':3,'Lon':3,'Qmean':1,'Qmed':2,'Qmax':1}).reset_index()
s.columns=['Location','River','Basin','DHM id','Lat (°N)','Lon (°E)','Elev (m)','Mean Q','Median Q','Max Q']
w('sites',md(s))
# T3 site stats
st=pd.read_csv('prelim_site_stats.csv',index_col=0).loc[order]
t=pd.DataFrame({'Monsoon (JJAS) share of P':(st.monsoon_P_share*100).round(0).astype(int).astype(str)+'%','Monsoon share of Q':(st.monsoon_Q_share*100).round(0).astype(int).astype(str)+'%','CV of Q':st.Q_cv.round(2),'Q99 / median':st.Q_p99_over_med.round(1),'P99 (mm/d)':st.P_p99.round(1),'Zero-Q days':(st.Q_zero*100).round(1).astype(str)+'%'})
w('sitestats',md(t,True))
# T4 lag
lag=pd.read_csv('prelim_lag_corr.csv',index_col=0).loc[order]
t=pd.DataFrame({'r(k=0)':lag['0'].round(2),'r(k=1)':lag['1'].round(2),'r(k=3)':lag['3'].round(2),'r(k=7)':lag['7'].round(2),'argmax k':lag.values.argmax(1),'max r':lag.max(1).round(2)})
w('lag',md(t,True))
# T5 annual maxima
am=df.groupby(['location',df.date.dt.year])[Q].max().unstack().round(1).loc[order]; am.columns=[f'{c}'+(' (to 31 Aug)' if c==2026 else '') for c in am.columns]
w('annmax',md(am,True))
# T6 event
rows=[]
for l in order:
    x=df[df.location==l].set_index('date'); win=x['2024-09-20':'2024-10-08']
    pre=x[Q]['2024-09-15':'2024-09-24'].median(); pp=win[P]['2024-09-24':'2024-09-30'].idxmax(); qp=win[Q].idxmax()
    rows.append([l,round(x[P]['2024-09-26':'2024-09-28'].sum(),1),round(pre,2),round(win[Q].max(),2),round(win[Q].max()/max(pre,.01),1),round(win[Q].max()/x[Q].quantile(.99),2),(qp-pp).days if l!='Chameliya/Nayalbadi' else '—'])
w('event',md(pd.DataFrame(rows,columns=['Location','P 26–28 Sep (mm)','Pre-event Q (median 15–24 Sep)','Peak Q 20 Sep–8 Oct','Peak / pre-event','Peak / Q99','Lag P-peak→Q-peak (d)'])))
import os
if os.path.exists('baseline_results.csv'):
    # T7 baseline
    b=pd.read_csv('baseline_results.csv')
    for h in (1,3,7):
        d=b[b.h==h].set_index('location').loc[order]
        t=pd.DataFrame({'NSE persistence':d.NSE_persist,'NSE HGB':d.NSE_hgb,'logNSE persistence':d.logNSE_persist,'logNSE HGB':d.logNSE_hgb})
        t.loc['**Median**']=t.median()
        t=t.round(3)
        w(f'base{h}',md(t,True))
# T8 audit
a=df.groupby('location').apply(lambda z: pd.Series({'Zero P days':f'{(z[P]==0).mean()*100:.0f}%','P≠rain days':int(((z[P]-z.rain_mm)>0.05).sum()),'Repeated Q (ΔQ=0)':f'{(z[Q].diff()==0).mean()*100:.0f}%','Q<0.01':f'{(z[Q]<0.01).mean()*100:.1f}%','Mean temp °C':round(z.temperature_mean_c.mean(),1)})).loc[order]
w('audit',md(a,True))
