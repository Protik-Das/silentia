# ============================================================
# SILENTIA | Figures 2, 5 and 6
# Colab-ready | 16 cm | 300 dpi | TIFF + JPG + EPS
# Also exports the plotted data and bootstrap results to Excel.
# ============================================================

# 1) Upload these two CSV files when prompted:
#    silence_global.csv
#    country_level_silence_governance.csv

from google.colab import files
import io, os, glob, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

uploaded = files.upload()

def find_uploaded(key):
    for name, content in uploaded.items():
        if key.lower() in name.lower():
            return pd.read_csv(io.BytesIO(content))
    raise FileNotFoundError(f'Please upload a CSV whose filename contains: {key}')

panel = find_uploaded('silence_global')
gov = find_uploaded('country_level_silence_governance')

OUT = '/content/SILENTIA_Figures_2_5_6'
os.makedirs(OUT, exist_ok=True)
CM = 1/2.54
DPI = 300
INK, MID, LIGHT = '#222222', '#666666', '#B8B8B8'
GROUP_ORDER = ['LIC', 'LMC', 'UMC']
GROUP_LABELS = {'LIC':'Low income', 'LMC':'Lower-middle income', 'UMC':'Upper-middle income'}
GROUP_MARKERS = {'LIC':'o', 'LMC':'s', 'UMC':'^'}
GROUP_COLORS = {'LIC':'#333333', 'LMC':'#777777', 'UMC':'#BBBBBB'}

plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':7.5,
    'axes.labelsize':7.5, 'axes.titlesize':8,
    'xtick.labelsize':7, 'ytick.labelsize':7,
    'axes.edgecolor':INK, 'axes.linewidth':0.65,
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.facecolor':'white', 'figure.facecolor':'white',
    'savefig.facecolor':'white', 'pdf.fonttype':42, 'ps.fonttype':42
})

def export(fig, stem):
    fig.savefig(f'{OUT}/{stem}.tif', dpi=DPI, format='tiff',
                pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(f'{OUT}/{stem}.jpg', dpi=DPI, format='jpeg',
                pil_kwargs={'quality':95})
    fig.savefig(f'{OUT}/{stem}.eps', format='eps')
    plt.show()
    plt.close(fig)

# ------------------------------------------------------------
# Validate and prepare the two supplied datasets
# ------------------------------------------------------------
panel['year'] = pd.to_numeric(panel['year'], errors='raise').astype(int)
panel['M_missing'] = pd.to_numeric(panel['M_missing'], errors='coerce')
panel = panel.sort_values(['iso','year']).copy()
if panel.duplicated(['iso','year']).any():
    raise ValueError('Duplicate country-year rows found in silence_global.csv')

# FIGURE 2 | Anatomy of silence: three panels
# (a) year x income-group mean M
# (b) country-level mean M by income group
# (c) top 15 countries by mean M
annual = (panel.groupby(['year','income'], as_index=False)['M_missing']
          .mean().rename(columns={'M_missing':'mean_M'}))
country_mean = (panel.groupby(['iso','country','income'], as_index=False)['M_missing']
                .mean().rename(columns={'M_missing':'mean_M'}))
country_mean = country_mean.sort_values('mean_M', ascending=False)
top15 = country_mean.head(15).sort_values('mean_M', ascending=True)

fig, axes = plt.subplots(1,3,figsize=(16*CM,6.5*CM),
                         gridspec_kw={'width_ratios':[1.45,1,1.25]})
ax=axes[0]
for g in GROUP_ORDER:
    z=annual[annual.income==g]
    ax.plot(z.year,z.mean_M,marker=GROUP_MARKERS[g],ms=2.8,lw=1.15,
            color=GROUP_COLORS[g],label=GROUP_LABELS[g])
ax.set_xlabel('Year'); ax.set_ylabel('Mean missing share (M)')
ax.legend(frameon=False,fontsize=6.3,loc='best')
ax.set_title('(a) Mean M by year and income group',loc='left',fontsize=7.2,pad=6)

ax=axes[1]
data=[country_mean.loc[country_mean.income==g,'mean_M'].dropna().values for g in GROUP_ORDER]
parts=ax.boxplot(data,patch_artist=True,widths=.55,showfliers=False,
                 medianprops={'color':'black','linewidth':1},
                 boxprops={'edgecolor':MID,'linewidth':.8},
                 whiskerprops={'color':MID,'linewidth':.8},
                 capprops={'color':MID,'linewidth':.8})
for patch in parts['boxes']:
    patch.set_facecolor('white')
for i,g in enumerate(GROUP_ORDER,1):
    vals=data[i-1]
    # deterministic light jitter for visibility
    rng=np.random.default_rng(100+i)
    jitter=rng.normal(0,.045,len(vals))
    ax.scatter(np.full(len(vals),i)+jitter,vals,s=8,
               color=GROUP_COLORS[g],alpha=.65,linewidths=0)
ax.set_xticks([1,2,3],['LIC','LMC','UMC'])
ax.set_ylabel('Country mean M')
ax.set_title('(b) Country-level mean M',loc='left',fontsize=7.2,pad=6)

ax=axes[2]
ax.barh(top15.country,top15.mean_M,color='#777777',edgecolor='black',linewidth=.35)
ax.set_xlabel('Country mean M')
ax.set_title('(c) 15 most silent countries',loc='left',fontsize=7.2,pad=6)
ax.tick_params(axis='y',labelsize=5.9)
ax.set_xlim(0,max(.5,float(top15.mean_M.max())*1.12))
fig.subplots_adjust(left=.07,right=.99,bottom=.16,top=.86,wspace=.48)
export(fig,'Figure_2_Anatomy_of_Silence')

# FIGURE 5 | Silence vs governance
# Country-level mean M matched by ISO; Spearman rho and country bootstrap CI.
gov['M']=pd.to_numeric(gov['M'],errors='coerce')
gov['GovEff']=pd.to_numeric(gov['GovEff'],errors='coerce')
gov['SPI']=pd.to_numeric(gov['SPI'],errors='coerce')
merged=gov.dropna(subset=['M','GovEff']).copy()

def spearman_boot_ci(data, x, y, B=500, seed=20261118):
    d=data[[x,y]].dropna().reset_index(drop=True)
    rho=float(spearmanr(d[x],d[y]).statistic)
    rng=np.random.default_rng(seed)
    boots=[]
    n=len(d)
    for _ in range(B):
        s=d.iloc[rng.integers(0,n,n)]
        if s[x].nunique()>1 and s[y].nunique()>1:
            boots.append(spearmanr(s[x],s[y]).statistic)
    lo,hi=np.quantile(boots,[.025,.975])
    return rho,float(lo),float(hi),len(d),np.asarray(boots)

res_ge=spearman_boot_ci(merged,'M','GovEff',seed=101)
spi_data=gov.dropna(subset=['M','SPI']).copy()
res_spi=spearman_boot_ci(spi_data,'M','SPI',seed=102)

fig,axes=plt.subplots(1,2,figsize=(16*CM,6.3*CM),sharey=True)
for ax,dat,x,xlabel,res,panel_label in [
    (axes[0],merged,'GovEff','Government effectiveness',res_ge,'(a)'),
    (axes[1],spi_data,'SPI','Statistical Performance Indicator',res_spi,'(b)')]:
    for g in GROUP_ORDER:
        z=dat[dat.income==g]
        ax.scatter(z[x],z['M'],s=21,marker=GROUP_MARKERS[g],
                   facecolor=GROUP_COLORS[g],edgecolor='black',linewidth=.35,
                   alpha=.82,label=GROUP_LABELS[g])
    # Descriptive linear fit, as stated in the manuscript caption
    z=dat[[x,'M']].dropna()
    if len(z)>1:
        b1,b0=np.polyfit(z[x],z['M'],1)
        xx=np.linspace(z[x].min(),z[x].max(),100)
        ax.plot(xx,b1*xx+b0,color=MID,lw=1,ls='--')
    rho,lo,hi,n,_=res
    ax.text(.04,.96,f"Spearman rho = {rho:.2f}\n95% CI [{lo:.2f}, {hi:.2f}]\nn = {n}",
            transform=ax.transAxes,ha='left',va='top',fontsize=6.8,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
    ax.set_xlabel(xlabel)
    ax.set_title(f'{panel_label} {xlabel}',loc='left',fontsize=7.2,pad=6)
axes[0].set_ylabel('Country mean missing share (M)')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles[:3],labels[:3],frameon=False,ncol=3,loc='lower center',
           bbox_to_anchor=(.5,-.005),fontsize=6.5)
fig.subplots_adjust(left=.09,right=.98,bottom=.22,top=.86,wspace=.24)
export(fig,'Figure_5_Silence_vs_Governance')

# FIGURE 6 | Reconstruct event study using the manuscript's stated rules.
# Collapse: within-country annual increase in M >= 0.20.
# Controls: five non-collapse countries in same income group, nearest on
# four-year pre-collapse mean M. All collapse-episode countries are excluded
# from the control pool, to avoid treated episodes serving as controls.
# DiD: (treated M at relative year - treated pre-period mean) minus
#      (control M at relative year - control pre-period mean).
# Bootstrap resamples the ten episodes (2,000 draws).
wide=panel.pivot(index='iso',columns='year',values='M_missing')
meta=panel[['iso','country','income']].drop_duplicates('iso').set_index('iso')
wide=wide.join(meta)
wide=wide.sort_index(axis=1)
changes=panel.sort_values(['iso','year']).copy()
changes['delta_M']=changes.groupby('iso')['M_missing'].diff()
episodes=changes.loc[changes.delta_M>=.20,['iso','country','income','year']].copy()
episodes=episodes.sort_values(['year','iso']).reset_index(drop=True)
expected={'ERI':2012,'SSD':2016,'VEN':2018,'YEM':2019,'AFG':2021,'SDN':2023,'SYR':2023,'COD':2024,'NIC':2024,'LBN':2024}
found=dict(zip(episodes.iso,episodes.year))
if found != expected:
    print('WARNING: Detected collapse episodes differ from manuscript list:')
    print(found)
else:
    print('Verified: all 10 collapse episodes match the manuscript list.')

collapse_isos=set(episodes.iso)
rel_years=list(range(-4,4))
records=[]
for _,e in episodes.iterrows():
    iso,year,inc=e.iso,int(e.year),e.income
    if not all(y in wide.columns for y in range(year-4,year+1)):
        raise ValueError(f'Missing pre-period years for {iso} {year}')
    pre=wide.loc[iso,[year-4,year-3,year-2,year-1]].astype(float).mean()
    candidates=wide[(wide.income==inc) & (~wide.index.isin(collapse_isos))].copy()
    candidates['pre_mean']=candidates[[year-4,year-3,year-2,year-1]].mean(axis=1)
    candidates=candidates.dropna(subset=['pre_mean'])
    candidates['distance']=(candidates.pre_mean-pre).abs()
    controls=candidates.sort_values(['distance'],kind='mergesort').head(5).index.tolist()
    if len(controls)!=5:
        raise ValueError(f'Could not find five matched controls for {iso} {year}')
    ctrl_pre=wide.loc[controls,[year-4,year-3,year-2,year-1]].mean(axis=1).mean()
    for rel in rel_years:
        yr=year+rel
        treated=float(wide.loc[iso,yr]) if yr in wide.columns else np.nan
        ctrl=float(wide.loc[controls,yr].mean()) if yr in wide.columns else np.nan
        records.append({'episode_iso':iso,'episode_country':e.country,'collapse_year':year,
                        'income':inc,'relative_year':rel,'calendar_year':yr,
                        'treated_M':treated,'control_M':ctrl,
                        'treated_pre_mean':float(pre),'control_pre_mean':float(ctrl_pre),
                        'n_controls':len(controls),'control_isos':';'.join(controls)})
event_long=pd.DataFrame(records)
event_long['DiD']=(event_long.treated_M-event_long.treated_pre_mean)-(
                    event_long.control_M-event_long.control_pre_mean)

# Aggregate means and episode-bootstrap percentile intervals.
B=2000
rng=np.random.default_rng(20261118)
summary=[]
for rel in rel_years:
    z=event_long[event_long.relative_year==rel].dropna(subset=['treated_M','control_M','DiD'])
    n=len(z)
    if n==0: continue
    treated_mean=z.treated_M.mean(); control_mean=z.control_M.mean(); did_mean=z.DiD.mean()
    boots_t=[]; boots_c=[]; boots_d=[]
    vals=z[['treated_M','control_M','DiD']].to_numpy()
    for _ in range(B):
        sample=vals[rng.integers(0,n,n)]
        boots_t.append(sample[:,0].mean()); boots_c.append(sample[:,1].mean()); boots_d.append(sample[:,2].mean())
    tlo,thi=np.quantile(boots_t,[.025,.975]); clo,chi=np.quantile(boots_c,[.025,.975]); dlo,dhi=np.quantile(boots_d,[.025,.975])
    summary.append({'relative_year':rel,'n_episodes':n,
                    'treated_mean_M':treated_mean,'treated_CI_low':tlo,'treated_CI_high':thi,
                    'control_mean_M':control_mean,'control_CI_low':clo,'control_CI_high':chi,
                    'DiD':did_mean,'DiD_CI_low':dlo,'DiD_CI_high':dhi})
event_summary=pd.DataFrame(summary)

fig,axes=plt.subplots(1,2,figsize=(16*CM,6.3*CM),gridspec_kw={'width_ratios':[1.35,1]})
ax=axes[0]
for group,label,color,marker,low,high in [
    ('treated','Collapse countries','#222222','o','treated_CI_low','treated_CI_high'),
    ('control','Matched controls','#888888','s','control_CI_low','control_CI_high')]:
    ycol='treated_mean_M' if group=='treated' else 'control_mean_M'
    ax.plot(event_summary.relative_year,event_summary[ycol],
            color=color,marker=marker,ms=3.2,lw=1.25,label=label)
    ax.fill_between(event_summary.relative_year,event_summary[low],event_summary[high],
                    color=color,alpha=.13,linewidth=0)
ax.axvline(-.5,color=MID,lw=.8,ls='--')
ax.set_xlabel('Year relative to reporting collapse')
ax.set_ylabel('Mean missing share (M)')
ax.set_xticks(rel_years)
ax.legend(frameon=False,fontsize=6.5,loc='best')
ax.set_title('(a) Collapse countries and matched controls',loc='left',fontsize=7.2,pad=6)

ax=axes[1]
z=event_summary.dropna(subset=['DiD'])
ax.axhline(0,color=MID,lw=.8,ls='--')
ax.errorbar(z.relative_year,z.DiD,
            yerr=[z.DiD-z.DiD_CI_low,z.DiD_CI_high-z.DiD],
            fmt='o',color='#222222',ecolor='#555555',elinewidth=1,
            capsize=2.5,markersize=3.5)
ax.set_xlabel('Year relative to reporting collapse')
ax.set_ylabel('Difference-in-differences in M')
ax.set_xticks(rel_years)
ax.set_title('(b) Episode-bootstrap DiD estimates',loc='left',fontsize=7.2,pad=6)
fig.subplots_adjust(left=.09,right=.98,bottom=.16,top=.86,wspace=.38)
export(fig,'Figure_6_Event_Study')

# Export all plotted data and reproducibility details to Excel
xlsx=f'{OUT}/SILENTIA_Figures_2_5_6_Data.xlsx'
with pd.ExcelWriter(xlsx,engine='openpyxl') as writer:
    annual.to_excel(writer,sheet_name='Fig2_Year_Income_M',index=False)
    country_mean.to_excel(writer,sheet_name='Fig2_Country_Mean_M',index=False)
    top15.to_excel(writer,sheet_name='Fig2_Top15_Silent',index=False)
    merged[['iso','country','income','M','GovEff']].to_excel(writer,sheet_name='Fig5_GovEff_Data',index=False)
    spi_data[['iso','country','income','M','SPI']].to_excel(writer,sheet_name='Fig5_SPI_Data',index=False)
    pd.DataFrame([
        {'comparison':'M vs Government Effectiveness','rho':res_ge[0],'CI_low':res_ge[1],'CI_high':res_ge[2],'n':res_ge[3],'bootstrap_draws':500},
        {'comparison':'M vs SPI','rho':res_spi[0],'CI_low':res_spi[1],'CI_high':res_spi[2],'n':res_spi[3],'bootstrap_draws':500}
    ]).to_excel(writer,sheet_name='Fig5_Spearman_CI',index=False)
    episodes.to_excel(writer,sheet_name='Fig6_Collapse_Episodes',index=False)
    event_long.to_excel(writer,sheet_name='Fig6_Episode_RelativeYear',index=False)
    event_summary.to_excel(writer,sheet_name='Fig6_Summary_Bootstrap',index=False)
    pd.DataFrame([
        ['Collapse rule','Annual increase in M >= 0.20'],
        ['Matched controls','Five non-collapse countries; same income group; nearest four-year pre-collapse mean M'],
        ['DiD definition','(treated M at relative year - treated pre-period mean) - (control M at relative year - control pre-period mean)'],
        ['Event bootstrap','2,000 resamples of episodes; percentile 95% intervals'],
        ['Important','This event study is reconstructed from the supplied panel using the manuscript methods. It is not a recovered original matched-control file.']
    ],columns=['Item','Method']).to_excel(writer,sheet_name='Readme_Method',index=False)

# Bundle all outputs
zip_path='/content/SILENTIA_Figures_2_5_6_Bundle.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as zf:
    for name in os.listdir(OUT):
        zf.write(os.path.join(OUT,name),arcname=name)

print('\nAll three figures and Excel data workbook are ready:')
print(OUT)
files.download(zip_path)
