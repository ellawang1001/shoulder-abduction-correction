"""Render Figures 2–4 at their manuscript width with legible text.

Reads existing observations and statistics; does not recompute or modify results.
Supports both the revision-package and flat public-deposit directory layouts.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sim_proof import build_scene, METHODS

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = (ROOT/'data/real_estimates_8subj.csv').exists()
FIG = ROOT.parent/'01_submission/figures' if PACKAGED else ROOT/'figures'
VECTOR = ROOT/'source/figures' if PACKAGED else FIG
df = pd.read_csv(ROOT/('data/real_estimates_8subj.csv' if PACKAGED else 'dataset_per_clip.csv'))
syn = pd.read_csv(ROOT/('data/synthetic_dataset.csv' if PACKAGED else 'synthetic_dataset.csv'))
summary = json.loads((ROOT/'results/statistics.json').read_text(encoding='utf-8'))
for m in METHODS:
    df['err_'+m] = abs(df[m]-df.true_angle)
    syn['err_'+m] = abs(syn[m]-syn.true_angle)
main = df[df.azimuth <= 60]
people = sorted(df.person.unique())
labels = ['Nose baseline', 'Raw 2D', 'Bone-length correction', '3D reference']
colors = ['#777777', '#D55E00', '#0072B2', '#009E73']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12.5,
    'axes.labelsize':12.5, 'axes.titlesize':13, 'xtick.labelsize':12.5,
    'ytick.labelsize':12.5, 'legend.fontsize':12.5,
    'axes.spines.top':False, 'axes.spines.right':False})


def save(fig, n):
    FIG.mkdir(parents=True, exist_ok=True)
    VECTOR.mkdir(parents=True, exist_ok=True)
    # Exact 6.4-inch canvas: no tight-bbox rescaling of the nominal font size.
    for ext in (['png','svg','tiff'] if PACKAGED else ['png','svg']):
        target = FIG if ext == 'png' else VECTOR
        fig.savefig(target/f'fig{n}.{ext}', dpi=300)
    plt.close(fig)


fig = plt.figure(figsize=(6.4, 6.6))
ax = fig.add_axes([.14,.60,.73,.32], projection='3d')
scene = build_scene(60)
for key,c in [('hip','#444444'),('elbow',colors[2])]:
    p=scene[key]; ax.plot([0,p[0]],[0,p[2]],[0,p[1]],color=c,lw=3)
    if key == 'elbow':
        ax.text(p[0]+.04,p[2]-.04,p[1]-.08,key,fontsize=12.5)
    else:
        ax.text(p[0],p[2],p[1],key,fontsize=12.5)
phi = np.linspace(0,np.pi/2,60)
# Preserve the existing geometry and camera path; this update changes typography only.
ax.plot(.55*np.sin(phi),.55*np.cos(phi),np.zeros(60),color='#777777',ls='--')
for deg in [0,30,45,60,90]:
    p=np.radians(deg);x,z=.55*np.sin(p),.55*np.cos(p)
    ax.scatter(x,z,0,color='#333333',s=16)
    ax.text(x,z,.12 if deg == 45 else .05,f'{deg}°',fontsize=12.5)
ax.text(0,0,.055,'shoulder',fontsize=12.5)
ax.set(xlabel='Lateral (m)',ylabel='Anterior (m)',xlim=(-.03,.6),ylim=(-.03,.6),
       xticks=[0,.3],yticks=[0,.3],zticks=[-.4,-.2,0])
ax.tick_params(pad=0)
ax.xaxis.labelpad=2
ax.yaxis.labelpad=2
ax.view_init(20,35)
fig.text(.5,.97,'A  Synthetic arm and camera azimuth',ha='center',fontsize=13)
fig.text(.06,.91,'Vertical (m)',fontsize=12.5)
fig.text(.5,.50,'Arm moves in the coronal plane; camera positions\nare schematic (simulated distance: 2 m).',ha='center',fontsize=12.5)
ax = fig.add_axes([.14,.09,.82,.28])
for m,label,c in zip(METHODS,labels,colors):
    s=syn.groupby('azimuth')['err_'+m].mean()
    ax.plot(s.index,s.values,'o-',label=label,color=c,ms=4)
ax.set(xlabel='Camera azimuth (°)',ylabel='Mean absolute error (°)',
       xticks=[0,15,30,45,60,75,90],ylim=(0,130))
fig.text(.14,.465,'B  Noisy perspective simulation',fontsize=13)
ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,
          borderaxespad=0,handlelength=1.2,columnspacing=1.0,labelspacing=.2)
save(fig,2)

fig = plt.figure(figsize=(6.4,6.6))
axes = [fig.add_axes([.14,.61,.82,.25]),fig.add_axes([.14,.095,.82,.28])]
g=df.groupby(['person','azimuth'])[['err_'+m for m in METHODS]].mean()
for index,(ax,methods,title) in enumerate([(axes[0],METHODS,'A  All four estimators'),
                                        (axes[1],METHODS[1:],'B  Trunk-referenced estimators')]):
    for m in methods:
        j=METHODS.index(m);c=colors[j]
        vals=g['err_'+m].unstack('azimuth').reindex(people);az=vals.columns.to_numpy()
        ax.errorbar(az,vals.mean(),yerr=vals.sem(),color=c,marker='o',ms=4,lw=1.5,capsize=3,label=labels[j])
        for i,p in enumerate(people):
            ax.scatter(az+(j-1.5)*.6+(i-3.5)*.3,vals.loc[p],s=13,alpha=.55,color=c,edgecolors='none')
    ax.set(xlabel='Camera azimuth (°)',ylabel='Mean absolute error (°)',
           xticks=[0,30,45,60,90],ylim=(0,130 if index==0 else 65))
    ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,
              borderaxespad=0,handlelength=1.2,columnspacing=1.0,labelspacing=.2)
    fig.text(.14,.975 if index==0 else .49,title,fontsize=13)
save(fig,3)

fig,ax=plt.subplots(figsize=(6.4,4.5))
fig.subplots_adjust(left=.15,right=.97,bottom=.16,top=.96)
x=(main.bonelength+main.true_angle)/2;y=main.bonelength-main.true_angle
ax.scatter(x,y,s=18,alpha=.6,color=colors[2]);ba=summary['bland_altman']
ax.axhline(ba['bias'],color='#333333',lw=1.5,label=f"Mean bias {ba['bias']:.1f}°")
for key in ['loa_low','loa_high']:
    ax.axhline(ba[key],color='#C44E52',ls='--',lw=1.3,label=f"95% limit {ba[key]:.1f}°")
ax.set(xlabel='Mean of corrected and reference angles (°)',ylabel='Corrected − reference angle (°)',ylim=(-45,55))
ax.legend(loc='upper right',frameon=False,labelspacing=.2)
save(fig,4)
print('Rendered Figures 2–4 at 6.4 inches with 12.5 pt minimum ordinary text.')
