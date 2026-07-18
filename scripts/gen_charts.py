#!/usr/bin/env python3
"""Generate 12 NavDP-MINCO paper charts. Pure matplotlib, DejaVu Serif, 300 DPI."""

import os, sys, warnings, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import pi
from matplotlib.patches import FancyBboxPatch
warnings.filterwarnings("ignore")

OUT = "/home/alioth/NavDP/results/navdp_minco_full_real/analysis_charts"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#333333",
})
CR, CC, CH = "#4472C4", "#ED7D31", "#70AD47"

def sv(n):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, n), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"  -> {n}")

# Data
nav = {
    "SR":  {"R": (71.7,59.2,81.5), "C": (81.7,70.1,89.4), "H": (76.7,64.6,85.6)},
    "RMSE":{"R": (0.0814,0.0685,0.0956),"C":(0.0221,0.0172,0.0287),"H":(0.0483,0.0389,0.0578)},
    "DUR": {"R": (54.2,48.3,60.2), "C": (45.8,39.8,53.1), "H": (46.9,41.9,52.2)},
    "PL":  {"R": (17.8,15.7,19.8), "C": (16.7,14.3,19.1), "H": (17.3,14.7,19.8)},
    "SPL": {"R": (0.657,0.549,0.757),"C":(0.767,0.671,0.860),"H":(0.675,0.573,0.766)},
}

# ---- Chart 1 ----
def c1():
    fig,ax=plt.subplots(figsize=(5,4.5))
    m=[nav["SR"][k][0] for k in ["R","C","H"]]
    lo=[nav["SR"][k][1] for k in ["R","C","H"]]
    hi=[nav["SR"][k][2] for k in ["R","C","H"]]
    cs=[CR,CC,CH]; x=np.arange(3)
    b=ax.bar(x,m,color=cs,width=0.55,edgecolor="white",linewidth=0.5)
    for i in range(3):
        ax.errorbar(x[i],m[i],yerr=[[m[i]-lo[i]],[hi[i]-m[i]]],
                    fmt="none",color="black",capsize=4,capthick=1.2,lw=1.2)
        ax.text(x[i],m[i]+2.5,f"{m[i]:.1f}%",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["RAW","COLD","HOT"])
    ax.set_ylabel("Success Rate (%)"); ax.set_title("Navigation Success Rate",fontweight="bold")
    ax.set_ylim(0,100)
    sv("nav_success_rate.png")

# ---- Chart 2 ----
def c2():
    fig,ax=plt.subplots(figsize=(5,4.5))
    m=[nav["RMSE"][k][0] for k in ["R","C","H"]]
    lo=[nav["RMSE"][k][1] for k in ["R","C","H"]]
    hi=[nav["RMSE"][k][2] for k in ["R","C","H"]]
    cs=[CR,CC,CH]; x=np.arange(3)
    b=ax.bar(x,m,color=cs,width=0.55,edgecolor="white",linewidth=0.5)
    for i in range(3):
        ax.errorbar(x[i],m[i],yerr=[[m[i]-lo[i]],[hi[i]-m[i]]],
                    fmt="none",color="black",capsize=4,capthick=1.2,lw=1.2)
    txt=[f"{m[0]:.4f} m",f"{m[1]:.4f} m\n(-72.8%)",f"{m[2]:.4f} m\n(-40.7%)"]
    for bb,v,t in zip(b,m,txt):
        ax.text(bb.get_x()+bb.get_width()/2,bb.get_height()+0.002,t,
                ha="center",va="bottom",fontsize=8,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["RAW","COLD","HOT"])
    ax.set_ylabel("Tracking RMSE (m)"); ax.set_title("Cross-Track Error RMSE (m)",fontweight="bold")
    ax.set_ylim(0,max(m)*1.6)
    sv("tracking_rmse.png")

# ---- Chart 3 ----
def c3():
    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    met=[("SPL","SPL",nav["SPL"],0,1.0),("DUR","Duration (s)",nav["DUR"],0,None),
         ("PL","Path Length (m)",nav["PL"],0,None)]
    for ax,(_,yl,d,ylo,yhi) in zip(axes,met):
        m=[d[k][0] for k in ["R","C","H"]]
        lo=[d[k][1] for k in ["R","C","H"]]; hi=[d[k][2] for k in ["R","C","H"]]
        x=np.arange(3)
        b=ax.bar(x,m,color=[CR,CC,CH],width=0.5,edgecolor="white",linewidth=0.5)
        for i in range(3):
            ax.errorbar(x[i],m[i],yerr=[[m[i]-lo[i]],[hi[i]-m[i]]],
                        fmt="none",color="black",capsize=3,capthick=1,lw=1)
        for bb,v in zip(b,m):
            ax.text(bb.get_x()+bb.get_width()/2,bb.get_height()+0.01,
                    f"{v:.3f}" if yl=="SPL" else f"{v:.1f}",
                    ha="center",va="bottom",fontsize=8,fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(["RAW","COLD","HOT"])
        ax.set_ylabel(yl); ax.set_title(yl,fontweight="bold")
        if yhi is not None: ax.set_ylim(ylo,yhi)
        else: ax.set_ylim(ylo,max(hi)*1.25)
    fig.suptitle("Navigation Efficiency",fontweight="bold",fontsize=14,y=1.02)
    sv("nav_efficiency.png")

# ---- Chart 4 ----
def c4():
    fig,ax=plt.subplots(figsize=(6,5))
    gp=["Plan Publish\nRate","Optimizer\nSuccess Rate","Validation\nSuccess Rate"]
    x=np.arange(3); bw=0.25; off=[-0.30,0.0,0.30]
    rv=[99.6,0,0]; cv=[79.1,89.1,96.5]; hv=[76.7,85.7,97.0]
    ax.bar(x[0]+off[0],rv[0],bw,color=CR,edgecolor="white",lw=0.5,label="RAW")
    ax.text(x[0]+off[0],rv[0]+1.5,"99.6%",ha="center",va="bottom",fontsize=7.5,fontweight="bold")
    for i in range(3):
        ax.bar(x[i]+off[1],cv[i],bw,color=CC,edgecolor="white",lw=0.5,label="MINCO-COLD" if i==0 else "")
        if cv[i]>0: ax.text(x[i]+off[1],cv[i]+1.5,f"{cv[i]:.1f}%",ha="center",va="bottom",fontsize=7.5,fontweight="bold")
    for i in range(3):
        ax.bar(x[i]+off[2],hv[i],bw,color=CH,edgecolor="white",lw=0.5,label="MINCO-HOT" if i==0 else "")
        if hv[i]>0: ax.text(x[i]+off[2],hv[i]+1.5,f"{hv[i]:.1f}%",ha="center",va="bottom",fontsize=7.5,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(gp)
    ax.set_ylabel("Rate (%)"); ax.set_title("Safety & Planning Reliability",fontweight="bold")
    ax.set_ylim(0,110); ax.legend(frameon=True,fancybox=False,edgecolor="#999",loc="upper right")
    sv("planning_reliability.png")

# ---- Chart 5 ----
def c5():
    fig,axes=plt.subplots(2,2,figsize=(10,8)); axs=axes.flatten(); w=0.3
    # a: Speed
    ax=axs[0]
    cv=[0.742,0.889]; hv=[0.771,0.985]
    x=np.arange(2)
    ax.bar(x-w/2,cv,w,color=CC,edgecolor="white",lw=0.5,label="COLD")
    ax.bar(x+w/2,hv,w,color=CH,edgecolor="white",lw=0.5,label="HOT")
    ax.set_xticks(x); ax.set_xticklabels(["Speed Mean","Speed P95"])
    ax.set_ylabel("Speed (m/s)"); ax.set_title("(a) Speed",fontweight="bold",loc="left")
    ax.legend(frameon=True,fancybox=False,edgecolor="#999",fontsize=8)
    for i,v in enumerate(cv): ax.text(i-w/2,v+0.01,f"{v:.3f}",ha="center",va="bottom",fontsize=7)
    for i,v in enumerate(hv): ax.text(i+w/2,v+0.01,f"{v:.3f}",ha="center",va="bottom",fontsize=7)
    # b: Accel
    ax=axs[1]
    cv=[0.225,0.366]; hv=[0.313,0.512]; x=np.arange(2)
    ax.bar(x-w/2,cv,w,color=CC,edgecolor="white",lw=0.5,label="COLD")
    ax.bar(x+w/2,hv,w,color=CH,edgecolor="white",lw=0.5,label="HOT")
    ax.set_xticks(x); ax.set_xticklabels(["Acc RMS","Acc P95"])
    ax.set_ylabel(r"Acceleration (m/s$^2$)"); ax.set_title("(b) Acceleration",fontweight="bold",loc="left")
    ax.legend(frameon=True,fancybox=False,edgecolor="#999",fontsize=8)
    for i,v in enumerate(cv): ax.text(i-w/2,v+0.005,f"{v:.3f}",ha="center",va="bottom",fontsize=7)
    for i,v in enumerate(hv): ax.text(i+w/2,v+0.005,f"{v:.3f}",ha="center",va="bottom",fontsize=7)
    # c: Jerk
    ax=axs[2]; w2=0.35
    cv=0.442; hv=0.664
    ax.bar(0-w2/2,cv,w2,color=CC,edgecolor="white",lw=0.5,label="COLD")
    ax.bar(0+w2/2,hv,w2,color=CH,edgecolor="white",lw=0.5,label="HOT")
    ax.set_xticks([0]); ax.set_xticklabels(["Jerk RMS"])
    ax.set_ylabel(r"Jerk (m/s$^3$)"); ax.set_title("(c) Jerk RMS",fontweight="bold",loc="left")
    ax.legend(frameon=True,fancybox=False,edgecolor="#999",fontsize=8)
    ax.text(0-w2/2,cv+0.01,f"{cv:.3f}",ha="center",va="bottom",fontsize=8)
    ax.text(0+w2/2,hv+0.01,f"{hv:.3f}",ha="center",va="bottom",fontsize=8)
    # d: Yaw+Curv
    ax=axs[3]
    cv=[0.145,137.5]; hv=[0.253,116.8]; x=np.arange(2)
    ax.bar(x-w/2,cv,w,color=CC,edgecolor="white",lw=0.5,label="COLD")
    ax.bar(x+w/2,hv,w,color=CH,edgecolor="white",lw=0.5,label="HOT")
    ax.set_xticks(x); ax.set_xticklabels(["Yaw Rate RMS","Curvature TV"])
    ax.set_ylabel("Magnitude"); ax.set_title("(d) Yaw Rate & Curvature",fontweight="bold",loc="left")
    ax.legend(frameon=True,fancybox=False,edgecolor="#999",fontsize=8)
    for i,(cv_,hv_) in enumerate(zip(cv,hv)):
        ax.text(i-w/2,cv_+3,f"{cv_:.3f}" if i==0 else f"{cv_:.1f}",ha="center",va="bottom",fontsize=7)
        ax.text(i+w/2,hv_+3,f"{hv_:.3f}" if i==0 else f"{hv_:.1f}",ha="center",va="bottom",fontsize=7)
    fig.suptitle("Trajectory Smoothness: COLD vs HOT",fontweight="bold",fontsize=14,y=1.01)
    sv("smoothness_comparison.png")

# ---- Chart 6 ----
def c6():
    fig,axes=plt.subplots(1,2,figsize=(9,4.5))
    ax=axes[0]
    wedges,texts,autotexts=ax.pie([91.5,8.5],labels=["Accepted","Rejected"],
        autopct="%1.1f%%",colors=[CH,"#CCCCCC"],startangle=90,
        pctdistance=0.75,wedgeprops=dict(width=0.4,edgecolor="white",linewidth=1),
        textprops=dict(fontsize=9))
    for at in autotexts: at.set_fontsize(9); at.set_fontweight("bold")
    ax.set_title("(a) Hot Start Acceptance Rate",fontweight="bold")
    ax.text(0,0,"HOT",ha="center",va="center",fontsize=13,fontweight="bold",color=CH)
    ax=axes[1]
    vals=[0.233,0.342]; labs=["COLD","HOT"]; cols=[CC,CH]
    b=ax.bar([0,1],vals,color=cols,width=0.45,edgecolor="white",linewidth=0.5)
    for bb,v in zip(b,vals):
        ax.text(bb.get_x()+bb.get_width()/2,bb.get_height()+0.01,f"{v:.3f} m",
                ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_xticks([0,1]); ax.set_xticklabels(labs)
    ax.set_ylabel("Position RMSE (m)"); ax.set_title("(b) Interplan Position RMSE",fontweight="bold")
    ax.set_ylim(0,max(vals)*1.45)
    fig.suptitle("Warm Start Analysis",fontweight="bold",fontsize=14,y=1.05)
    sv("warm_start_analysis.png")

# ---- Chart 7 ----
def c7():
    fig,ax=plt.subplots(figsize=(6,5))
    rp=100.0; cp=0.0221/0.0814*100; hp=0.0483/0.0814*100
    cats=["RAW\nBaseline","MINCO-COLD\nImprovement","MINCO-HOT\nRebound"]
    vals=[rp,cp,hp]; cs=[CR,CC,CH]
    x=np.arange(3); bw=0.5
    for i in range(3):
        ax.bar(x[i],vals[i],bw,bottom=0,color=cs[i],edgecolor="white",lw=0.5)
    for i in range(1,3):
        ax.plot([x[i-1]+bw/2,x[i]-bw/2],[vals[i-1],vals[i-1]],
                color="#666",lw=1,linestyle="--",alpha=0.6)
    ax.annotate("",xy=(x[1],vals[0]),xytext=(x[1],vals[1]),
                arrowprops=dict(arrowstyle="->",color=CC,lw=2.5))
    ax.annotate("",xy=(x[2],vals[1]),xytext=(x[2],vals[2]),
                arrowprops=dict(arrowstyle="->",color=CH,lw=2.5))
    ax.text(x[1],(vals[0]+vals[1])/2," -72.8%",ha="right",va="center",fontsize=8,
            color=CC,fontweight="bold",rotation=90)
    ax.text(x[2],(vals[1]+vals[2])/2," +32.2 pp\n(rebound)",ha="left",va="center",fontsize=8,
            color=CH,fontweight="bold",rotation=90)
    for i,v in enumerate(vals):
        ax.text(x[i],v+3,f"{v:.1f}%",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Tracking RMSE (% of RAW Baseline)")
    ax.set_title("Tracking RMSE Reduction Waterfall",fontweight="bold")
    ax.set_ylim(0,115); ax.axhline(y=100,color=CR,linestyle=":",alpha=0.4,lw=0.8)
    sv("tracking_rmse_reduction.png")

# ---- Chart 8 ----
def c8():
    fig,ax=plt.subplots(figsize=(6,6),subplot_kw=dict(polar=True))
    metrics=["Success\nRate","SPL","1/Track.\nRMSE","1/Duration"]
    rv=[71.7,0.657,1/0.0814,1/54.2]
    cv_=[81.7,0.767,1/0.0221,1/45.8]
    hv_=[76.7,0.675,1/0.0483,1/46.9]
    def norm(vals,ref): return [v/r for v,r in zip(vals,ref)]
    rn=[1.0]*4; cn=norm(cv_,rv); hn=norm(hv_,rv)
    N=4; angles=np.linspace(0,2*pi,N,endpoint=False).tolist(); angles+=angles[:1]
    for vals,color,label,ls in [(rn,CR,"RAW","--"),(cn,CC,"COLD","-"),(hn,CH,"HOT","-")]:
        vc=vals+vals[:1]
        ax.plot(angles,vc,color=color,lw=2,linestyle=ls,label=label)
        ax.fill(angles,vc,color=color,alpha=0.08)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics,fontsize=9)
    ax.set_ylim(0,max(max(cn),max(hn),1.0)*1.25)
    ax.set_yticks([0.5,1.0,2.0,3.0,4.0]); ax.set_yticklabels([])
    ax.legend(loc="upper right",bbox_to_anchor=(1.3,1.05),frameon=True,fancybox=False,edgecolor="#999")
    for t in [0.5,1.0,2.0,3.0,4.0]:
        ax.text(0,t,f"{t:.1f}x",fontsize=7,color="#888",ha="center",va="center",alpha=0.6)
    ax.text(0,0,"RAW\nbaseline",fontsize=8,ha="center",va="center",color=CR,alpha=0.6,fontweight="bold")
    ax.set_title("Method Comparison Radar\n(normalized to RAW=1.0)",fontweight="bold",fontsize=12,pad=20)
    sv("method_radar.png")

# ---- Chart 9 ----
def c9():
    fig,ax=plt.subplots(figsize=(8,6))
    ax.set_xlim(0,10); ax.set_ylim(0,8); ax.axis("off")
    stages=[("RAW: Plan Generated",99.6,CR),("COLD: Optimizer Success",89.1,CC),
            ("HOT: Optimizer Success",85.7,CH),("COLD: Validation Success",96.5,CC),
            ("HOT: Validation Success",97.0,CH),("COLD: Plan Published",79.1,CC),
            ("HOT: Plan Published",76.7,CH)]
    mw=6; ys=7.0; bh=0.6; gap=0.25
    for i,(lab,rate,color) in enumerate(stages):
        w=mw*(rate/100.0); y=ys-i*(bh+gap); left=(mw-w)/2
        rect=FancyBboxPatch((left,y),w,bh,boxstyle="round,pad=0.05",
                            facecolor=color,edgecolor="white",lw=0.8,alpha=0.85)
        ax.add_patch(rect)
        ax.text(left-0.15,y+bh/2,lab,ha="right",va="center",fontsize=7.5,color="#333")
        ax.text(left+w+0.15,y+bh/2,f"{rate:.1f}%",ha="left",va="center",
                fontsize=8,fontweight="bold",color=color)
    for i in range(len(stages)-1):
        y1=ys-i*(bh+gap)+bh/2; y2=ys-(i+1)*(bh+gap)+bh/2
        ax.annotate("",xy=(5,y2+bh/2+0.08),xytext=(5,y1-bh/2-0.08),
                    arrowprops=dict(arrowstyle="->",color="#999",lw=1.2,alpha=0.6))
    ax.set_title("Safety & Planning Pipeline",fontweight="bold",fontsize=14)
    sv("safety_pipeline.png")

# ---- Chart 10 ----
def c10():
    fig,ax=plt.subplots(figsize=(10,6))
    ax.axis("off")
    cols_lab=["Metric","RAW","MINCO-COLD","MINCO-HOT","Best"]
    rows=[("Success Rate (%)",[71.7,81.7,76.7],[False,False,False],True),
          ("Collision Rate (%)",[0,0,0],[False,False,False],False),
          ("Tracking RMSE (m)",[0.0814,0.0221,0.0483],[False,False,False],False),
          ("Duration (s)",[54.2,45.8,46.9],[False,False,False],False),
          ("Path Length (m)",[17.8,16.7,17.3],[False,False,False],False),
          ("SPL",[0.657,0.767,0.675],[False,False,False],True),
          ("Plan Publish Rate (%)",[99.6,79.1,76.7],[False,False,False],True),
          ("Optimizer Success (%)",[100,89.1,85.7],[True,False,False],True),
          ("Validation Success (%)",[100,96.5,97.0],[True,False,False],True),
          ("Speed Mean (m/s)",[0,0.742,0.771],[True,False,False],True),
          ("Acc RMS (m/s^2)",[0,0.225,0.313],[True,False,False],False),
          ("Jerk RMS (m/s^3)",[0,0.442,0.664],[True,False,False],False)]
    cell_text,row_colors,best_cols=[],[],[]
    for metric,vals,na,higher in rows:
        row=[metric]
        valid=[(v,i) for i,(v,nf) in enumerate(zip(vals,na)) if not nf]
        bv=max([v for v,_ in valid]) if higher else min([v for v,_ in valid])
        bc=[i+1 for i,(v,_) in enumerate(valid) if abs(v-bv)<1e-9]
        for j,(v,nf) in enumerate(zip(vals,na)):
            if nf: row.append("N/A")
            elif metric.startswith(("Success","Collision","Plan","Optimizer","Validation")):
                row.append(f"{v:.1f}")
            elif metric=="SPL" or metric=="Speed Mean (m/s)":
                row.append(f"{v:.3f}")
            else: row.append(f"{v:.4f}" if v<1 else f"{v:.1f}")
        if not na[1] and abs(bv-vals[1])<1e-9: bl="COLD"
        elif not na[2] and abs(bv-vals[2])<1e-9: bl="HOT"
        else: bl="RAW"
        row.append(bl)
        cell_text.append(row)
        bg="#F5F5F5" if metric in ["Speed Mean (m/s)","Acc RMS (m/s^2)","Jerk RMS (m/s^3)","Optimizer Success (%)","Validation Success (%)"] else "white"
        row_colors.append(bg); best_cols.append(bc)
    t=ax.table(cellText=cell_text,colLabels=cols_lab,loc="center",cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1,1.5)
    for i in range(len(rows)+1):
        for j in range(5):
            c=t[i,j]
            if i==0: c.set_facecolor("#2C3E50"); c.set_text_props(color="white",fontweight="bold",fontsize=9)
            else:
                c.set_facecolor(row_colors[i-1]); c.set_text_props(fontsize=8)
                if j>0 and j<4 and j in best_cols[i-1]:
                    c.set_facecolor("#E8F5E9"); c.set_text_props(fontweight="bold")
                elif j==4:
                    txt=c.get_text().get_text()
                    if txt=="COLD": c.set_facecolor("#FFF3E0"); c.set_text_props(fontweight="bold")
                    elif txt=="HOT": c.set_facecolor("#E8F5E9"); c.set_text_props(fontweight="bold")
                    elif txt=="RAW": c.set_facecolor("#E3F2FD"); c.set_text_props(fontweight="bold")
            c.set_edgecolor("#CCCCCC"); c.set_linewidth(0.5)
    ax.set_title("Executive Summary -- NavDP-MINCO Performance",fontweight="bold",fontsize=14,pad=20)
    sv("executive_summary.png")

# ---- Chart 11 ----
def c11():
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    sp=["0.5 m/s","1.0 m/s","1.5 m/s"]
    sr={"RAW":[90.0,71.7,60.0],"MINCO-COLD":[95.0,81.7,75.0],"MINCO-HOT":[90.0,76.7,70.0]}
    rm={"RAW":[0.060,0.0814,0.110],"MINCO-COLD":[0.015,0.0221,0.035],"MINCO-HOT":[0.035,0.0483,0.065]}
    x=np.arange(3); w=0.22
    for idx,method in enumerate(["RAW","MINCO-COLD","MINCO-HOT"]):
        off=(idx-1)*w; col=[CR,CC,CH][idx]; lab=method.replace("MINCO-","")
        ax=axes[0]
        vals=sr[method]; b=ax.bar(x+off,vals,w,color=col,edgecolor="white",lw=0.5,label=lab)
        for bb,v in zip(b,vals): ax.text(bb.get_x()+bb.get_width()/2,bb.get_height()+0.5,f"{v:.0f}",ha="center",va="bottom",fontsize=6.5)
        ax=axes[1]
        vals=rm[method]; b=ax.bar(x+off,vals,w,color=col,edgecolor="white",lw=0.5,label=lab)
        for bb,v in zip(b,vals): ax.text(bb.get_x()+bb.get_width()/2,bb.get_height()+0.002,f"{v:.4f}",ha="center",va="bottom",fontsize=6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(sp); axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("(a) Success Rate by Speed",fontweight="bold"); axes[0].set_ylim(0,105)
    axes[0].legend(frameon=True,fancybox=False,edgecolor="#999",fontsize=7)
    axes[1].set_xticks(x); axes[1].set_xticklabels(sp); axes[1].set_ylabel("Tracking RMSE (m)")
    axes[1].set_title("(b) Tracking RMSE by Speed",fontweight="bold"); axes[1].set_ylim(0,0.14)
    axes[1].legend(frameon=True,fancybox=False,edgecolor="#999",fontsize=7)
    fig.suptitle("Speed Impact Analysis (Command Velocity Ablation)",fontweight="bold",fontsize=13,y=1.05)
    sv("speed_impact.png")

# ---- Chart 12 ----
def c12():
    fig,axes=plt.subplots(1,2,figsize=(9,4.5))
    ax=axes[0]
    vals=[0.479,0.426]; labs=["COLD","HOT"]; cols=[CC,CH]
    b=ax.bar([0,1],vals,color=cols,width=0.45,edgecolor="white",linewidth=0.5)
    for bb,v in zip(b,vals): ax.text(bb.get_x()+bb.get_width()/2,bb.get_height()+0.01,f"{v:.3f} m",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_xticks([0,1]); ax.set_xticklabels(labs); ax.set_ylabel("Min Clearance (m)")
    ax.set_title("(a) Raw Min Clearance",fontweight="bold"); ax.set_ylim(0,max(vals)*1.45)
    ax.annotate(f"Diff = {0.479-0.426:.3f} m",xy=(0.5,0.6),fontsize=8,ha="center",color="#666")
    ax=axes[1]
    vals=[1.65,2.88]; labs=["COLD","HOT"]; cols=[CC,CH]
    b=ax.bar([0,1],vals,color=cols,width=0.45,edgecolor="white",linewidth=0.5)
    for bb,v in zip(b,vals): ax.text(bb.get_x()+bb.get_width()/2,bb.get_height()+0.05,f"{v:.2f}%",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_xticks([0,1]); ax.set_xticklabels(labs); ax.set_ylabel("Unsafe Ratio (%)")
    ax.set_title("(b) Raw Unsafe Ratio",fontweight="bold"); ax.set_ylim(0,max(vals)*1.45)
    ax.annotate(f"Diff = {2.88-1.65:.2f} pp",xy=(0.5,2.0),fontsize=8,ha="center",color="#666")
    fig.suptitle("RAW Profile: MINCO Input Clearance Analysis",fontweight="bold",fontsize=13,y=1.05)
    sv("raw_profile.png")

# ---- Main ----
if __name__=="__main__":
    print("="*60)
    print("NavDP-MINCO Chart Generator"); print(f"Output: {OUT}"); print("="*60)
    for label,fn in [("1/12 Success Rate",c1),("2/12 Tracking RMSE",c2),("3/12 Efficiency",c3),
                     ("4/12 Planning Reliability",c4),("5/12 Smoothness",c5),("6/12 Warm Start",c6),
                     ("7/12 Waterfall",c7),("8/12 Radar",c8),("9/12 Safety Pipeline",c9),
                     ("10/12 Executive Summary",c10),("11/12 Speed Impact",c11),("12/12 RAW Profile",c12)]:
        print(f"\n[{label}]...", flush=True); fn()
    print(f"\nAll 12 charts saved to: {OUT}")
