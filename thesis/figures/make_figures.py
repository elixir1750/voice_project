import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":11,"savefig.bbox":"tight"})
FIG="figures/"

# ---- Fig: WER/CER by layer ----
layers=[6,9,12]; wer=[0.4138,0.2954,0.8596]; cer=[0.1186,0.0768,0.3426]
fig,ax=plt.subplots(figsize=(3.4,2.5))
ax.plot(layers,wer,"o-",color="#c0504d",label="WER")
ax.plot(layers,cer,"s--",color="#4a7fb5",label="CER")
for x,y in zip(layers,wer): ax.annotate(f"{y:.2f}",(x,y),textcoords="offset points",xytext=(0,7),ha="center",fontsize=9)
ax.set_xticks(layers); ax.set_xlabel("wav2vec2 layer"); ax.set_ylabel("Error rate")
ax.set_ylim(0,0.95); ax.legend(frameon=False,fontsize=10); ax.grid(alpha=.3)
plt.savefig(FIG+"fig_layer.pdf"); plt.close()

# ---- Fig: WER vs params (decoder) ----
names=["Linear","MLP","Transformer"]; params=[23070,410654,1784094]
werd=[0.6027,0.2954,0.3163]; rtf=[0.0102,0.0111,0.0104]
fig,ax=plt.subplots(figsize=(3.4,2.5))
ax.plot(params,werd,"o-",color="#5a9c6b",markersize=7)
for n,p,w,r in zip(names,params,werd,rtf):
    ax.annotate(f"{n}\nWER={w:.2f}, RTF={r:.3f}",(p,w),textcoords="offset points",
                xytext=(0,10 if n!="MLP" else -28),ha="center",fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("Decoder parameters (log)"); ax.set_ylabel("WER")
ax.set_ylim(0.20,0.70); ax.grid(alpha=.3,which="both")
plt.savefig(FIG+"fig_decoder.pdf"); plt.close()

# ---- Fig: WER/CER vs data ----
n=[900,1800,3600,5400,7200]; werx=[0.4278,0.3449,0.2954,0.2579,0.2493]; cerx=[0.1174,0.0911,0.0768,0.0674,0.0634]
fig,ax=plt.subplots(figsize=(3.4,2.5))
ax.plot(n,werx,"o-",color="#c0504d",label="WER")
ax.plot(n,cerx,"s--",color="#4a7fb5",label="CER")
ax.set_xlabel("Training utterances"); ax.set_ylabel("Error rate")
ax.legend(frameon=False,fontsize=10); ax.grid(alpha=.3); ax.set_ylim(0,0.47)
plt.savefig(FIG+"fig_data.pdf"); plt.close()

# ---- Fig: system pipeline (English, simple) ----
BLUE=("#e8eef7","#5b7fb4"); OR=("#fdeede","#d08a3e"); GR=("#e7f3ea","#5a9c6b")
fig,ax=plt.subplots(figsize=(3.3,4.6)); ax.set_xlim(0,10); ax.set_ylim(0,12.4); ax.axis("off")
def box(y,t,c,h=1.0):
    fc,ec=c; ax.add_patch(FancyBboxPatch((1,y-h/2),8,h,boxstyle="round,pad=0.05,rounding_size=0.1",
        lw=1.4,ec=ec,fc=fc)); ax.text(5,y,t,ha="center",va="center",fontsize=8.5)
    return y
ys=[11.2,9.4,7.6,5.8,4.0,2.2,0.7]
labs=[("Waveform 16 kHz",GR),
      ("Frozen wav2vec2: CNN (320x down)\n+ 12 Transformer layers, 50 fps",BLUE),
      ("Tap hidden states of layer l\n(l in {6, 9, 12})",GR),
      ("Representation (continuous)",OR),
      ("CTC head: linear / MLP / Transformer\nframe logits [T, 30]",OR),
      ("CTC loss (train) / greedy (infer)",OR),
      ("WER / CER / RTF",GR)]
for y,(t,c) in zip(ys,labs):
    hh=1.2 if "\n" in t else 0.9; box(y,t,c,hh)
for a,b in zip(ys[:-1],ys[1:]):
    ax.add_patch(FancyArrowPatch((5,a-0.62),(5,b+0.62),arrowstyle="-|>",mutation_scale=13,lw=1.5,color="#777"))
plt.savefig(FIG+"fig_system.pdf"); plt.close()
print("figures done")
