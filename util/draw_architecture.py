"""按当前代码绘制识别系统与 CNN 架构，输出 PNG/SVG/PDF。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).resolve().parents[1] / 'docs/architecture'
FONT = FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc')
COLORS = dict(input='#dceff0', process='#ffe38a', cnn='#bde9fb', bert='#d9d2f2', output='#f6ced4', mask='#d7e9c7')

def text(ax,x,y,s,size=11,**kw):
    ax.text(x,y,s,fontproperties=FONT,fontsize=size,ha='center',va='center',color='#20313e',**kw)

def box(ax,x,y,w,h,s,color='process',size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.08',facecolor=COLORS.get(color,color),edgecolor='#50616c',lw=1.1))
    text(ax,x+w/2,y+h/2,s,size)

def arrow(ax,a,b,dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.25,color='#50616c',linestyle='--' if dashed else '-'))

def panel(ax,x,y,w,h,title,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.18',facecolor=color,edgecolor='#9baab4',lw=1))
    text(ax,x+w/2,y+h-.3,title,13)

def canvas(w,h,title):
    fig,ax=plt.subplots(figsize=(w,h));fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.02)
    ax.set(xlim=(0,w),ylim=(0,h));ax.axis('off');text(ax,w/2,h-.35,title,20)
    return fig,ax

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=220,facecolor='white')
    plt.close(fig)

fig,ax=canvas(18,12,'英文作文识别与错误检测系统结构图')
text(ax,9,11.16,'依据 web_detect.py、OCR.py、detect.py、bert_detect.py 与当前 config.toml · 2026-11-27',10)
panel(ax,.25,8.65,17.5,2.12,'① 图片输入与 OCR 识别','#f1f7f7')
items=[(.5,2.7,'图片上传 / 摄像头拍照','input'),(3.65,2.7,'Flask 接口\nPOST /api/recognize','input'),(6.8,2.7,'图片校验\nEXIF 方向校正 → RGB','process'),(9.95,3.0,'Apple Vision OCR\n英文 · Accurate · 关闭语言纠正','process'),(13.4,3.85,'OCR 文本 + 行框 + 置信度\n按标点 / 换行分句 → 规则分词','output')]
for x,w,s,c in items:box(ax,x,9.05,w,.9,s,c,10)
for (x,w,_,_),(nx,_,_,_) in zip(items,items[1:]):arrow(ax,(x+w,9.5),(nx,9.5))
box(ax,6.8,7.65,4.4,.65,'配置选择 backend：bert / cnn（二选一）','input',12)
arrow(ax,(15.35,9.05),(15.35,8.35));arrow(ax,(15.35,8.35),(9,8.35));arrow(ax,(9,8.35),(9,8.3))
panel(ax,.25,2.25,8.5,4.95,'② A · BERT 检测分支（当前启用）','#f4f1fa')
panel(ax,9.25,2.25,8.5,4.95,'② B · 三层空洞 CNN（配置切换后使用）','#edf7fc')
arrow(ax,(7.7,7.65),(4.5,7.2));arrow(ax,(10.3,7.65),(13.5,7.2),True)
bert=[('WordPiece 分词 + 特殊 token\n按完整词分块，每块 ≤ 256 子词', 'bert'),('BERT Embedding · hidden = 768\n词嵌入 + 位置嵌入 + 类型嵌入','bert'),('Transformer Encoder × 12\n每层 12 个注意力头 · FFN 768 → 3072 → 768','bert'),('Linear 768 → 3 → Softmax\n取原始词首个子词的 P(类别 0)','output')]
cnn=[('词表编码（未登录词 → UNK）\nGloVe Embedding · 100 维 · 冻结','cnn'),('Embedding Dropout(0.2) → 转置\n[B, L, 100] → [B, 100, L] → PAD 清零','cnn'),('空洞 Conv1d × 3 · kernel = 7\ndilation = 1 / 2 / 4 · channels = 128','cnn'),('转置 → Linear 128 → 2 → Softmax\n取每个原始词的 P(类别 1)','output')]
for x,rows in [(1,bert),(10,cnn)]:
    for i,(s,c) in enumerate(rows):
        y=5.75-i*.94;box(ax,x,y,7,.69,s,c,10)
        if i<3:arrow(ax,(x+3.5,y),(x+3.5,y-.25))
text(ax,4.5,2.52,'BERT：0 = 错误，1 = 正确，2 = 未对应有效训练标签',9)
text(ax,13.5,2.52,'CNN：0 = 正确，1 = 错误；每层卷积后屏蔽 PAD',9)
box(ax,5.25,1.12,7.5,.66,'P(错误) ≥ 0.30 → 疑似错误词；恢复整篇文本字符偏移','output',11)
arrow(ax,(4.5,2.25),(4.5,1.45));arrow(ax,(4.5,1.45),(5.25,1.45));arrow(ax,(13.5,2.25),(13.5,1.45));arrow(ax,(13.5,1.45),(12.75,1.45))
box(ax,5.25,.15,7.5,.6,'网页展示：图片预览 + OCR 原文 + 错误高亮 + 错误概率','input',11);arrow(ax,(9,1.12),(9,.75))
text(ax,2.55,.58,'模型均为 eval / 无梯度推理\nDropout 在检测时关闭',9)
text(ax,15.3,.58,'摄像头受角度和对焦影响\n建议正对纸面拍照上传',9)
save(fig,'recognition_system')

fig,ax=canvas(18,10,'三层空洞 CNN · 词级错误检测网络结构')
text(ax,9,9.15,'model.py / ErrorModel · 当前配置 kernel_size=7，dilations=[1,2,4]，conv_channels=128',11)
panel(ax,4.25,5.65,13.4,2.8,'空洞卷积模块展开（每层均执行）','#fff8df')
for x,w,s,c in [(4.55,2.15,'Conv1d\nk=7，d=d_i','process'),(7.15,1.55,'ReLU','process'),(9.15,2.3,'Dropout\np=0.3','output'),(11.9,2.35,'PAD 位置清零','mask'),(14.7,2.6,'输出特征\n[B,128,L]','cnn')]:box(ax,x,6.55,w,.8,s,c,11)
for a,b in [(6.7,7.15),(8.7,9.15),(11.45,11.9),(14.25,14.7)]:arrow(ax,(a,6.95),(b,6.95))
text(ax,10.95,6.02,'stride=1；padding=d_i×(7-1)/2，序列长度 L 保持不变',11)
main=[('输入 token ID\n[B,L]','input'),('GloVe Embedding\n100 维 · freeze=True','cnn'),('Dropout(0.2)\n转置 + PAD 清零','output'),('空洞卷积模块 1\n100 → 128 · d=1 · p=3','cnn'),('空洞卷积模块 2\n128 → 128 · d=2 · p=6','cnn'),('空洞卷积模块 3\n128 → 128 · d=4 · p=12','cnn'),('转置 + Linear(128,2)\n[B,L,2]','process'),('Softmax → P(错误类别 1)\n≥ 0.30 标为疑似错误','output')]
for i,(s,c) in enumerate(main):
    y=8.05-i*1.02;box(ax,.3,y,3.45,.74,s,c,10)
    if i<7:arrow(ax,(2.025,y),(2.025,y-.28))
arrow(ax,(3.75,5.36),(4.25,6.95),True)
panel(ax,4.25,2.65,13.4,2.5,'局部上下文覆盖范围（感受野）','#edf7fc')
for x,s in [(4.6,'第 1 层\n7 个 token'),(8.9,'第 2 层\n19 个 token'),(13.2,'第 3 层\n43 个 token')]:box(ax,x,3.5,3.6,.85,s,'cnn',13)
arrow(ax,(8.2,3.92),(8.9,3.92));arrow(ax,(12.5,3.92),(13.2,3.92))
text(ax,10.95,2.98,'R = 1 + (k - 1) × Σ dilation = 1 + 6 × (1 + 2 + 4) = 43',12)
panel(ax,4.25,.35,13.4,1.8,'训练与检测说明','#f4f6f8')
text(ax,10.95,1.37,'训练：加权交叉熵 [正确=1.0, 错误=2.0]；标签 PAD=-100，不计入损失',11)
text(ax,10.95,.9,'检测：model.eval()，全部 Dropout 关闭；保留每个词的位置，不使用池化',11)
text(ax,10.95,.54,'此图描述 CNN 路径；网页当前默认后端为 BERT，二者不是串联网络。',10)
save(fig,'dilated_cnn_structure')
print(OUT)
