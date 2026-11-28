"""在本地 FCE 测试集上重评 BERT 与 CNN；不生成不存在的训练曲线。"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tomllib
from zoneinfo import ZoneInfo
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bert_detect import BertDetector
from detect import Detector, find_latest_weights
from util.TSV import read_fce_tsv

ROOT = Path(__file__).resolve().parent

def metrics(labels, scores, threshold):
    predicted = scores >= threshold
    tp = int(((labels == 1) & predicted).sum()); fp = int(((labels == 0) & predicted).sum())
    tn = int(((labels == 0) & ~predicted).sum()); fn = int(((labels == 1) & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return dict(accuracy=(tp + tn) / len(labels), precision=precision, recall=recall,
                f05=1.25*precision*recall/(.25*precision+recall) if precision+recall else 0,
                confusion_matrix=[[tn,fp],[fn,tp]])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'config.toml')
    parser.add_argument('--cnn-weights',type=Path)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    torch.set_num_threads(4)
    config_path=args.config.resolve()
    with config_path.open('rb') as f: config=tomllib.load(f)
    test_path=Path(config['paths']['test'])
    if not test_path.is_absolute(): test_path=config_path.parent/test_path
    samples=read_fce_tsv(test_path)
    output=ROOT/'result/comparison'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d/%H-%M-%S')
    output.mkdir(parents=True,exist_ok=False)
    bert=BertDetector(config_path=config_path)
    bert_scores=[]; unused=[]
    for start in tqdm(range(0,len(samples),args.batch_size),desc='BERT 测试'):
        for row in bert.predict_words_batch([s[0] for s in samples[start:start+args.batch_size]],args.batch_size):
            bert_scores.extend(p[0] for p in row); unused.extend(p[2] for p in row)
    cnn_path=args.cnn_weights or find_latest_weights(config_path)
    cnn=Detector(cnn_path,config_path=config_path)
    cnn_scores=[]
    with torch.inference_mode():
        for words,_ in tqdm(samples,desc='CNN 测试'):
            ids=torch.tensor([[cnn.word_to_id.get(w,cnn.word_to_id['<UNK>']) for w in words]])
            cnn_scores.extend(cnn.model(ids).softmax(-1)[0,:,1].tolist())
    labels=np.array([y for _,ys in samples for y in ys]); threshold=bert.threshold
    scores={'BERT':np.array(bert_scores),'CNN':np.array(cnn_scores)}
    assert all(len(s)==len(labels) for s in scores.values())
    summary={'test_file':str(test_path),'test_sha256':hashlib.sha256(test_path.read_bytes()).hexdigest(),
             'sentences':len(samples),'tokens':len(labels),'threshold':threshold,'bert_weights':str(bert.weights),
             'cnn_weights':str(cnn_path),'alignment':'First subword per original TSV word; special tokens excluded; whole-word chunks <=256 subwords.',
             'label_mapping':'BERT 0=error, 1=correct, 2=unused extra output. Raw three-class softmax P(0) compared to threshold.',
             'limitations':'Local FCE test reevaluation, not a reproduction of author metrics. External training overlap not independently audited. Threshold curve is descriptive, not test-set threshold tuning.',
             'bert_unused_probability_mean':float(np.mean(unused)),
             'metrics':{name:metrics(labels,s,threshold) for name,s in scores.items()}}
    (output/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (output/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['sentence','word_index','word','label','bert_error_probability','cnn_error_probability'])
        index=0
        for si,(words,ys) in enumerate(samples):
            for wi,(word,label) in enumerate(zip(words,ys)):
                writer.writerow([si,wi,word,label,bert_scores[index],cnn_scores[index]]);index+=1
    names=['accuracy','precision','recall','f05']; x=np.arange(4)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for i,name in enumerate(scores):
        bars=ax.bar(x+(i-.5)*.35,[summary['metrics'][name][m] for m in names],.35,label=name)
        ax.bar_label(bars,fmt='%.3f',padding=3)
    ax.set(xticks=x,xticklabels=names,ylim=(0,1.12),title=f'FCE test metrics | threshold={threshold:.2f}');ax.legend()
    fig.savefig(output/'metrics_comparison.png',dpi=160);plt.close(fig)
    for name,s in scores.items():
        cm=np.array(summary['metrics'][name]['confusion_matrix'])
        fig,ax=plt.subplots(figsize=(6,5),layout='constrained');im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
        for y in range(2):
            for x in range(2): ax.text(x,y,str(cm[y,x]),ha='center',va='center',color='white' if cm[y,x]>cm.max()/2 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Correct','Error'],yticklabels=['Correct','Error'],xlabel='Predicted',ylabel='True',title=f'{name} | threshold={threshold:.2f}')
        fig.savefig(output/f'{name.lower()}_confusion_matrix.png',dpi=160);plt.close(fig)
    thresholds=np.linspace(0,1,101)
    with (output/'threshold_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['model','threshold',*names])
        fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
        for name,s in scores.items():
            values=[metrics(labels,s,t) for t in thresholds]
            for t,m in zip(thresholds,values):writer.writerow([name,t,*[m[n] for n in names]])
            ax.plot(thresholds,[m['f05'] for m in values],label=name)
        ax.axvline(threshold,color='gray',ls='--');ax.set(xlabel='Threshold',ylabel='F0.5',title='Test-set threshold sensitivity (descriptive only)');ax.legend()
        fig.savefig(output/'threshold_f05.png',dpi=160);plt.close(fig)
    print(json.dumps(summary['metrics'],indent=2));print(output)

if __name__=='__main__': main()
