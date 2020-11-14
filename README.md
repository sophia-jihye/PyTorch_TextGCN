# Modifications by Jihye
* Multi-label classification
    - `data/PatentDataImporter.py`
        - `data/textdataset/Patent.txt`
            - multi-label 형태로 변환
    - `trainer.py`
        - MultiLabelBinarizer
            - label을 index가 아닌 One-hot encoding 형태로 변환
        - y.float() & torch.nn.BCEWithLogitsLoss
            - float타입의 logits 값과 float타입의 one-hot encoded y 값을 비교하여 loss 계산
    - `utils.py`
        - Accuracy, MacroF1
            - logits값을 sigmoid함수 통해 변환한 후 0.5 이상이면 1, 0.5 미만이면 0으로 변환 후 계산
            - TP,FP,FN 계산 식 수정 
* Dataset: [USPTO-2M](http://mleg.cse.sc.edu/DeepPatent/)
    - 
```
2015_USPTO.json 
+---------+-----------+-------------+------------+------------+
|  Nodes  |   Edges   |  Selfloops  |  Isolates  |  Coverage  |
+---------+-----------+-------------+------------+------------+
|  68337  |  6226625  |      0      |     0      |   1.0000   |
+---------+-----------+-------------+------------+------------+
```

*****

# Graph Convolutional Networks for Text Classification in PyTorch

PyTorch 1.6 and Python 3.7 implementation of Graph Convolutional Networks for Text Classification [1].

Tested on the 20NG/R8/R52/Ohsumed/MR data set, the code on this repository can achieve the effect of the paper.

## Benchmark

* MR Dataset
```
+---------+----------+-------------+------------+------------+
|  Nodes  |  Edges   |  Selfloops  |  Isolates  |  Coverage  |
+---------+----------+-------------+------------+------------+
|  29426  |  918186  |      0      |     0      |   1.0000   |
+---------+----------+-------------+------------+------------+

  (gc1): GraphConvolution (29426 -> 200)
  (gc2): GraphConvolution (200 -> 2)
```

| dataset       | 20NG | R8 | R52 | Ohsumed | MR  |
|---------------|----------|------|--------|--------|--------|
| TextGCN(official) | 0.8634    | 0.9707 | 0.9356   | 0.6836   | 0.7674   |
| This repo.    | 0.8618     | 0.9704 | 0.9354   | 0.6827  | 0.7643  |

NOTE: The result of the experiment is to repeat the run 10 times, and then take the average of accuracy.

## Requirements
* fastai==2.0.15
* PyTorch==1.6.0
* scipy==1.5.2
* pandas==1.0.1
* spacy==2.3.1
* nltk==3.5
* prettytable==1.0.0
* numpy==1.18.5
* networkx==2.5
* tqdm==4.49.0
* scikit_learn==0.23.2

## Usage
1. Process the data first, run `data_processor.py` (Already done)
2. Generate graph, run `build_graph.py` (Already done)
3. Training model, run `trainer.py`

## References
[1] [Yao, L. , Mao, C. , & Luo, Y. . (2018). Graph convolutional networks for text classification.](https://arxiv.org/abs/1809.05679)
