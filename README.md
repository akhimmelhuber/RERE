# RERE

This repository contains the source code for the paper 'Receptive Field Reducer for Explaining GraphNeural Networks'

#### GCN model 

This is the model that will be explained. We do provide pretratined models for all of the experiments
that are shown in the paper, which can be found in RERE/ckpt.

#### Training a Policy

To train a policy for node classification, run:

```
python rere.py --dataset=DATASET --explain_node=None
```
with Datasets containing [syn1, syn4, syn5]

To train a policy for graph classification 'Mutagenicity' run":
```
python rere.py --dataset=Mutagenicity --explain_node=None --input_dim=21 --graph-idx=3 --graph-mode 
```

The trained policy can be found in RERE/log/models


#### Applying a Policy

To run the explainer for node classification tasks [syn1, syn4, syn5], run the following:

```
python rere.py --dataset=DATASET
```

To run the explainer for the graph classification example "Mutagenicity" run:

```
python rere.py --dataset=Mutagenicity --input_dim=21 --graph-idx=3 --graph-mode
```
To run the explainer for the graph classification example "REDDIT-BINARY" run:

```
python rere.py --dataset=REDDIT-BINARY --graph-idx=3 --graph-mode
```

The policy can be changed in rere.py file. 


