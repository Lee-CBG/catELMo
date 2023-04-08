# Context-Aware Amino Acid Embedding Advances Analysis of TCR-Epitope Interactions

catELMo is a bi-directional amino acid embedding model that learns contextualized amino acid representations, treating an amino acid as a word and a sequence as a sentence. It learns patterns of amino acid sequences with its self-supervision signal, by predicting each the next amino acid token given its previous tokens. It has been trained on 4,173,895 TCR $\beta$ CDR3 sequences (52 million of amino acid tokens) from [ImmunoSEQ](https://www.immunoseq.com/). catELMo yields a real-valued representation vector for a sequence of amino acids, which can be used as input features of various downstream tasks. This is the official implementation of catELMo.
<br/>
<br/>

<p align="center"><img width=100% alt="Overview" src="https://github.com/Lee-CBG/catELMo/blob/main/figures/Fig4_Methods.png"></p>

## Dependencies

+ Linux
+ Python 3.6.13
+ Keras 2.6.0
+ TensorFlow 2.6.0

## Steps to train a Binding Affinity Prediction model for TCR-epitope pairs.

### 1. Clone the repository
```bash
git clone https://github.com/Lee-CBG/catELMo
cd catELMo/
conda create --name bap python=3.6.13
pip install -r requirements.txt
source activate bap
```

### 2. Prepare TCR-epitope pairs for training and testing
- Download training and testing data from `datasets` folder.
- Obtain embeddings for TCR and epitopes following instructions from `embedders` folder.


### 3. Train models
An example for training the transformer-based model

```bash
python -W ignore bap.py \
                --embedding catELMo_4_layers_1024 \
                --split epitope \
                --gpu 0 \
                --fraction 1 \
                --seed 42
```

## Citation
If you use this code or use our catELMo for your research, please cite our paper:
```

```

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
