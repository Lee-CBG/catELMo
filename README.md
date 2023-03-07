# Cracking TCR-Epitope Interations using Language Model Representations

catELMo is a bi-directional amino acid embedding model that learns contextualized amino acid representations, treating an amino acid as a word and a sequence as a sentence. It learns patterns of amino acid sequences with its self-supervision signal, by predicting each the next amino acid token given its previous tokens. It has been trained on 4,173,895 TCR$\beta$ CDR3 sequences (52 million of amino acid tokens) from [ImmunoSEQ](https://www.immunoseq.com/). catELMo yields a real-valued representation vector for a sequence of amino acids, which can be used as input features of various downstream tasks. This is the official implementation of catELMo.
<br/>
<br/>

<p align="center"><img width=100% alt="Overview" src="https://github.com/Lee-CBG/catELMo/blob/main/figures/Fig4_Methods.png"></p>

## Dependencies

+ Linux
+ Python 3.6.13
+ Keras 2.6.0
+ TensorFlow 2.6.0

## Want to embed your sequences using catELMo?

## Want to training your own catELMo on customed data?

## Want to fine-tune from our catELMo?
You can download the weights of our catELMo from below links.

## Citation
If you use this code or use our catELMo for your research, please cite our paper:
```

```

## License

Released under the [ASU GitHub Project License](./LICENSE).
