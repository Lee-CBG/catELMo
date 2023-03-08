# catELMo Usage

This repository describes how to embed amino acids of TCR and epitope sequences into real-valued representations.

## Our available pretrained weights
Each of those embedding models were trained on the same TCR repertoires (more than four million TCR sequences) collected from ImmunoSEQ. We refer training details to [here](https://github.com/allenai/bilm-tf) and [here](https://github.com/google-research/bert). 

| Embedding models      |Backbone Structures| Number of Backbone layers | Embedding Size | 
|---------------------|:--:|:------------:| :-----------:|
| [catELMo](https://www.dropbox.com/sh/jpw6z71bsn1t7ev/AADRiL7_amT0vQrpep45PcOPa?dl=0)              |`Bidirectional LSTM`| 4 | 1,024 |
| [catELMo-Shallow](https://www.dropbox.com/sh/4no85yecsuaiiw4/AAA2UxA5E9RNdjPBleYITXhsa?dl=0)               |`Bidirectional LSTM`| 2 | 1,024 |
| [catELMo-Deep](https://www.dropbox.com/sh/ua1x0ateod5ntui/AAANxH8OrJn_pcZY8SwyiypDa?dl=0)            |`Bidirectional LSTM`| 8 | 1,024 |
| [BERT-Tiny-TCR](https://www.dropbox.com/sh/at9j5gtt0a46wy4/AABWZpoSWmf_R3DVNi8mtjrJa?dl=0)            |`Transformer Encoder`| 2 | 768 |
| [BERT-Base-TCR](https://www.dropbox.com/sh/bz6fx2l8fwbtlpz/AADPaaVo4gZ6OhivkpzqynQ3a?dl=0)            |`Transformer Encoder`| 12 | 768 |
| [BERT-Large-TCR](https://www.dropbox.com/sh/xswmoi5tnlc1nuj/AACIGo1MW_5zx6lmgGaeTCT0a?dl=0)           |`Transformer Encoder`| 30 | 1,024 |


## How to embed?

### 1. Requirements
python 3.6, tensorflow 1.4.0, allennlp 0.9.0, torch 1.9.1 and other common packages listed in `catELMo.yml`.

### 2. Installation 

```bash
git clone https://github.com/Lee-CBG/catELMo.git
cd catELMo/embedders
conda env create -n catELMo -f catELMo.yml

# Note that the requirement for embedding is differnt from the one used for downstream tasks.
# If you have already activated the 'bap' environment, make sure deactivate it before conducting embedding.
# conda deactivate bap

source activate catELMo
```

### 3. Running the scripts

#### Example 1: Obtain embeddings from **catELMo**.
```python
import pandas as pd
from pathlib import Path
import torch
from allennlp.commands.elmo import ElmoEmbedder

model_dir = Path('./path/of/your/downloaded/catELMo')
weights = model_dir/'weights.hdf5'
options = model_dir/'options.json'
embedder  = ElmoEmbedder(options,weights,cuda_device=-1) # cuda_device=-1 for CPU

def catELMo_embedding(x):
    return torch.tensor(embedder.embed_sentence(list(x))).sum(dim=0).mean(dim=0).tolist()

dat = pd.read_csv('./path/of/binding/affinity/prediction/data.csv')
dat['tcr_embeds'] = None
dat['epi_embeds'] = None

dat['epi_embeds'] = dat[['epi']].applymap(lambda x: catELMo_embedding(x))['epi']
dat['tcr_embeds'] = dat[['tcr']].applymap(lambda x: catELMo_embedding(x))['tcr']

dat.to_pickle("./path/of/binding/affinity/prediction/data.pkl")
```

#### Example 2: Obtain embeddings from **BERT-Base-TCR**.
```python
import re
import pandas as pd
from tqdm import tqdm
from transformers import TFBertModel,BertModel, BertForPreTraining, BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained("../../../bert/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("../../../bert/ours_checkpoints/scratch/bin_our_bert_scratch")

def BERT_embedding(x):
    seq = " ".join(x)
    seq = re.sub(r"[UZOB]", "X", seq)
    encoded_input = tokenizer(seq, return_tensors='pt')
    output = model2(**encoded_input)
    return output
    
dat = pd.read_csv('./path/of/binding/affinity/prediction/data.csv')
dat['tcr_embeds'] = None
dat['epi_embeds'] = None

for i in tqdm(range(len(dat))):
    dat.epi_embeds[i] = BERT_embedding(dat.epi[i])[0].reshape(-1,1024).mean(dim=0).tolist()
    dat.tcr_embeds[i] = BERT_embedding(dat.tcr[i])[0].reshape(-1,1024).mean(dim=0).tolist()

dat.to_pickle("./path/of/binding/affinity/prediction/data.pkl")
```

## Citation
If you use catELMo for your research, please cite our papers:
```

```
