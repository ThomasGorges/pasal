import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from rdkit import Chem
from tqdm import tqdm


model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

fusion_df = pd.read_json('../../output/preprocessing/fusion.json', orient='index')
fusion_df.index.names = ['CID']

smiles =  fusion_df["IsomericSMILES"].values.tolist()

# Remove isomeric information since MolFormer was trained without this information
smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False) for s in smiles]

tokenized_input = tokenizer(smiles, padding=True, return_tensors="pt")

batch_size = 128
embeddings = []
for start_idx in tqdm(range(0, len(smiles), batch_size)):
    end_idx = start_idx + batch_size
    batch = smiles[start_idx : end_idx]

    inputs = {
        "input_ids": tokenized_input["input_ids"][start_idx : end_idx],
        "attention_mask": tokenized_input["attention_mask"][start_idx : end_idx]
    }

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings.append(outputs["last_hidden_state"])

embeddings = torch.cat(embeddings, dim=0)
embeddings = embeddings.cpu().detach().numpy()
embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2])
df = pd.DataFrame(embeddings, index=fusion_df.index)
df.to_csv("../../output/preprocessing/molformer_features.csv")
