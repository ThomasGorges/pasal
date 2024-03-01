import pandas as pd
import numpy as np

# Export testset CIDs
testset_df = pd.read_csv('../../../data/dream/GSs2_new.txt', delimiter='\t', index_col='oID')

testset_cids = np.unique(testset_df.index.values)
testset_cid_df = pd.DataFrame(data={'CID': testset_cids})

testset_cid_df.to_csv('../../../data/dream/GSs2_cids.csv')


# Export validation CIDs
valset_df = pd.read_csv('../../../data/dream/LBs2.txt', delimiter='\t', index_col='#oID')

valset_cids = np.unique(valset_df.index.values)
valset_cid_df = pd.DataFrame(data={'CID': valset_cids})

valset_cid_df.to_csv('../../../data/dream/LBs2_cids.csv')
