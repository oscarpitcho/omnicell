import scanpy as sc
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
from torchclustermetrics import silhouette
import torch
from torch import nn, Tensor
import argparse
import sys
import os
os.chdir(os.getcwd()+'/UCE')
os.environ["OMP_NUM_THREADS"] = "12"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "12"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "12"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "12"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "12"
from accelerate import Accelerator
from tqdm.auto import tqdm
from model import TransformerModel
from utils import figshare_download
from torch.utils.data import DataLoader
from data_proc.data_utils import adata_path_to_prot_chrom_starts, \
    get_spec_chrom_csv, process_raw_anndata, get_species_to_pe
import pickle
import pandas as pd
import torch.utils.data as data
from typing import Dict, List, Optional, Tuple, Any
warnings.filterwarnings("ignore")



# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Analysis settings.')
parser.add_argument('--dataset', type=str, default='Kang_et_al', help='Dataset name.')
parser.add_argument('--species', type=str, default='human', help='Species argument for UCE')

device = torch.device("cpu")

# Suppress warnings
warnings.filterwarnings('ignore')

# Parse arguments from command line
args = parser.parse_args()

# Assign parsed arguments to variables
dataset_name = args.dataset
species = args.species


# Create a timestamp for output files
timestamp = datetime.today().strftime('%Y%m%d%H%M%S')

# Define path to dataset
data_folder_path = f'../Datasets/{dataset_name}/'
datafilename = [f for f in os.listdir(data_folder_path+'Data/') if f[-1]=='d'][0]

filnametrunc = datafilename[:-5]



# Prepare output directory
output_path = Path(f'{data_folder_path}UCE/{timestamp}')
output_path.mkdir(parents=True, exist_ok=True)



adata_path = f'{data_folder_path}Data/{datafilename}'
class AnndataProcessor:
    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.h5_folder_path = './'
        self.npz_folder_path = './'
        self.scp = ""

        # Check if paths exist, if not, create them
        self.check_paths()

        # Set up the anndata
        self.adata_name = datafilename
        self.adata_root_path = f'{data_folder_path}Data/'
        self.name = self.adata_name[:-5]
        self.proc_h5_path = self.h5_folder_path + f"{self.name}_proc.h5ad"
        self.adata = None

        # Set up the row
        row = pd.Series()
        row.path = self.adata_name
        row.covar_col = np.nan
        row.species = species
        self.row = row

        # Set paths once to be used throughout the class
        self.pe_idx_path = f"./{self.name}_pe_idx.torch"
        self.chroms_path = f"./{self.name}_chroms.pkl"
        self.starts_path = f"./{self.name}_starts.pkl"
        self.shapes_dict_path = f"./{self.name}_shapes_dict.pkl"

    def check_paths(self):
        """
        Check if the paths exist, if not, create them
        """
        figshare_download("https://figshare.com/ndownloader/files/42706558",
                                './model_files/species_chrom.csv')
        figshare_download("https://figshare.com/ndownloader/files/42706555",
                                './model_files/species_offsets.pkl')
        if not os.path.exists('./model_files/protein_embeddings/'):
            figshare_download("https://figshare.com/ndownloader/files/42715213",
                'model_files/protein_embeddings.tar.gz')
        figshare_download("https://figshare.com/ndownloader/files/42706585",
                                './model_files/all_tokens.torch')
 

    def preprocess_anndata(self):
        if self.accelerator.is_main_process:
            self.adata, num_cells, num_genes = \
                process_raw_anndata(self.row,
                                    self.h5_folder_path,
                                    self.npz_folder_path,
                                    self.scp,
                                    True,
                                    True,
                                    root=self.adata_root_path)

            if (num_cells is not None) and (num_genes is not None):
                self.save_shapes_dict(self.name, num_cells, num_genes,
                                       self.shapes_dict_path)

            if self.adata is None:
                self.adata = sc.read(self.proc_h5_path)

    def save_shapes_dict(self, name, num_cells, num_genes, shapes_dict_path):
        shapes_dict = {name: (num_cells, num_genes)}
        with open(shapes_dict_path, "wb+") as f:
            pickle.dump(shapes_dict, f)
            print("Wrote Shapes Dict")

    def generate_idxs(self):
        if self.accelerator.is_main_process:
            if os.path.exists(self.pe_idx_path) and \
                    os.path.exists(self.chroms_path) and \
                    os.path.exists(self.starts_path):
                print("PE Idx, Chrom and Starts files already created")

            else:
                species_to_pe = get_species_to_pe('./model_files/protein_embeddings/')
                with open('./model_files/species_offsets.pkl', "rb") as f:
                    species_to_offsets = pickle.load(f)

                gene_to_chrom_pos = get_spec_chrom_csv(
                    './model_files/species_chrom.csv')
                dataset_species = species
                spec_pe_genes = list(species_to_pe[dataset_species].keys())
                offset = species_to_offsets[dataset_species]
                pe_row_idxs, dataset_chroms, dataset_pos = adata_path_to_prot_chrom_starts(
                    self.adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset)

                # Save to the temp dict
                torch.save({self.name: pe_row_idxs}, self.pe_idx_path)
                with open(self.chroms_path, "wb+") as f:
                    pickle.dump({self.name: dataset_chroms}, f)
                with open(self.starts_path, "wb+") as f:
                    pickle.dump({self.name: dataset_pos}, f)

    def run_evaluation(self):
        self.accelerator.wait_for_everyone()
        with open(self.shapes_dict_path, "rb") as f:
            shapes_dict = pickle.load(f)
        adata = run_eval(self.adata, self.name, self.pe_idx_path, self.chroms_path,
                 self.starts_path, shapes_dict, self.accelerator)
        return adata


def get_ESM2_embeddings():
    # Load in ESM2 embeddings and special tokens
    all_pe = torch.load('./model_files/all_tokens.torch')
    if all_pe.shape[0] == 143574:
        torch.manual_seed(23)
        CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, 5120))
        # 1895 is the total number of chromosome choices, it is hardcoded for now
        all_pe = torch.vstack(
            (all_pe, CHROM_TENSORS))  # Add the chrom tensors to the end
        all_pe.requires_grad = False

    return all_pe


def padding_tensor(sequences):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len, 1280)

    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    out_dims2 = (num, max_len)

    mask = sequences[0].data.new(*out_dims2).fill_(float('-inf'))
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor.permute(1, 0, 2), mask


def run_eval(adata, name, pe_idx_path, chroms_path, starts_path, shapes_dict,
             accelerator):

    #### Set up the model ####
    token_dim = 5120
    emsize = 1280  # embedding dimension
    d_hid = 5120  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 33  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 20  # number of heads in nn.MultiheadAttention
    dropout = 0.05  # dropout probability
    model = TransformerModel(token_dim=token_dim, d_model=emsize, nhead=nhead,
                             d_hid=d_hid,
                             nlayers=nlayers, dropout=dropout,
                             output_dim=1280)

    # intialize as empty
    empty_pe = torch.zeros(145469, 5120)
    empty_pe.requires_grad = False
    model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
    model.load_state_dict(torch.load('24320806/33l_8ep_1024t_1280.torch', map_location="cpu"),
                          strict=True)
    # Load in the real token embeddings
    all_pe = get_ESM2_embeddings()
    # This will make sure that you don't overwrite the tokens in case you're embedding species from the training data
    # We avoid doing that just in case the random seeds are different across different versions. 
    if all_pe.shape[0] != 145469: 
        all_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
    model = model.eval()
    model = accelerator.prepare(model)
    batch_size = 12
    #### Run the model ####
    # Dataloaders
    dataset = MultiDatasetSentences(sorted_dataset_names=[name],
                                    shapes_dict=shapes_dict,
                                    npzs_dir='./',
                                    dataset_to_protein_embeddings_path=pe_idx_path,
                                    datasets_to_chroms_path=chroms_path,
                                    datasets_to_starts_path=starts_path
                                    )
    multi_dataset_sentence_collator = MultiDatasetSentenceCollator()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=multi_dataset_sentence_collator,
                            num_workers=0)
    dataloader = accelerator.prepare(dataloader)
    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    dataset_embeds = []
    with torch.no_grad():
        for batch in pbar:
            batch_sentences, mask, idxs = batch[0], batch[1], batch[2]
            batch_sentences = batch_sentences.permute(1, 0)
            batch_sentences = model.pe_embedding(batch_sentences.long())
            batch_sentences = nn.functional.normalize(batch_sentences,
                                                      dim=2)  # Normalize token outputs now
            _, embedding = model.forward(batch_sentences, mask=mask)
            # Fix for duplicates in last batch
            accelerator.wait_for_everyone()
            embeddings = accelerator.gather_for_metrics((embedding))
            if accelerator.is_main_process:
                dataset_embeds.append(embeddings.detach().cpu().numpy())

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        dataset_embeds = np.vstack(dataset_embeds)
        adata.obsm["X_uce"] = dataset_embeds
        return adata


class MultiDatasetSentences(data.Dataset):
    def __init__(self, sorted_dataset_names, shapes_dict, 
                 dataset_to_protein_embeddings_path= "/lfs/local/0/yanay/reduced_datasets_to_pe_chrom_5120_new.torch",
                 datasets_to_chroms_path="/lfs/local/0/yanay/dataset_to_chroms_new.pkl",
                 datasets_to_starts_path="/lfs/local/0/yanay/dataset_to_starts_new.pkl",
                 npzs_dir="/lfs/local/0/yanay/uce_proc/") -> None:
        super(MultiDatasetSentences, self).__init__()
        # self.xs = {}
        self.num_cells = {}
        self.num_genes = {}
        self.shapes_dict = shapes_dict

        self.total_num_cells = 0
        for name in sorted_dataset_names:
            num_cells, num_genes = self.shapes_dict[name]
            # self.xs[name] = X
            self.num_cells[name] = num_cells
            self.num_genes[name] = num_genes

            self.total_num_cells += num_cells

        self.datasets = sorted_dataset_names

        # TODO: preferably not hard-coded here
        self.dataset_to_protein_embeddings = torch.load(dataset_to_protein_embeddings_path)
        with open(datasets_to_chroms_path, "rb") as f:
            self.dataset_to_chroms = pickle.load(f)
        with open(datasets_to_starts_path, "rb") as f:
            self.dataset_to_starts = pickle.load(f)
        
        self.npzs_dir = npzs_dir

    def __getitem__(self, idx):
        if isinstance(idx, int):
            for dataset in sorted(self.datasets):
                if idx < self.num_cells[dataset]:
                    #cts = np.memmap(f"/lfs/local/0/yanay/cxg_npzs/" + f"{dataset}_counts.npz",
                    #        dtype='int64', mode='r', shape=self.shapes_dict[dataset])
                    cts = np.memmap(self.npzs_dir + f"{dataset}_counts.npz", dtype='int64', mode='r', shape=self.shapes_dict[dataset])
                    counts = cts[idx]
                    counts = torch.tensor(counts).unsqueeze(0)
                    weights = torch.log1p(counts)
                    weights = (weights / torch.sum(weights))
                    batch_sentences, mask, seq_len, cell_sentences = \
                        sample_cell_sentences(counts, weights, dataset,
                            dataset_to_protein_embeddings= self.dataset_to_protein_embeddings,
                            dataset_to_chroms=self.dataset_to_chroms,
                            dataset_to_starts=self.dataset_to_starts)
                    return batch_sentences, mask, idx, seq_len, cell_sentences
                else:
                    idx -= self.num_cells[dataset]
            raise IndexError
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


class MultiDatasetSentenceCollator(object):
    def __init__(self):
        self.pad_length = 1536

    def __call__(self, batch):
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length))
        mask = torch.zeros((batch_size, self.pad_length))
        cell_sentences = torch.zeros((batch_size, self.pad_length))

        idxs = torch.zeros(batch_size)

        i = 0
        max_len = 0
        for bs, msk, idx, seq_len, cs in batch:
            batch_sentences[i, :] = bs
            cell_sentences[i, :] = cs
            max_len = max(max_len, seq_len)
            mask[i, :] = msk
            idxs[i] = idx

            i += 1

        return batch_sentences[:, :max_len] , mask[:, :max_len], idxs, cell_sentences

def sample_cell_sentences(counts, batch_weights, dataset,
                          dataset_to_protein_embeddings,
                          dataset_to_chroms,
                          dataset_to_starts):

    dataset_idxs = dataset_to_protein_embeddings[dataset] # get the dataset specific protein embedding idxs
    cell_sentences = torch.zeros((counts.shape[0], 1536)) # init the cell representation as 0s
    mask = torch.zeros((counts.shape[0], 1536)) # start of masking the whole sequence
    chroms = dataset_to_chroms[dataset] # get the dataset specific chroms for each gene
    starts = dataset_to_starts[dataset] # get the dataset specific genomic start locations for each gene

    longest_seq_len = 0 # we need to keep track of this so we can subset the batch at the end

    for c, cell in enumerate(counts):
        weights = batch_weights[c].numpy()
        weights = weights / sum(weights)  # RE NORM after mask
        
        # randomly choose the genes that will make up the sample, weighted by expression, with replacement
        choice_idx = np.random.choice(np.arange(len(weights)),
                                      size=1024, p=weights,
                                      replace=True)
        choosen_chrom = chroms[choice_idx] # get the sampled genes chromosomes
        # order the genes by chromosome
        chrom_sort = np.argsort(choosen_chrom)  
        choice_idx = choice_idx[chrom_sort]

        # sort the genes by start
        new_chrom = chroms[choice_idx]
        choosen_starts = starts[choice_idx]

        ordered_choice_idx = np.full((1536),
                                     3)  # start with cls
        # i= 0 first token is CLS
        i = 1  # continue on to the rest of the sequence with left bracket being assumed.
        # Shuffle the chroms now, there's no natural order to chromosomes
        uq_chroms = np.unique(new_chrom)
        np.random.shuffle(uq_chroms) # shuffle
        
        # This loop is actually just over one cell
        for chrom in uq_chroms:
            # Open Chrom token
            ordered_choice_idx[i] = int(chrom) + 143574 # token of this chromosome # i = 1 next token is a chrom open
            i += 1
            # now sort the genes by start order within the chroms
            loc = np.where(new_chrom == chrom)[0]
            sort_by_start = np.argsort(
                choosen_starts[loc])  # start locations for this chromsome

            to_add = choice_idx[loc[sort_by_start]]
            ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]
            i += len(to_add)
            ordered_choice_idx[i] = 2 # add the chrom sep again
            i += 1  # add the closing token again

        longest_seq_len = max(longest_seq_len, i)
        remainder_len = (1536 - i)

        cell_mask = torch.concat((torch.ones(i),
                                  # pay attention to all of these tokens, ignore the rest!
                                  torch.zeros(remainder_len)))

        mask[c, :] = cell_mask

        ordered_choice_idx[i:] = 0 # the remainder of the sequence
        cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)
        
    cell_sentences_pe = cell_sentences.long() # token indices
    
    return cell_sentences_pe, mask, longest_seq_len, cell_sentences


checkembed = [f for f in os.listdir(f'.') if f==filnametrunc+'_UCEEmbed.h5ad']
if len(checkembed)==0:
    accelerator = Accelerator(project_dir='./')
    processor = AnndataProcessor(accelerator)
    processor.preprocess_anndata()
    processor.generate_idxs()
    adata = processor.run_evaluation()
    adata.write_h5ad(f'{filnametrunc}_UCEEmbed.h5ad')
else:
    adata = sc.read(f'{filnametrunc}_UCEEmbed.h5ad')




if dataset_name == 'Kang_et_al':
    adata.obs['perturbed'] = adata.obs['condition'] == 'stimulated'
    pathways = ['All']
    pathway_unperturbed = 'None'
    pathways = [p for p in pathways if p != pathway_unperturbed]
    adata.obs['pathway'] = 'All'
    
if dataset_name == 'Nault_single':
    adata.obs['perturbed'] = adata.obs.Dose >= 30
    pathways = ['All']
    pathway_unperturbed = 'None'
    pathways = [p for p in pathways if p != pathway_unperturbed]
    adata.obs['pathway'] = 'All'
    adata.var.index = [f.upper() for f in adata.var.index]
    adata.obs['cell_type'] = adata.obs['celltype']
    adata.write_h5ad('UCE/temp.h5ad')
    
if dataset_name == 'srivatsan':
    
    adata.obs['perturbed'] = adata.obs.dose >= 10000
    adata.obs['unperturbed'] = adata.obs.dose == 0
    adata = adata[adata.obs['perturbed'] | adata.obs['unperturbed']]
    adata.obs['pathway'] = adata.obs['pathway_level_1']
    pathways = np.unique(adata.obs['pathway_level_1'].to_numpy())
    pathway_unperturbed = 'Vehicle'
    pathways = [p for p in pathways if p != pathway_unperturbed]
    
adata.X.data = adata.X.data.astype(np.float32)

adata.obs['perturbed'] = adata.obs['perturbed'].astype('category')


cell_types = list(adata.obs['cell_type'].cat.categories)
allscores = []
for ct in cell_types:
    for pt in pathways:
        curdata = adata[adata.obs['cell_type'] == ct].copy()
        curdata = curdata[(curdata.obs['pathway']==pt) | (curdata.obs['pathway']==pathway_unperturbed)]
        sc.pp.neighbors(curdata, n_neighbors=15, use_rep='X_uce')
        curdata.obsm["X_pca"] = sc.tl.pca(curdata.obsm["X_uce"])
        sc.tl.umap(curdata, min_dist=0.1, random_state=42)        
        with plt.rc_context():  
            sc.pl.umap(curdata, color=["perturbed", "cell_type"], frameon=False, show=False)
            plt.savefig(str(output_path) + '/'+str(ct)+'_umap.png', bbox_inches="tight")
            sc.pl.pca(curdata, color=["perturbed", "cell_type"], frameon=False, show=False)
            plt.savefig(str(output_path) + '/'+str(ct)+'_pca.png', bbox_inches="tight")
        curdataX_torch = torch.from_numpy(curdata.obsm['X_umap'].astype(np.float32))
        curdataX_torch = curdataX_torch.to(device)
        condition_torch = torch.from_numpy(curdata.obs["perturbed"].to_numpy().astype(np.float32))
        condition_torch = condition_torch.to(device)
        asw = silhouette.score(curdataX_torch, condition_torch)
        ctl = torch.sum(condition_torch==0).item()
        pert = torch.sum(condition_torch==1).item()
        print(f'Silhouette Score is {asw}, ctl={ctl}, pert={pert} for {ct}_{pt}')
        with open(str(output_path) + '/asw.txt', "a") as f:
            f.write(f"{ct}_{pt} = {asw}\n")
            


