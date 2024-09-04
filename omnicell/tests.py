import unittest
import torch
import scanpy as sc
import numpy as np

from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY
from omnicell.models.nearest_neighbor.predictor import NearestNeighborPredictor

import logging



class TestAcrossCells(unittest.TestCase):

    def test_mean_transfer(self):

        pd = {
            CELL_KEY: ['A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'],
            PERT_KEY : [CONTROL_PERT, CONTROL_PERT, CONTROL_PERT, CONTROL_PERT, 'pert', 'pert', CONTROL_PERT, 'pert']
        }
        
        #Mean effect of pert on B is (1, 1)
        X = np.array([[3,3], [3, 3], [0, 0],[0, 0], [1, 1], [1, 1], [100, 100], [100, 100]])

        adata = sc.AnnData(X=X, obs=pd)

        predictor = NearestNeighborPredictor(None)
       
        train_adata = adata[(adata.obs[CELL_KEY] == 'B') | (adata.obs[CELL_KEY] == 'C')]
        eval_adata = adata[adata.obs[CELL_KEY] == 'A']

        predictor.train(train_adata)

        pred = predictor.make_predict(eval_adata, 'pert', 'A')

        self.assertTrue(torch.allclose(torch.tensor(pred).type(torch.float32), torch.tensor([[4,4], [4,4]]).type(torch.float32)))


if __name__ == '__main__':
    logging.basicConfig(filename= f'output_TESTS.log', filemode= 'w', level='DEBUG', format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    unittest.main()