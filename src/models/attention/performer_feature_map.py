# Adapted from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/feature_maps/fourier_features.py
import math
import torch

from einops import rearrange, repeat

from fast_transformers.feature_maps.base import FeatureMap

from src.models.attention.projection_utils import gaussian_orthogonal_random_matrix
from src.models.attention.performer_utils import softmax_kernel


class PerformerFeatures(FeatureMap):
    """Random Fourier Features for the RBF kernel according to [1].
    [1]: "Weighted Sums of Random Kitchen Sinks: Replacing minimization with
         randomization in learning" by A. Rahimi and Benjamin Recht.
    Arguments
    ---------
        query_dims: int, The input query dimensions in order to sample
                          the noise matrix
        n_features: int, The size of the feature map (should be divisible by 2)
                (default: query_dims)
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dims))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """
    def __init__(self, query_dims, n_features=None, ortho_scaling=0, softmax_temp=None,
                 orthogonal=False, redraw=1, deterministic_eval=False, eps=1e-4,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(query_dims)
        self.n_features = n_features or int(query_dims * math.log(query_dims))
        self.ortho_scaling = ortho_scaling
        # TODO: we're not using @orthogonal atm
        self.orthogonal = orthogonal
        # TODO: we're not using @softmax_temp atm
        self.softmax_temp = 1 / math.sqrt(query_dims) if softmax_temp is None else softmax_temp
        # self.redraw = redraw
        # TODO: not redrawing atm, so I'm setting it to an irrational number
        self.redraw = math.pi
        self.deterministic_eval = deterministic_eval
        self.eps = eps  # Stabilizer for softmax kernel

        # Make a buffer for storing the sampled projection_matrix
        self.register_buffer("projection_matrix", torch.zeros(self.query_dims, self.n_features,
                                                              **factory_kwargs))
        self.factory_kwargs = factory_kwargs
        self._calls = -1

    def new_feature_map(self, device):
        # If we are not training skip the generation of a new feature map
        if self.deterministic_eval and not self.training:
            return

        # Only redraw the new feature map every self.redraw times
        self._calls += 1
        if (self._calls % self.redraw) != 0:
            return

        projection_matrix = gaussian_orthogonal_random_matrix(nrows=self.n_features,
                                                              ncols=self.query_dims,
                                                              scaling=self.ortho_scaling,
                                                              device=device,
                                                              dtype=self.factory_kwargs['dtype'])
        self.register_buffer("projection_matrix", projection_matrix)

    def forward_queries(self, x):
        return softmax_kernel(x, projection_matrix=self.projection_matrix, is_query=True,
                              eps=self.eps)

    def forward_keys(self, x):
        return softmax_kernel(x, projection_matrix=self.projection_matrix, is_query=False,
                              eps=self.eps)
