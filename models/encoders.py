import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import manifolds
import utils.math_utils as pmath
import torch as th
from utils import *
from utils import pre_utils
from utils.pre_utils import *
from manifolds import *
#from manifolds import LorentzManifold
from layers.CentroidDistance import CentroidDistance


class H2HGCN(nn.Module):

    def __init__(self, args, logger):
        super(H2HGCN, self).__init__()
        self.debug = False
        self.args = args
        self.logger = logger
        self.set_up_params()
        self.activation = nn.SELU()

        self.linear = nn.Linear(
                int(args.feature_dim), int(args.dim),
        )
        nn_init(self.linear, self.args.proj_init)
        self.args.eucl_vars.append(self.linear)	

        if self.args.task == 'nc':
            self.distance = CentroidDistance(args, logger, args.manifold)


    def create_params(self):
        """
        create the GNN params for a specific msg type
        """
        msg_weight = []
        layer = self.args.num_layers if not self.args.tie_weight else 1
        for iii in range(layer):
            M = th.zeros([self.args.dim-1, self.args.dim-1], requires_grad=True)
            init_weight(M, 'orthogonal')
            M = nn.Parameter(M)
            self.args.stie_vars.append(M)
            msg_weight.append(M)
        return nn.ParameterList(msg_weight)

    def set_up_params(self):
        """
        set up the params for all message types
        """
        self.type_of_msg = 1

        for i in range(0, self.type_of_msg):
            setattr(self, "msg_%d_weight" % i, self.create_params())

    def apply_activation(self, node_repr):
        """
        apply non-linearity for different manifolds
        """
        if self.args.select_manifold in {"poincare", "euclidean"}:
            return self.activation(node_repr)
        elif self.args.select_manifold == "lorentz":
            return self.args.manifold.from_poincare_to_lorentz(
                self.activation(self.args.manifold.from_lorentz_to_poincare(node_repr))
            )

    def split_graph_by_negative_edge(self, adj_mat, weight):
        """
        Split the graph according to positive and negative edges.
        """
        mask = weight > 0
        neg_mask = weight < 0

        pos_adj_mat = adj_mat * mask.long()
        neg_adj_mat = adj_mat * neg_mask.long()
        pos_weight = weight * mask.float()
        neg_weight = -weight * neg_mask.float()
        return pos_adj_mat, pos_weight, neg_adj_mat, neg_weight

    def split_graph_by_type(self, adj_mat, weight):
        """
        split the graph according to edge type for multi-relational datasets
        """
        multi_relation_adj_mat = []
        multi_relation_weight = []
        for relation in range(1, self.args.edge_type):
            mask = (weight.int() == relation)
            multi_relation_adj_mat.append(adj_mat * mask.long())
            multi_relation_weight.append(mask.float())
        return multi_relation_adj_mat, multi_relation_weight

    def split_input(self, adj_mat, weight):
        return [adj_mat], [weight]

    def p2k(self, x, c):
        denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
        return 2 * x / denom

    def k2p(self, x, c):
        denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
        return x / denom

    def lorenz_factor(self, x, *, c=1.0, dim=-1, keepdim=False):
        """
            Calculate Lorenz factors
        """
        x_norm = x.pow(2).sum(dim=dim, keepdim=keepdim)
        x_norm = torch.clamp(x_norm, 0, 0.9)
        tmp = 1 / torch.sqrt(1 - c * x_norm)
        return tmp
     
    def from_lorentz_to_poincare(self, x):
        """
        Args:
            u: [batch_size, d + 1]
        """
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    def h2p(self, x):
        return self.from_lorentz_to_poincare(x)

    def from_poincare_to_lorentz(self, x, eps=1e-3):
        """
        Args:
            u: [batch_size, d]
        """
        x_norm_square = x.pow(2).sum(-1, keepdim=True)
        tmp = th.cat((1 + x_norm_square, 2 * x), dim=1)
        tmp = tmp / (1 - x_norm_square)
        return  tmp

    def p2h(self, x):
        return  self.from_poincare_to_lorentz(x)

    def p2k(self, x, c=1.0):
        denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
        return 2 * x / denom

    def k2p(self, x, c=1.0):
        denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
        return x / denom

    def h2k(self, x):
        tmp = x.narrow(-1, 1, x.size(-1)-1) / x.narrow(-1, 0, 1)
        return tmp
        
    def k2h(self, x):
        x_norm_square = x.pow(2).sum(-1, keepdim=True)
        x_norm_square = torch.clamp(x_norm_square, max=0.9)
        tmp = torch.ones((x.size(0),1)).cuda().to(self.args.device)
        tmp1 = th.cat((tmp, x), dim=1)
        tmp2 = 1.0 / torch.sqrt(1.0 - x_norm_square)
        tmp3 = (tmp1 * tmp2)
        return tmp3 


    def hyperbolic_mean(self, y, node_num, max_neighbor, real_node_num, weight, dim=0, c=1.0, ):
        '''
        y [node_num * max_neighbor, dim]
        '''
        x = y[0:real_node_num*max_neighbor, :]
        weight_tmp = weight.view(-1,1)[0:real_node_num*max_neighbor, :]
        x = self.h2k(x)
        
        lamb = self.lorenz_factor(x, c=c, keepdim=True)
        lamb = lamb  * weight_tmp 
        lamb = lamb.view(real_node_num, max_neighbor, -1)

        x = x.view(real_node_num, max_neighbor, -1) 
        k_mean = (torch.sum(lamb * x, dim=1, keepdim=True) / (torch.sum(lamb, dim=1, keepdim=True))).squeeze()
        h_mean = self.k2h(k_mean)

        virtual_mean = torch.cat((torch.tensor([[1.0]]), torch.zeros(1,y.size(-1)-1)), 1).cuda().to(self.args.device)
        tmp = virtual_mean.repeat(node_num-real_node_num, 1)

        mean = torch.cat((h_mean, tmp), 0)
        return mean	

    def test_lor(self, A):
        tmp1 = (A[:,0] * A[:,0]).view(-1)
        tmp2 = A[:,1:]
        tmp2 = th.diag(tmp2.mm(tmp2.transpose(0,1)))
        return (tmp1 - tmp2)

    def retrieve_params(self, weight, step):
        """
        Args:
            weight: a list of weights
            step: a certain layer
        """
        layer_weight = th.cat((th.zeros((self.args.dim-1, 1)).cuda().to(self.args.device), weight[step]), dim=1)
        tmp = th.zeros((1, self.args.dim)).cuda().to(self.args.device)
        tmp[0,0] = 1
        layer_weight = th.cat((tmp, layer_weight), dim=0)
        return layer_weight

    def aggregate_msg(self, node_repr, adj_mat, weight, layer_weight, mask):
        """
        message passing for a specific message type.
        """
        node_num, max_neighbor = adj_mat.shape[0], adj_mat.shape[1] 
        combined_msg = node_repr.clone()

        tmp = self.test_lor(node_repr)
        msg = th.mm(node_repr, layer_weight) * mask
        real_node_num = (mask>0).sum()
        
        # select out the neighbors of each node
        neighbors = th.index_select(msg, 0, adj_mat.view(-1)) 
        combined_msg = self.hyperbolic_mean(neighbors, node_num, max_neighbor, real_node_num, weight)
        return combined_msg 

    def get_combined_msg(self, step, node_repr, adj_mat, weight, mask):
        """
        perform message passing in the tangent space of x'
        """
        gnn_layer = 0 if self.args.tie_weight else step
        combined_msg = None
        for relation in range(0, self.type_of_msg):
            layer_weight = self.retrieve_params(getattr(self, "msg_%d_weight" % relation), gnn_layer)
            aggregated_msg = self.aggregate_msg(node_repr,
                                                adj_mat[relation],
                                                weight[relation],
                                                layer_weight, mask)
            combined_msg = aggregated_msg if combined_msg is None else (combined_msg + aggregated_msg)
        return combined_msg


    def encode(self, node_repr, adj_list, weight):
        node_repr = self.activation(self.linear(node_repr))
        adj_list, weight = self.split_input(adj_list, weight)
        
        mask = torch.ones((node_repr.size(0),1)).cuda().to(self.args.device)
        node_repr = self.args.manifold.exp_map_zero(node_repr)

        for step in range(self.args.num_layers):
            node_repr = node_repr * mask
            tmp = node_repr
            combined_msg = self.get_combined_msg(step, node_repr, adj_list, weight, mask)
            combined_msg = (combined_msg) * mask
            node_repr = combined_msg * mask
            node_repr = self.apply_activation(node_repr) * mask
            real_node_num = (mask>0).sum()
            node_repr = self.args.manifold.normalize(node_repr)
        if self.args.task == 'nc':
            _, node_centroid_sim = self.distance(node_repr, mask) 
            return node_centroid_sim.squeeze()
        return node_repr

class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output
