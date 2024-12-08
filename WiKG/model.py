import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention,GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention,GATv2Conv
from torchvision import transforms
import timm
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
)
        model.load_state_dict(torch.load(("./pretrain/pytorch_model.bin")), strict=True)
        # model.load_state_dict(torch.load(("D:/Downloads/pytorch_model.bin"), map_location="cuda:0"), strict=True)
        self.model=model
        # self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x=x.unsqueeze(0)
        # print(x.shape)
        x = self.model(x)
        
        return x




# class GAT(torch.nn.Module):
#     """
#     3 GAT implementations
#     """

#     def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
#                  dropout=0.6, layer_type=LayerType.IMP3, log_attention_weights=False):
#         super().__init__()
#         assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

#         GATLayer = GATLayerImp3  # fetch one of 3 available implementations
#         num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that we can nicely create GAT layers below

#         gat_layers = []  # collect GAT layers
#         for i in range(num_of_layers):
#             layer = GATLayer(
#                 num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
#                 num_out_features=num_features_per_layer[i+1],
#                 num_of_heads=num_heads_per_layer[i+1],
#                 concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
#                 activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
#                 dropout_prob=dropout,
#                 add_skip_connection=add_skip_connection,
#                 bias=bias,
#                 log_attention_weights=log_attention_weights
#             )
#             gat_layers.append(layer)

#         self.gat_net = nn.Sequential(
#             *gat_layers,
#         )

#     def forward(self, data):
#         return self.gat_net(data)


# class GATLayer(torch.nn.Module):
#     """
#     Base class for all implementations as there is much code that would otherwise be copy/pasted.
#     """

#     head_dim = 1

#     def __init__(self, num_in_features, num_out_features, num_of_heads, layer_type, concat=True, activation=nn.ELU(),
#                  dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

#         super().__init__()

#         self.num_of_heads = num_of_heads
#         self.num_out_features = num_out_features
#         self.concat = concat  # whether we should concatenate or average the attention heads
#         self.add_skip_connection = add_skip_connection

#         #
#         # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
#         # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
#         #

#         if layer_type == LayerType.IMP1:
#             self.proj_param = nn.Parameter(torch.Tensor(num_of_heads, num_in_features, num_out_features))
#         else:
#             # You can treat this one matrix as num_of_heads independent W matrices
#             self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

#         # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
#         # which gives us unnormalized score "e". Here we split the "a" vector - but the semantics remain the same.

#         # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
#         # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
#         self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
#         self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

#         if layer_type == LayerType.IMP1:  # simple reshape in the case of implementation 1
#             self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_of_heads, num_out_features, 1))
#             self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_of_heads, num_out_features, 1))

#         # Bias is definitely not crucial to GAT - feel free to experiment
#         if bias and concat:
#             self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
#         elif bias and not concat:
#             self.bias = nn.Parameter(torch.Tensor(num_out_features))
#         else:
#             self.register_parameter('bias', None)

#         if add_skip_connection:
#             self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
#         else:
#             self.register_parameter('skip_proj', None)

#         #
#         # End of trainable weights
#         #

#         self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper
#         self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
#         self.activation = activation
#         self.dropout = nn.Dropout(p=dropout_prob)

#         self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
#         self.attention_weights = None  # for later visualization purposes, I cache the weights here

#         self.init_params(layer_type)

#     def init_params(self, layer_type):
#         nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight)
#         nn.init.xavier_uniform_(self.scoring_fn_target)
#         nn.init.xavier_uniform_(self.scoring_fn_source)

#         if self.bias is not None:
#             torch.nn.init.zeros_(self.bias)

#     def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
#         if self.log_attention_weights:  # potentially log for later visualization in playground.py
#             self.attention_weights = attention_coefficients

#         # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
#         # only imp1 will enter this one
#         if not out_nodes_features.is_contiguous():
#             out_nodes_features = out_nodes_features.contiguous()

#         if self.add_skip_connection:  # add skip or residual connection
#             if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
#                 # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
#                 # thus we're basically copying input vectors NH times and adding to processed vectors
#                 out_nodes_features += in_nodes_features.unsqueeze(1)
#             else:
#                 # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
#                 # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
#                 out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

#         if self.concat:
#             # shape = (N, NH, FOUT) -> (N, NH*FOUT)
#             out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
#         else:
#             # shape = (N, NH, FOUT) -> (N, FOUT)
#             out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

#         if self.bias is not None:
#             out_nodes_features += self.bias

#         return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


# class GATLayerImp3(GATLayer):
#     """
#     Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

#     It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
#     into a single graph with multiple components and this layer is agnostic to that fact! <3
#     """

#     src_nodes_dim = 0  # position of source nodes in edge index
#     trg_nodes_dim = 1  # position of target nodes in edge index

#     nodes_dim = 0      # node dimension/axis
#     head_dim = 1       # attention head dimension/axis

#     def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
#                  dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

#         # Delegate initialization to the base class
#         super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP3, concat, activation, dropout_prob,
#                       add_skip_connection, bias, log_attention_weights)

#     def forward(self, data):
#         #
#         # Step 1: Linear Projection + regularization
#         #

#         in_nodes_features, edge_index = data  # unpack data
#         num_of_nodes = in_nodes_features.shape[self.nodes_dim]
#         assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

#         # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
#         # We apply the dropout to all of the input node features (as mentioned in the paper)
#         # Note: for Cora features are already super sparse so it's questionable how much this actually helps
#         in_nodes_features = self.dropout(in_nodes_features)

#         # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
#         # We project the input node features into NH independent output features (one for each attention head)
#         nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

#         nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

#         #
#         # Step 2: Edge attention calculation
#         #

#         # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
#         # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
#         # Optimization note: torch.sum() is as performant as .sum() in my experiments
#         scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
#         scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

#         # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
#         # the possible combinations of scores we just prepare those that will actually be used and those are defined
#         # by the edge index.
#         # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
#         scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
#         scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

#         # shape = (E, NH, 1)
#         attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
#         # Add stochasticity to neighborhood aggregation
#         attentions_per_edge = self.dropout(attentions_per_edge)

#         #
#         # Step 3: Neighborhood aggregation
#         #

#         # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
#         # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
#         nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

#         # This part sums up weighted and projected neighborhood feature vectors for every target node
#         # shape = (N, NH, FOUT)
#         out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

#         #
#         # Step 4: Residual/skip connections, concat and bias
#         #

#         out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
#         return (out_nodes_features, edge_index)

#     #
#     # Helper functions (without comments there is very little code so don't be scared!)
#     #

#     def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
#         """
#         As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
#         Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
#         into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
#         in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
#         (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
#          i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

#         Note:
#         Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
#         and it's a fairly common "trick" used in pretty much every deep learning framework.
#         Check out this link for more details:

#         https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
#         """
        
#         # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
#         scores_per_edge = scores_per_edge - scores_per_edge.max()
#         exp_scores_per_edge = scores_per_edge.exp()  # softmax

#         # Calculate the denominator. shape = (E, NH)
#         neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

#         # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
#         # possibility of the computer rounding a very small number all the way to 0.
#         attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

#         # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
#         return attentions_per_edge.unsqueeze(-1)

#     def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
#         # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
#         trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

#         # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
#         size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
#         size[self.nodes_dim] = num_of_nodes
#         neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

#         # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
#         # target index)
#         neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

#         # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
#         # all the locations where the source nodes pointed to i (as dictated by the target index)
#         # shape = (N, NH) -> (E, NH)
#         return neighborhood_sums.index_select(self.nodes_dim, trg_index)

#     def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
#         size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
#         size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
#         out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

#         # shape = (E) -> (E, NH, FOUT)
#         trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
#         # aggregation step - we accumulate projected, weighted node features for all the attention heads
#         # shape = (E, NH, FOUT) -> (N, NH, FOUT)
#         out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

#         return out_nodes_features

#     def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
#         """
#         Lifts i.e. duplicates certain vectors depending on the edge index.
#         One of the tensor dims goes from N -> E (that's where the "lift" comes from).

#         """
#         src_nodes_index = edge_index[self.src_nodes_dim]
#         trg_nodes_index = edge_index[self.trg_nodes_dim]

#         # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
#         scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
#         scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
#         nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

#         return scores_source, scores_target, nodes_features_matrix_proj_lifted

#     def explicit_broadcast(self, this, other):
#         # Append singleton dimensions until this.dim() == other.dim()
#         for _ in range(this.dim(), other.dim()):
#             this = this.unsqueeze(-1)

#         # Explicitly expand so that shapes are the same
#         return this.expand_as(other)





    
def build_edge_index(topk_index):
    """
    Build edge_index for the graph based on topk neighbors for each node (image).
    
    Args:
    - topk_index (Tensor): Top-k indices of neighbors for each image (batch_size, num_nodes, k)
    
    Returns:
    - edge_index (Tensor): A 2D tensor representing the graph edges (2, num_edges)
    """
    # Number of nodes (images)
    # print(topk_index)
    num_nodes = topk_index.size(1)  # Number of nodes (images)
    k = topk_index.size(2)          # Number of neighbors per image

    # Create source nodes (repeat node indices for each of its top-k neighbors)
    # src corresponds to the "from" node (each node connects to its neighbors)
    src = torch.arange(num_nodes, device=topk_index.device).repeat_interleave(k).view(-1)

    # Flatten topk_index to obtain target nodes (neighbors)
    # dst corresponds to the "to" node (each node connects to its neighbors)
    dst = topk_index.view(-1)

    # Stack the source and destination nodes to form edge_index (2, num_edges)
    edge_index = torch.stack([src, dst], dim=0)
    
    return edge_index




    
def build_edge_index(topk_index):
    """
    Build edge_index for the graph based on topk neighbors for each node (image).
    
    Args:
    - topk_index (Tensor): Top-k indices of neighbors for each image (batch_size, num_nodes, k)
    
    Returns:
    - edge_index (Tensor): A 2D tensor representing the graph edges (2, num_edges)
    """
    # Number of nodes (images)
    # print(topk_index)
    num_nodes = topk_index.size(1)  # Number of nodes (images)
    k = topk_index.size(2)          # Number of neighbors per image

    # Create source nodes (repeat node indices for each of its top-k neighbors)
    # src corresponds to the "from" node (each node connects to its neighbors)
    src = torch.arange(num_nodes, device=topk_index.device).repeat_interleave(k).view(-1)

    # Flatten topk_index to obtain target nodes (neighbors)
    # dst corresponds to the "to" node (each node connects to its neighbors)
    dst = topk_index.view(-1)

    # Stack the source and destination nodes to form edge_index (2, num_edges)
    edge_index = torch.stack([src, dst], dim=0)
    
    return edge_index
import numpy as np

def create_edge_index(pos, k=6):
    """
    Create edge_index for a graph based on the k-nearest neighbors.

    Parameters:
    pos (torch.Tensor): Tensor of shape (n, 2) representing positions of nodes.
    k (int): Number of nearest neighbors to connect to each node.

    Returns:
    edge_index (torch.Tensor): Tensor of shape (2, num_edges) representing edges.
    """
    # Convert positions to NumPy for efficient computation
    pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
    
    # Compute pairwise distances
    n = pos_np.shape[0]
    k = min(k,n)
    distances = np.linalg.norm(pos_np[:, None, :] - pos_np[None, :, :], axis=-1)
    
    # Find k-nearest neighbors for each node (excluding itself)
    neighbors = np.argsort(distances, axis=1)[:, 1:k+1]
    
    # Create edges
    source = np.repeat(np.arange(n), k)
    target = neighbors.flatten()
    # print(source.shape,target.shape,n)
    # Convert to PyTorch tensor
    edge_index = torch.tensor(np.array([source, target]), dtype=torch.long)
    
    return edge_index
class WiKG(nn.Module):
    def __init__(self, dim_in=1024, dim_hidden=512, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3, pool='attn',deivce='cuda:0'):
        super().__init__()
        # self.image_encoder = ImageEncoder()
        # self.image_encoder = ImageEncoder()
        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.LeakyReLU())
        self.device= deivce
        # self.W_head = nn.Linear(dim_hidden, dim_hidden)
        # self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.agg_type = agg_type

        # self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        # self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        # self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)

        # if self.agg_type == 'gcn':
        #     self.linear = nn.Linear(dim_hidden, dim_hidden)
        # elif self.agg_type == 'sage':
        #     self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        # elif self.agg_type == 'bi-interaction':
        #     self.linear1 = nn.Linear(dim_hidden, dim_hidden)
        #     self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        # else:
        #     raise NotImplementedError
        
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim_hidden)
        self.fc = nn.Linear(dim_hidden, n_classes)
 

        self.gat = GATv2Conv(dim_hidden, dim_hidden, heads=3, concat=False)
        self.gene_head = nn.Sequential(
            nn.Linear(dim_hidden, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, n_classes)
        )


        if pool == "mean":
            self.readout = global_mean_pool 
        elif pool == "max":
            self.readout = global_max_pool 
        elif pool == "attn":
            att_net=nn.Sequential(nn.Linear(dim_hidden, dim_hidden // 2), nn.LeakyReLU(), nn.Linear(dim_hidden//2, 1))     
            self.readout = GlobalAttention(att_net)


    
    
    def forward(self, X):
        # try:
        #     x = x["feature"]
        # except:
        #     x = x.squeeze(1)

        # x = self.image_encoder(x)
        #-------------
        x=X['feature'].to(self.device)
        pos=X['position'].to(self.device)
        
        # x = self._fc1(x).unsqueeze(0)    # [B, N, C]
        # x = self._fc1(x) #for testing python model.py
        #--------------
        edge_index= create_edge_index(pos,k=6)

        
        
        # Construct neighbor relationships
        # print(edge_index)
        # print(f"e_t: {e_t.shape}, topk_index: {topk_index.shape}")
        
        

        
        # print(h.shape)
        # Remove pooling; process each node separately
        #----------------
        # print(x.shape)
        x=x.squeeze(0)
        edge_index = edge_index.to(x.device)

        x = self.gat(x, edge_index)
        # h = self.fc(h).squeeze(0) 
        
        #-----------------
        x = self.fc(x).squeeze(0)  # Classify each node
        return x

            
if __name__ == "__main__":
    # data = torch.randn(( 1,1000, 256)).cuda()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # data = torch.randn((18, 3, 224, 224)).to(device)
    data = torch.randn(( 1,9, 1024)).to(device)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # data = torch.randn((18, 3, 224, 224)).to(device)
    data = torch.randn(( 1,9, 1024)).to(device)

    model = WiKG(dim_in=1024, dim_hidden=512, topk=6, n_classes=460, agg_type='bi-interaction', dropout=0.3, pool='attn').to(device)

    output = model(data)
    # print(output.shape,output)
    # print(output.shape,output)

# e_t: torch.Size([1, 1000, 512]), topk_index: torch.Size([1, 1000, 6]), Nb_h: torch.Size([1, 1000, 6, 512])
# e_t: torch.Size([7, 512]), topk_index: torch.Size([7, 6])
