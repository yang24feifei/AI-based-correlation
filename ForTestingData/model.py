import math

import torch
# import torch.nn.functional as F
from torch import nn

from log_attention import LogAttention

        
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000, dropout=0.1):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)  # Shape: (max_len, 1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

#         pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
#         pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

#         # pe=pe.to(device)
#         self.register_buffer('pe', pe)  # Saves as a non-trainable buffer

#     def forward(self, x):
        
#         result = torch.zeros_like(x)        
#         # Add tensor1 to each slice of tensor2
#         for i in range(0, x.size(0), 10):
#             if i + 10 <= x.size(0):
#                 result[i:i+10, :] = x[i:i+10, :] + self.pe  # Regular 10-sized slices
#             else:
#                 result[i:, :] = x[i:, :] + self.pe[:x.size(0) - i, :]  # Last slice with smaller size
#         # x=x+self.pe
#         # print(result.size())  # (64,768)
        
#         return self.dropout(result)
               

class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first']  

    def __init__(self, d_model, nhead,  dim_feedforward=3072, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first= True, # # False, #  chatgpt
                 param_vocab_size=100, param_embed_dim=32,
                 device=None, dtype=None) -> None:
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                        **factory_kwargs)
        self.self_attn = LogAttention(d_model, nhead, param_vocab_size, param_embed_dim, batch_first=batch_first) #, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src,is_causal=False, mask=None, src_key_padding_mask=None,   **kwargs):
        r"""Pass the input through the encoder layer.

        Args:      # for nn.MultiheadAttention()
            src: the sequence to the encoder layer (required).

        Shape:
            see the docs in Transformer class.
        """
        # for nn.MultiheadAttention()
        # src2 = self.self_attn(src, src, src,src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)[0]

        # for LogAttention
        src2=self.self_attn(src[:,:-1], src[:,-1].long()) #, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        src2 = self.dropout1(src2)
        src = self.norm1(src[:,:-1] + src2)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        return src2 # src



class Model(nn.Module):
    def __init__(self, mode, num_layers=4, dim=768, window_size=10, nhead=8, dim_feedforward=3072,
                 dim_ff_reduced=128, dropout=0.1, eventID_size=0, eventID_embed_dim=128, param_vocab_size=100, param_embed_dim=3):
        super(Model, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.mode= mode
        self.event_size=eventID_size
        self.param_vocab_size=param_vocab_size
        
        self.embedding=nn.Embedding(self.event_size, 128) # for eventID emebedding
            
        if self.mode == 'paramEncoder': 
            encoder_layer = TransformerEncoderLayer(
                dim, nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True,
                param_vocab_size=self.param_vocab_size, param_embed_dim=param_embed_dim)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                dim, nhead, dim_feedforward, dropout, batch_first=True)

        self.trans_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(dim, nhead, dim_feedforward, dropout, batch_first=True)
        self.trans_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_layer = nn.Linear(dim, dim)
        
        #self.pos_encoder1 = PositionalEncoding(d_model=dim, max_len=window_size)
        ### self.pos_encoder2 = LearnedPositionEncoding(
        ###    d_model=768, max_len=window_size)
        ## self.fc1 = nn.Linear(dim * window_size, 2)
        ## self.fc1 = nn.Linear(dim , 1)
        ## self.sigmoid=nn.Sigmoid()

        # Linear over sequence dim (transpose-based)
        self.linear_seq =  nn.Linear(self.window_size, self.window_size)  # nn.Identity()  # Placeholder, dynamically created later

        # Dim reduction
        self.reduce_em_dim = nn.Linear(dim, 1)
        self.reduce_win_dim = nn.Linear(self.window_size, 1)
        self.expand_win_dim = nn.Linear(1, dim_ff_reduced)

        # Transformer encoder on reduced dims
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=eventID_embed_dim, nhead=2, dim_feedforward=dim_ff_reduced, dropout=dropout, batch_first=True)
        self.trans_encoder_seq = nn.TransformerEncoder(encoder_layer2, num_layers=1)

         # Decoder  on reduced dims
        decoder_layer2 = nn.TransformerDecoderLayer(d_model=eventID_embed_dim, nhead=2, dim_feedforward=dim_ff_reduced, dropout=dropout, batch_first=True)
        self.trans_decoder_seq = nn.TransformerDecoder(decoder_layer2, num_layers=1)

        # Output projection
        self.output_layer_seq = nn.Linear(eventID_embed_dim, self.event_size)
        
        # # Final prediction layer
        # self.out = nn.Sigmoid()
        
    def process_window(self, xs,xc):
        """
        xs: the sequences, [seq_len, 1]
        xc: the context attention vector , [seq_len, window_size]
        Concern the attention vector from Context Builder. 
        element multiply the attention vector to the context.
        Alternative: multiply on spicific features. can try in the future.(TF)
        """
        # Initialize bb as a tensor of the same shape as aa
        bb = torch.zeros_like(xc)

        if xc.size(0)>self.window_size:
            bb[:self.window_size, :]=xc[:self.window_size,:]
            # Iterate over each row 
            for j in range(self.window_size, xc.size(0)):
                bb[j]=xs[ max(0, j-self.window_size):j,0].flip(0) * xc[j, :]
        else:
            bb=xc
            
        return bb

    # dynamically update param_vocab_size at runtime by replacing the nn.Embedding layer in  model.  
    def update_param_vocab(self, new_vocab_size):
        for layer in self.trans_encoder.layers:
            if hasattr(layer.self_attn, "char_embedding"):
                old_weights = layer.self_attn.char_embedding.weight.data
                old_vocab_size, embed_dim = old_weights.shape
    
                # Create new embedding
                new_embedding = nn.Embedding(new_vocab_size, embed_dim)
                num_to_copy = min(old_vocab_size, new_vocab_size)
                new_embedding.weight.data[:num_to_copy] = old_weights[:num_to_copy]
    
                # Replace
                layer.self_attn.char_embedding = new_embedding
            

    
    def forward(self, x):
        """
        x: shape (batch_size, seq_len, [field_name_content_embedding_dim, field_value, context_vector])
        returns: (batch_size, seq_len) or (batch_size,) if use_cls_token is True
        Note: no batch_size in dataloader
        """
        # batch_size, seq_len, _ = x.shape  # seq_len, _

        ################## transformer on each event############################
        x1=x[:,0:self.dim]
        # x1 = self.pos_encoder1(x1)
        # x = self.pos_encoder2(x)

        # if mask_input:
        #     x1 = random_masking(x1)

             
        if self.mode == 'paramEncoder': 
            x_combined = torch.cat((x1, x[:,self.dim].unsqueeze(1)), dim=1)
            x1 = self.trans_encoder(x_combined) # append the value column            
        else:
            x1 = self.trans_encoder(x1)  # mask默认None
        
        # Decode from latent
        recon1 = self.trans_decoder(x1, x1)
        recon1 = self.output_layer(recon1)

         # Transpose and apply sequence-wise linear layer
        x1_seq = recon1.transpose(0, 1)  # (batch, dim, seq)
        # if isinstance(self.linear_seq, nn.Identity):
        #     self.linear_seq = nn.Linear(x1_seq.shape[-1], x1_seq.shape[-1])  # Learn over seq_len

        if self.linear_seq.in_features !=x1_seq.shape[-1]:
            self.linear_seq=nn.Linear( x1_seq.shape[-1], x1_seq.shape[-1]).to(x1_seq.device)  # Learn over seq_len
        
        x1_seq = self.linear_seq(x1_seq)
        x1 = x1_seq.transpose(0, 1)  # Back to (batch, seq, dim)

        ######################## transformer on seq ##########################
        # Reduce embedding_dim → 1
        x1_reduced = self.reduce_em_dim(x1)  # (batch, seq, 1)

        x_window=self.process_window(x1_reduced, x[:, -self.window_size:]) # (batch, seq, window_size)
        x_window=x_window.unsqueeze(-1)  # (batch, seq, window_size,1)
        x_window=self.expand_win_dim(x_window)
        
        # Transformer on event sequence 
        x_window = self.trans_encoder_seq(x_window)  # (batch, seq, windwo_size, eventID_embed_dim)
        recon2=self.trans_decoder_seq(x_window, x_window)
        recon2=self.output_layer_seq(recon2)
        # # print(x_trans.size())

        return recon1, recon2

        # x_trans=self.reduce_win_dim(x_trans)  # (batch, seq,1)
        
        # # Final prediction
        # scores = self.out(x_trans).squeeze(-1)  # (batch, seq)

        # # if scores.ndim==0:
        # #     scores=torch.tensor([scores])
        
        # return scores  # (batch, seq_len)

    # def random_masking(x, mask_ratio=0.15):
    #     """
    #     Mask input randomly for augmentation.
    #     Args:
    #         x: Tensor (B, seq_len, dim)
    #     Returns:
    #         masked_x: same shape
    #     """
    #     mask = torch.rand(x.shape[:1], device=x.device) < mask_ratio
    #     mask = mask.unsqueeze(-1).expand_as(x)
    #     return x.masked_fill(mask, 0.0)
        
