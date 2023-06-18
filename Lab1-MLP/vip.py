import torch.nn as nn

class WeightedMLP(nn.Module):
    def __init__(self, dim, qkv_bias=False, dropout=0.7):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)
        self.reweight = nn.Sequential(nn.Linear(dim, dim//4), nn.ReLU(), nn.Linear(dim//4, dim*3))
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        '''
        Input:
        x :[B, H, W, C]
        '''
        B, H, W, C = x.shape
        print(x.shape)
        num_heads = H
        head_dim = C // H
        # col-wise
        h = x.reshape(B, H, W, num_heads, head_dim).permute(0, 3, 2, 1, 4).reshape(B, num_heads, W, H*head_dim)
        h = self.mlp_h(h).reshape(B, num_heads, W, H, head_dim).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)
        # row-wise
        w = x.reshape(B, H, W, num_heads, head_dim).permute(0, 1, 3, 2, 4).reshape(B, H, num_heads, W*head_dim)
        w = self.mlp_w(w).reshape(B, H, num_heads, W, head_dim).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)
        # channel-wise
        c = self.mlp_c(x)
        
        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)  #keep channel, mean pooling across the whole image, [B, C]
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2) #[3, B, 1, 1, C]

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.out_proj(x)
        x = self.drop(x)
        return x
        
class PermutatorBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.vip_block = WeightedMLP(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.ReLU(), nn.Linear(dim*4, dim))
         
    def forward(self, x):
        x = self.vip_block(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x
    
class Vip(nn.Module):
    def __init__(self, input_size=32, embed_dim=384, patch_size=2, num_layers=6, num_classes=100):
        super().__init__()
        self.embedding = nn.Conv2d(in_channels=1, 
                                   out_channels=embed_dim, 
                                   kernel_size=patch_size, 
                                   stride=patch_size, 
                                   )
        encoder_list = []
        for i in range(num_layers):
            encoder_list.append(PermutatorBlock(embed_dim))
        self.encoder = nn.ModuleList(encoder_list)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        '''
        Input:
        x: [B, C, H, W]
        '''
        x = self.embedding(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)   #[B, H, W, C]
        for module in self.encoder:
            x = module(x)
        x = self.norm(x)
        x = x.reshape((B, H*W, C)).mean(1)  #[B, C]
        x = self.head(x)
        return x