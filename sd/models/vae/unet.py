from torch import nn 
import torch 
from sd.models.vae.attention import LinearAttention
import numpy as np 

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)


def nonlinearity(x):
    return x * torch.sigmoid(x)

class ResnetBlock(nn.Module):

    def __init__(self,
                 *,
                 in_channels,
                 out_channels=None,
                 conv_shortcut=False,
                 dropout,
                 temb_channels=512):
        
        super().__init__()
        self.in_channels = in_channels
        
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels=in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
            
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels,
                                     out_channels=out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels=out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
                
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
                

            
    def forward(self, x, temb):
        h = x 
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]


        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)


        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)

            else:
                x = self.nin_shortcut(x)


        return x + h 
        
class AttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        self.k = torch.nn.Conv2d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        self.v = torch.nn.Conv2d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        

        self.proj_out = torch.nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        

    def forward(self, x):
        h_ = x 
        h_ = self.norm(h_)
        
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention 
        b, c, h, w = q.shape 
        q = q.reshape(b, c, h * w)
        q = torch.permute(input=q,
                          dims=(0, 2, 1))  # b, hw, c 
        
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values 
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)

        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_
    
        
class LinAttentionBlock(LinearAttention):
    """to match AttnBlock usage."""

    def __init__(self,
                 in_channels):
        super().__init__(dim=in_channels,
                         heads=1,
                         dim_head=in_channels)
        




def make_attn(in_channels, 
              attn_type="vanilla"):
    
    assert attn_type in ['vanilla', 'linear', 'none'], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")

    if attn_type == "vanilla":
        return AttnBlock(in_channels=in_channels)
    
    elif attn_type == "none":
        return nn.Identity(in_channels)
    
    else:
        return LinAttentionBlock(in_channels=in_channels)

class Downsample(nn.ModuleList):

    def __init__(self, in_channels, with_conv):
        super().__init__()

        self.with_conv = with_conv
        if self.with_conv:

            # no asymmetric padding in torch conv, must do it ourselves 
            self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)
            

    def forward(self, x):

        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)

        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        return x 
    

class Encoder(nn.Module):

    def __init__(self, 
                 *,
                 ch,
                 out_ch,
                 ch_mult=(1,  2, 4, 8),
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 resolution,
                 z_channels,
                 double_z=True,
                 use_linear_attn=False,
                 attn_type="vanilla",
                 **ignore_kwargs):
        
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch 
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling 
        self.conv_in = torch.nn.Conv2d(in_channels,
                                        self.ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                

                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(in_channels=block,
                                          attn_type=attn_type))
                    
            down = nn.ModuleList()
            down.block = block 
            down.attn = attn 
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(in_channels=block_in,
                                             with_conv=resamp_with_conv)
                curr_res = curr_res // 2 

            self.down.append(down)


        # middle 
        self.mid = nn.ModuleList()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        

        self.mid.attn_1 = make_attn(in_channels=block_in,
                                    attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout
                                       )
        

        # end 
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                     2 * z_channels if double_z else z_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        


    def forward(self, x):

        # timestep embedding 
        temb = None 

        # downsampling 
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

                hs.append(h)


            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))



        # middle 
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)


        # end 
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h 
    


# ------------------------------------------------------------------------------------------------------------------

class LatentRescaler(nn.Module):

    def __init__(self, 
                 factor, 
                 in_channels, 
                 mid_channels,
                out_channels,
                 depth=2):
        
        super().__init__()
        self.factor = factor

        self.conv_in = nn.Conv2d(in_channels=in_channels,
                                 out_channels=mid_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        
        self.res_block1 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])
        
        self.attn = AttnBlock(mid_channels)
        self.res_block2 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])
        

        self.conv_out = nn.Conv2d(mid_channels,
                                  out_channels=out_channels,
                                  kernel_size=1)
        


    def forward(self, x):

        x = self.conv_in(x)
        for block in self.res_block1:
            x = block(x, None)

        x = torch.nn.functional.interpolate(x, size=(int(round(x.shape[2]*self.factor)),
                                                     int(round(x.shape[3]*self.factor))))
        x = self.attn(x)
        for block in self.res_block2:
            x = block(x, None)

        x = self.conv_out(x)
        return x 
    




class Upsample(nn.Module):

    def __init__(self,
                 in_channels,
                 with_conv):
        
        super().__init__()
        self.with_conv = with_conv

        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
            

        
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)

        return x 
    




    



class Decoder(nn.Module):

    def __init__(self,
                 *,
                 ch,
                 out_ch,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 resolution,
                 z_channels,
                 give_pre_end=False,
                 tanh_out=False,
                 use_linear_attn=False,
                 attn_type="vanilla",
                 **ignorekwargs):
        
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch 
        self.temb_ch = 0 
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out


        # compute in_ch_mult, block_in and curr_res at lowest res 
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dims.")


        # z to block_in 
        self.conv_in = torch.nn.Conv2d(in_channels=z_channels,
                                       out_channels=block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        


        # middle 
        self.mid = nn.ModuleList()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        

        self.mid.attn_1 = make_attn(in_channels=block_in,
                                    attn_type=attn_type)
        
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)
        


        # upsampling 
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(in_channels=block_in,
                                          attn_type=attn_type))
                    
                
            up = nn.Module()
            up.block = block 
            up.attn = attn 
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2 

            self.up.insert(0, up)


        # End 
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels=out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        



    def forward(self, z):
        
        self.last_z_shape = z.shape 

        # timestep embeddings 
        temb = None

        # z to block_in 
        h = self.conv_in(z)

        # middle 
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)


        # upsampling 
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)

                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)


            if i_level != 0:
                h = self.up[i_level].upsample(h)


        # end 
        if self.give_pre_end:
            return h 
        

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)

        return h 
        


# -------------------------------------------------------------------------------------------------------------
class SimpleDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 *args,
                 **kwargs):
        super().__init__()

        self.model = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                                              out_channels=in_channels, 
                                              kernel_size=1,
                                              ),
                                              
                                    ResnetBlock(in_channels=in_channels,
                                                out_channels= 2 * in_channels,
                                                temb_channels=0, 
                                                dropout=0.0),
                                    ResnetBlock(in_channels=in_channels * 2,
                                                out_channels= 4 * in_channels,
                                                temb_channels=0, 
                                                dropout=0.0),
                                    ResnetBlock(in_channels=in_channels * 4,
                                                out_channels= 2 * in_channels,
                                                temb_channels=0, 
                                                dropout=0.0),

                                    nn.Conv2d(in_channels=2 * in_channels,
                                              out_channels = in_channels,
                                              kernel_size = 1
                                              ),
                                    Upsample(in_channels, with_conv=True)
                                                
                                 ])
        

        # end 
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        


    def forward(self, x):

        for i, layer in enumerate(self.model):
            if i in [1, 2, 3]:
                x = layer(x, None) 

            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)

        return x 
    


class UpsampleDecoder(nn.Module):

    def __init__(self, 
                 in_channels,
                 out_channels,
                 ch,
                 num_res_blocks,
                 resolution,
                 ch_mult=(2, 2),
                 dropout=0.0):
        
        super().__init__()

        # upsampling 
        self.temb_ch = 0 
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout))
                
                block_in = block_out


            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(in_channels=block_in,
                                                     with_conv=True))
                curr_res = curr_res * 2 


        # end 
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        


    def forward(self, x):

        # upsampling 
        h = x 
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)


            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)

            
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h 
    

# ---------------------------------------------------------------------------------------------------------------------------




class MergedRescaleEncoder(nn.Module):

    def __init__(self, 
                 in_channels, 
                 ch,
                 resolution,
                 out_ch,
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0.0,
                 resamp_with_conv=True,
                 ch_mult=(1, 2, 4, 8),
                 rescale_factor = 1.0,
                 rescale_module_depth=1):
        
        super().__init__()
        intermediate_chn = ch * ch_mult[-1]

        self.encoder = Encoder(ch=ch,
                               out_ch=out_ch,
                               ch_mult=ch_mult,
                               num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions,
                               dropout=dropout,
                               resamp_with_conv=resamp_with_conv,
                               in_channels=in_channels,
                               resolution=resolution,
                               z_channels=intermediate_chn, # error 
                               double_z=False,
                               

                               )
        
        # self.rescaler = LatentRescaler(factor=rescale_factor,
        #                                in_channels=intermediate_chn,
        #                                mid_channels=intermediate_chn,
        #                                out_channels=out_ch,
        #                                depth=rescale_module_depth)
        



    def forward(self, x):

        x = self.encoder(x)
        x = self.rescaler(x)
        return x 
    








        



if __name__ == "__main__":
    # x = torch.rand(4, 3, 256, 256)
    device = torch.device("cuda")
    # print(device)

    # encoder = Encoder(
    # ch=64,  # Base number of channels
    # out_ch=3,  # Output channels (e.g., for RGB images)
    # ch_mult=(1, 2, 4, 8),  # Channel multipliers
    # num_res_blocks=2,  # Number of residual blocks per resolution
    # attn_resolutions=[16],  # Apply attention at 16x16 resolution
    # dropout=0.1,  # Dropout rate
    # resamp_with_conv=True,  # Use convolutional resampling
    # in_channels=3,  # Input channels (e.g., RGB images)
    # resolution=256,  # Input resolution
    # z_channels=256,  # Latent space channels
    # double_z=True,  # Double the latent space channels
    # use_linear_attn=False,  # Don't use linear attention
    # attn_type="vanilla",  # Use standard self-attention
    # )

    # output = encoder(x).to(device)
    # print(output.shape)

# ------------------------------------------------------------------------------------------------------------------------------------------

    # z = torch.randn(4, 512, 32, 32)
    # decoder_config = {
    # "ch": 64,
    # "out_ch": 3,
    # "ch_mult": (1, 1, 2, 4),  # (1, 2, 4, 8)
    # "num_res_blocks": 2,
    # "attn_resolutions": [16],
    # "dropout": 0.1,
    # "resamp_with_conv": True,
    # "in_channels": 256,
    # "resolution": 256,
    # "z_channels": 256,
    # "give_pre_end": False,
    # "tanh_out": True,
    # "use_linear_attn": False,
    # "attn_type": "vanilla",
    # }

    # ##  Instantiate the Decoder
    # decoder = Decoder(**decoder_config).to(device)
    # decoder = decoder(z.to(device))
    # print(decoder)


# --------------------------------------------------------------------------------------------
    # x = torch.randn(4, 64, 32, 32)
    # simple_decoder = SimpleDecoder(in_channels=64,
    #                                out_channels=3)(x)
    
    # print(simple_decoder.shape)


# --------------------------------------------------------------------------------------------
    # x = torch.randn(4, 64, 8, 8)
    # upsample_decoder = UpsampleDecoder(in_channels=64, 
    #                                    out_channels=3,
    #                                    ch=64,
    #                                    num_res_blocks=2,
    #                                    resolution=32)(x).to(device)
    # print(upsample_decoder.shape)


# ----------------------------------------------------------------------------------------------
    merged_rescale_encoder = MergedRescaleEncoder(in_channels=3,
                                                  ch=64,
                                                  resolution=32,
                                                  out_ch=3,
                                                  num_res_blocks=2,
                                                  attn_resolutions=[16],
                                                  rescale_factor=2.0
                                                  )
    
    print(merged_rescale_encoder)

    

    # python -m sd.models.vae.unet