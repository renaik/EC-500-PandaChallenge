import math
import warnings
import torch
import torch.nn as nn
from einops import rearrange

from ..utils.layers import *


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("Mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        
        return tensor



def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # Adding residual consideration.
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]

    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)

    return joint_attention



class MLP(nn.Module):
    """ Multi-Layer Perceptron Layer """

    def __init__(self, d_in, d_hidden=None, d_out=None, dropout=0.0):
        super(MLP, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden or d_in
        self.d_out = d_out or d_in
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_in, d_hidden or d_in)
        self.fc2 = nn.Linear(d_hidden or d_in, d_out or d_in)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

    def relprop(self, cam, **kwargs):
        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.gelu.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)

        return cam



class MSA(nn.Module):
    """ Multi-Head Self Attention Layer """

    def __init__(self, d_model, n_head, qkv_bias=False, attn_dropout=0., fc_dropout=0.):
        super(MSA, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.fc_dropout = nn.Dropout(fc_dropout)

        self.d_k = d_model // n_head # Head dimension.
        self.temperature = self.d_k ** 0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.fc = nn.Linear(d_model, d_model)
        self.softmax = Softmax(dim=-1)

        self.attention = einsum('bhid,bhjd->bhij') # q * k^T.
        self.out = einsum('bhij,bhjd->bhid') # attn * v.

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        # [batch, num_patches, embed_dim] --> [batch, num_patches, 3*embed_dim]
        qkv = self.qkv(x)

        # [batch, num_patches, 3*embed_dim] --> 3 x [batch, num_heads, num_patches, head_dim]
        b, n, _, h = *x.shape, self.n_head
        q, k, v = rearrange(qkv, 'b n (qkv d h) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)

        attn = self.attention([q, k]) * self.temperature
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # Get attention.
        if False:
            from os import path

            if not path.exists('att_1.pt'):
                torch.save(attn, 'att_1.pt')
            elif not path.exists('att_2.pt'):
                torch.save(attn, 'att_2.pt')
            else:
                torch.save(attn, 'att_3.pt')

        # Comment in training.
        #if x.requires_grad:
            #self.save_attn(attn)
            #attn.register_hook(self.save_attn_gradients)

        out = self.out([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.fc(out)
        out = self.fc_dropout(out)

        return out

    def relprop(self, cam, **kwargs):
        cam_out = self.fc_dropout(cam_out, **kwargs)
        cam_out = self.fc(cam_out, **kwargs)
        cam_out = rearrange(cam_out, 'b n (h d) -> b h n d', h=self.n_head)

        (cam_attn, cam_v) = self.out.relprop(cam_out, **kwargs)
        cam_attn /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam_attn)

        cam_attn = self.attn_dropout.relprop(cam_attn, **kwargs)
        cam_attn = self.softmax.relprop(cam_attn, **kwargs)
        (cam_q, cam_k) = self.attention.relprop(cam_attn, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.n_head)
        cam = self.qkv.relprop(cam_qkv, **kwargs)

        return cam


class MSABlock(nn.Module):
    """ Multi-Head Self Attention Block """

    def __init__(self, d_model, n_head, mlp_ratio=4., qkv_bias=False, attn_dropout=0., fc_dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_hidden = int(d_model * mlp_ratio)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.msa = MSA(d_model, n_head, qkv_bias, attn_dropout, fc_dropout)
        self.mlp = MLP(d_model, self.d_hidden, fc_dropout)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.msa(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        
        return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.msa.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)

        return cam



class ViT(nn.Module):
    """ Vision Transformer """

    def __init__(self, d_model, n_class, n_layer=3, n_head=8, mlp_ratio=2.,
                 qkv_bias=False, mlp_head=False, attn_dropout=0., fc_dropout=0.):
        super(ViT, self).__init__()
        self.n_class = n_class
        self.d_model = d_model
        self.msa_blocks = nn.ModuleList([
            MSABlock(d_model, n_head, mlp_ratio, qkv_bias, attn_dropout, fc_dropout) for i in range(n_layer)
            ])
        self.norm = nn.LayerNorm(self.d_model, eps=1e-6)
        if mlp_head:
            self.head = MLP(self.d_model, int(self.d_model * mlp_ratio), n_class)
        else:
            self.head = nn.Linear(self.d_model, n_class)
        self.pool = IndexSelect()
        self.add = Add()
        self.inp_grad = None

    def save_inp_grad(self, grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        #if x.requires_grad:
            #x.register_hook(self.save_inp_grad)     # Comment it in train.

        for msa_blk in self.msa_blocks:
            x = msa_blk(x)

        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        
        return x

    def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        for msa_blk in reversed(self.msa_blocks):
            cam = msa_blk.relprop(cam, **kwargs)

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            cam = cam.sum(dim=1)

            return cam

        elif method == "rollout":
            attn_cams = []
            for msa_blk in self.msa_blocks:
                attn_heads = msa_blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]

            return cam
        
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for msa_blk in self.msa_blocks:
                grad = msa_blk.attn.get_attn_gradients()
                cam = msa_blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]

            return cam
            
        elif method == "last_layer":
            cam = self.msa_blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.msa_blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]

            return cam

        elif method == "last_layer_attn":
            cam = self.msa_blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]

            return cam

        elif method == "second_layer":
            cam = self.msa_blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.msa_blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]

            return cam