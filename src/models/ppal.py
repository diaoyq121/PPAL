import torch.nn as nn
from src import mynn
import clip
import torch
import pandas as pd
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
import numpy as np
from src.config import load_config
from src.utils import get_device
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            if False:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
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

class FeaturePrototypeClassifierWithLearnableEmbedding(nn.Module):
    def __init__(self, feature_dim=512, num_classes=2):
        super(FeaturePrototypeClassifierWithLearnableEmbedding, self).__init__()
        config = load_config('config.yaml')
        device = get_device(cuda_idx=config['cuda_device'])
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Learnable embeddings for AD and CN
        self.learnable_ad = nn.Parameter(torch.randn(1, feature_dim))  # (1, 512)
        self.learnable_cn = nn.Parameter(torch.randn(1, feature_dim))  # (1, 512)

        # Linear layer to project the combined features for classification
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.cross11 = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8).to(device)
        self.cross12 = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8).to(device)
        self.cross21 = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8).to(device)
        self.cross22 = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8).to(device)

        self.ffn1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),  # 使用 GELU 替代 ReLU
            nn.Linear(1024, 512),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),  # 使用 GELU 替代 ReLU
            nn.Linear(1024, 512),
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.alpha_ad = nn.Parameter(torch.tensor(0.5))  # Learnable weight for AD
        self.alpha_cn = nn.Parameter(torch.tensor(0.5))  # Learnable weight for CN
        # self.m = 0.99


    def forward(self, ad, cn, text_ad, text_cn):
        """
        ad: Tensor of shape (b, 2, 512)
        cn: Tensor of shape (b, 2, 512)
        text_ad: Tensor of shape (1, 512)
        text_cn: Tensor of shape (1, 512)
        """
        batch_size = ad.size(0)
        # print(ad.shape)
        ad = ad.unsqueeze(1)
        cn = cn.unsqueeze(1)


        # Expand learnable embeddings to batch size
        learnable_ad = self.learnable_ad.repeat(batch_size, 1).unsqueeze(1)  # (b, 512)
        learnable_cn = self.learnable_cn.repeat(batch_size, 1).unsqueeze(1) # (b, 512)

        cls_ad,_ = self.cross11(learnable_ad,ad,ad)
        cls_cn,_ = self.cross11(learnable_cn,cn,cn)

        # cls_ad = self.norm(cls_ad+learnable_ad)
        # cls_cn = self.norm(cls_cn+learnable_cn)
        # print(cls_ad.shape)

        # Combine learnable and text features

        combined_text_ad = cls_ad.squeeze(1) + text_ad
        combined_text_cn = cls_cn.squeeze(1) + text_cn
        # print(combined_text_ad.shape)

        combined_text_ad = combined_text_ad.unsqueeze(1)
        combined_text_cn = combined_text_cn.unsqueeze(1)


        # Combine image and text features

        # ad_combined = ad.mean(dim=1) + combined_text_ad  # (b, 512)
        # cn_combined = cn.mean(dim=1) + combined_text_cn  # (b, 512)
        ad_combined,_ = self.cross22(combined_text_ad,ad,ad)  # (b, 512)
        cn_combined,_ = self.cross22(combined_text_cn,cn,cn)  # (b, 512)

        ad_out = self.norm((ad_combined+combined_text_ad))
        cn_out = self.norm((cn_combined+combined_text_cn))

        ad_out = self.ffn1(ad_out)
        cn_out = self.ffn2(cn_out)

        ad_out = ad_out.mean(dim=0)
        cn_out = cn_out.mean(dim=0)


        # Aggregate features
        # final_features = (ad_combined + cn_combined) / 2  # (b, 512)
        #
        # # Classify using a linear layer
        # logits = self.classifier(final_features)  # (b, num_classes)
        # probabilities = self.softmax(logits)  # (b, num_classes)

        return ad_out,cn_out



class PPAL(nn.Module):
   
    def __init__(self, backbone, embedding_dim, num_slices, num_classes, dropout=0.4, return_attention_weights=False):
        super().__init__()
        # text_data = pd.read_csv('/home/diaoyueqin/AXIAL-main/textprompt.csv')
        text_data = np.array(pd.read_csv('/home/diaoyueqin/AXIAL-main/textprompt.csv', header=None)).squeeze()
        # print(text_data.shape)
        # text_data = df.values.flatten()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pro=FeaturePrototypeClassifierWithLearnableEmbedding(feature_dim=512, num_classes=num_classes)
        self.num_slices = num_slices
        # self.net = ResNet3D()
        self.feat_map_dim = embedding_dim
        self.backbone = backbone
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.attention = mynn.AttentionLayer(input_size=embedding_dim)
        self.attention2 = mynn.AttentionLayer(input_size=embedding_dim)
        self.attention3 = mynn.AttentionLayer(input_size=embedding_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            # nn.Linear(in_features=embedding_dim, out_features=num_classes),
            # nn.Linear(in_features=512, out_features=256),
            # nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes),
        )
        self.mlp = nn.Linear(in_features=512, out_features=512)
        self.mlp1 = nn.Linear(in_features=1024, out_features=512)
        self.mlp2 = nn.Linear(in_features=2048, out_features=512)
        self.mlp3 = nn.Linear(in_features=512, out_features=1024)

        self.return_attention_weights = return_attention_weights

        clip_model, _ = clip.load("RN50", device="cpu")

        self.prompt_learner = PromptLearner(text_data, clip_model.float())
        self.learnable_image_center = nn.Parameter(torch.zeros(*[2, 1, 512])).float()


        self.text_encoder = TextEncoder(clip_model.float())
        self.norm = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(1024)

        self.weight_param = nn.Parameter(torch.randn(2))
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )
        self.sig = nn.Softmax(dim=-1)


    def forward(self, x, y='?', training=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # config = load_config('config.yaml')
        # # Get the device to use
        # device = get_device(cuda_idx=config['cuda_device'])

        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        texta_features = text_features
    


        batch_size = x.size(0)

        x = x.view(-1, *x.size()[2:])  # e.g. (32, 80, 3, 224, 224) -> (32 * 80, 3, 224, 224)
   
        x = self.backbone(x)
   
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)

      

        x = x.view(-1, self.num_slices, *x.size()[1:])
       
        x, attention_weights = self.attention(x)  # e.g. (32, 80, 1280) -> (32, 1280)
       
        out = self.classifier(x)  # e.g. (32, 1280) -> (32, num_classes)
        t = self.sig(out)
        t = t.unsqueeze(2)
        a = x.unsqueeze(1).repeat(1, 2, 1)
        ad = a * t
        #
        cn0 = ad[:, 1, ]
        ad1 = ad[:, 0, ]
        #
        # texta_features = texta_features.unsqueeze(0)
        texta_features = self.mlp1(texta_features)
        ad, cn = self.pro(ad1, cn0, texta_features[0], texta_features[1])
        ad, cn = self.pro(ad1, cn0, ad, cn)
        # print(ad)
        ad, cn = self.pro(ad1, cn0, ad, cn)
      
        axial_text = torch.cat((ad, cn), dim=0)
       
        imge = x.clone()
        axial_text = axial_text.clone()

        logits_axial = imge @ axial_text.transpose(0, 1)
      
        if training:
            return logits_axial,out, logits_axial  # , mean_tensor

        if self.return_attention_weights:
            return x, attention_weights
        return logits_axial#+out



