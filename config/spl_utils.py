import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import Trainer, EvalPrediction
import wandb
from transformers.optimization import get_cosine_schedule_with_warmup

from torch.distributions import Normal
from pyro.nn import ConditionalAutoRegressiveNN

class PairEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim):
        super(PairEncoder, self).__init__()

        self._model = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, e):
        return self._model(e)


class SequenceEncoder(nn.Module):
    """
    Model to encode sequence of responses
    """

    def __init__(self, input_dim, latent_dim):
        super(SequenceEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.linear = nn.Identity()
        self.w_q = nn.Linear(input_dim, input_dim)
        self.w_k = nn.Linear(input_dim, input_dim)
        self.w_v = nn.Linear(input_dim, input_dim)
        self.mean_layer = nn.Linear(input_dim, latent_dim)
        self.log_var_layer = nn.Linear(input_dim, latent_dim)
        self.layer_norm = nn.Identity()     # nn.LayerNorm(latent_dim)

    def forward(
        self, sequences, seq_start_end
    ):  # (C_1+C_2+...+C_n, D), [(0, C_1), (C_1, C_1+C_2), ..., (C_1+...+C_n-1, C_1+...+C_n)]
        outputs = []
        for _, (start, end) in enumerate(seq_start_end):
            context = sequences[start:end]  # C_i x D
            q = self.w_q(context)
            k = self.w_k(context)
            attention_scores = torch.matmul(
                q, k.transpose(0, 1)
            )
            attention_scores = attention_scores / (context.shape[-1] ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)  # C_i x C_i
            weighted_values = torch.matmul(attention_weights, self.w_v(context))  # C_i x D
            output = torch.mean(weighted_values, dim=0)  # D
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)  # n x D

        mean = self.layer_norm(self.mean_layer(outputs))
        log_var = self.layer_norm(self.log_var_layer(outputs))
        
        return mean, log_var, outputs


class PairConditionalIAFLayer(nn.Module):
    """
    P-IAF
    """
    def __init__(self, z_dim, x_dim, d_dim, s_dim):
        super().__init__()

        P = torch.randperm(z_dim)
        self.mu_net  = ConditionalAutoRegressiveNN(
            input_dim=z_dim, 
            #context_dim=x_dim + d_dim, 
            context_dim=d_dim,
            hidden_dims=[z_dim*2, z_dim*2, z_dim],
            param_dims=[1], 
            permutation=P
        )
        for m in self.mu_net.modules():
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.data.zero_()
                m.bias.requires_grad_(False)

        self.logs_net = ConditionalAutoRegressiveNN(
            input_dim=z_dim, 
            #context_dim=x_dim + s_dim,
            context_dim=s_dim,
            hidden_dims=[z_dim*2, z_dim*2, z_dim],
            param_dims=[1], 
            permutation=P
        )

    def forward(self, z, x, d_ctx, s_ctx):
        #d = torch.cat([x, d_ctx], dim=-1)
        #s = torch.cat([x, s_ctx], dim=-1)
        d = d_ctx
        s = s_ctx
        mu      = self.mu_net(z, d)              # B x Z
        log_s   = self.logs_net(z, s)            # B x Z
        
        mu    = mu[0]   if isinstance(mu, tuple)   else mu
        log_s = log_s[0] if isinstance(log_s, tuple) else log_s
        
        mu = torch.clamp(mu, -1.0, 1.0)
        log_s   = torch.clamp(log_s, -5.0, 5.0)   
        z_new   = mu + torch.exp(0.5 * log_s) * z
        log_det = 0.5 * log_s.float().sum(-1)           
        return z_new, log_det, mu, log_s
    
'''

class Decoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(input_dim + z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, z):  
        
        if x.dim() == 2:    
            x = x.unsqueeze(1)
            
        B, L, D = x.shape
        z = z.unsqueeze(1)  

        z_proj = z.expand(-1, L, -1)
        xz = torch.cat([x, z_proj], dim=-1) 

        r = self.linear1(xz)
        r = self.act(r)
        r = self.linear2(r)
        r = self.act(r)
        r = self.linear3(r).squeeze(-1)  

        return r

'''

class Decoder(nn.Module):

    def __init__(self, x_dim: int, z_dim: int, hidden: int):
        super().__init__()

        self.linear1 = nn.Linear(x_dim + z_dim, hidden // 2)
        self.linear2 = nn.Linear(hidden // 2, hidden)
        self.act = nn.LeakyReLU(0.2)

        self.film_mlp = nn.Sequential(
            nn.Linear(z_dim, hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden * 2, hidden * 2),
        )

        self.out = nn.Linear(hidden, 1)

    def _forward_film(self, x, z):
        """
        x: [B, L, D], z: [B, Z]
        Return: [B, L]
        """

        if x.dim() == 2:                   # [B, D]  →  [B, 1, D]
            x = x.unsqueeze(1)
        
        B, L, D = x.shape
        #print("x_shape =", x.shape)
        z_proj = z.unsqueeze(1).expand(-1, L, -1)       # [B, L, Z]
        xz = torch.cat([x, z_proj], dim=-1)             # [B, L, D+Z]
        #print("xz_shape =", xz.shape)

        h = self.linear1(xz)                            # [B, L, H//2]
        h = self.act(h)
        h = self.linear2(h)                             # [B, L, H]
        h = self.act(h)

        γβ = self.film_mlp(z)                           # [B, 2H]
        #γ, β = γβ.chunk(2, dim=-1)                      # [B, H], [B, H]
        
        γ_raw, β_raw = γβ.chunk(2, dim=-1)                      # [B, H], [B, H]
        
        γ = 1.0 + 0.5 * torch.tanh(γ_raw)
        β = 0.5 * torch.tanh(β_raw)
        
        γ = γ.unsqueeze(1).expand(-1, L, -1)            # [B, L, H]
        β = β.unsqueeze(1).expand(-1, L, -1)            # [B, L, H]

        h = self.act(γ * h + β)                       # [B, L, H]
        out = self.out(h).squeeze(-1)                   # [B, L]
        return out

    def forward(self, x, z):
        r = self._forward_film(x, z)
        return r



class SPLModel(nn.Module):
    def __init__(self, encoder_embed_dim, decoder_embed_dim, hidden_dim, latent_dim, llm_encoder, llm_contexts_encoder,
                 use_iaf, num_iaf_flows, guiding = True,
                 fixed_contexts=False, fixed_llm_embeddings=False, use_causal_lm=False, use_attention_layer=False,
                 use_transformer=False, concat_chosen_rejected=False):
        super(SPLModel, self).__init__()
        self.llm_encoder = llm_encoder
        self.llm_contexts_encoder = llm_contexts_encoder
        self.pair_encoder = PairEncoder(encoder_embed_dim, hidden_dim, latent_dim//2)
        self.sequence_encoder = SequenceEncoder(latent_dim, latent_dim)
        self.decoder = Decoder(decoder_embed_dim, latent_dim, hidden_dim)

        self.d_dim = latent_dim // 2
        self.s_dim = latent_dim // 2

        # SPL Flow
        self.use_iaf = use_iaf
        if self.use_iaf:
             self.iaf_flows = nn.ModuleList()
             for _ in range(num_iaf_flows):
                self.iaf_flows.append(
                    PairConditionalIAFLayer(
                        z_dim=latent_dim,
                        x_dim=latent_dim,
                        d_dim=self.d_dim,
                        s_dim=self.s_dim
                    )
                )
                
        self.guiding = guiding

        self.latent_dim = latent_dim
        self.fixed_contexts = fixed_contexts
        self.fixed_llm_embeddings = fixed_llm_embeddings
        self.use_causal_lm = use_causal_lm
        self.use_attention_layer = use_attention_layer
        self.use_transformer = use_transformer
        self.concat_chosen_rejected = concat_chosen_rejected

        self.saved_embeddings = torch.Tensor(4, latent_dim)
        self.saved_embeddings.uniform_(-1, 1)

    
    def _summarize_pair_contexts(self, ctx_c, ctx_r, seq_start_end):
        d_all = (ctx_c - ctx_r) * 0.5 # [sum C_i, D]
        s_all = (ctx_c + ctx_r) * 0.5 # [sum C_i, D]

        d_list, s_list = [], []
        for (start, end) in seq_start_end:
            d_list.append(d_all[start:end].mean(dim=0))
            s_list.append(s_all[start:end].mean(dim=0))
        d_ctx = torch.stack(d_list, dim=0)  # [B, D]
        s_ctx = torch.stack(s_list, dim=0)  # [B, D]
        return d_ctx, s_ctx
    
    def reparameterization(self, mean, std, deterministic=False, flow_context=None):
        # base sampl
        if deterministic:
            eps = torch.zeros_like(std)  # No noise for eval
        else:
            eps = torch.randn_like(std)
        z0 = mean + std * eps  # z_0
        log_q0 = Normal(mean.float(), std.float()).log_prob(z0.float()).sum(-1)

        if not self.use_iaf:
            return z0, log_q0, [], []

        x, d_ctx, s_ctx = flow_context
        d_ctx = d_ctx.to(torch.bfloat16)
        s_ctx = s_ctx.to(torch.bfloat16)
        
        # IAF 
        B = mean.size(0)
        log_det = torch.zeros(B, device=mean.device)
        zk = z0
        
        last_mu, last_log_s = None, None
        for iaf in self.iaf_flows:
            zk, ld, mu, log_s = iaf(zk, x, d_ctx, s_ctx)
            log_det = log_det + ld
            
            last_mu, last_log_s = mu, log_s

        log_qk = log_q0 - log_det  # q_K(z_K)
        return zk, log_qk, last_mu, last_log_s
    
    
    def encode_pair(self, e_c, e_r):
        return self.pair_encoder(e_c), self.pair_encoder(e_r)

    def encode_sequence(self, sequences, seq_start_end):
        return self.sequence_encoder(sequences, seq_start_end)

    def decode(self, e_c, e_r, z):
        return self.decoder(e_c, z), self.decoder(e_r, z)

    def forward(
        self,
        target_chosen,
        target_rejected,
        context_chosen,
        context_rejected,
        seq_start_end,
        user_type,
        ground_truth_user_vector=False,
    ):
        pair_embed_chosen, pair_embed_rejected = self.encode_pair(context_chosen, context_rejected)
        
        pair_embed = torch.cat([pair_embed_chosen, pair_embed_rejected], dim=1)
        mean, log_var, x = self.encode_sequence(pair_embed, seq_start_end)
        
        with torch.no_grad():
            pair_embed_swap = torch.cat([pair_embed_rejected, pair_embed_chosen], dim=1)
            mean_swap, log_var_swap, x_swap = self.encode_sequence(pair_embed_swap, seq_start_end)
        
        mean = torch.clamp(mean, -1, 1)
        log_var = torch.clamp(log_var, -1, 1)
        mean_swap = torch.clamp(mean_swap, -1, 1)
        log_var_swap = torch.clamp(log_var_swap, -1, 1)
        
        if ground_truth_user_vector:
            z = torch.zeros_like(mean)
            self.saved_embeddings = self.saved_embeddings.to(mean.device)
            for idx in range(user_type.shape[0]):
                z[idx] = self.saved_embeddings[int(user_type[idx])]
        else:
            d_ctx, s_ctx = self._summarize_pair_contexts(pair_embed_chosen, pair_embed_rejected, seq_start_end)
            
            z, log_qz, last_mu, last_log_s = self.reparameterization(
                mean, torch.exp(0.5 * log_var), deterministic=not self.training, flow_context=(x, d_ctx, s_ctx)
            )
            
        rc, rr = self.decode(target_chosen, target_rejected, z)
        
        #with torch.no_grad():
        #    z_noise = torch.randn_like(z)
        #    rc_noise, rr_noise = self.decode(target_chosen, target_rejected, z_noise)
        
        # temp
        rc_noise, rr_noise = None, None
        
        if self.use_iaf:
            return rc, rr, mean, log_var, z, log_qz, last_mu, last_log_s, rc_noise, rr_noise, mean_swap, log_var_swap
        # not use
        else:
            return rc, rr, mean, log_var, z, rc_noise, rr_noise, mean_swap, log_var_swap

    def save_model(self, path):
        torch.save(self, path)


class SPLTrainer(Trainer):
    def __init__(
        self, *args, lr_lambda=None, kl_loss_weight=None, guiding_weight=None, use_annealing=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lr_lambda = lr_lambda
        self.kl_loss_weight = kl_loss_weight
        self.guiding_weight = guiding_weight
        self.use_annealing = use_annealing
        self.annealer = Annealer(
            total_steps=1e4, shape="cosine", baseline=0.0, cyclical=True
        ) 
    
    def get_llama_backbone(self, m):

        base = m

        if hasattr(base, "base_model"):
            base = base.base_model

        if hasattr(base, "model"):
            base = base.model

        if hasattr(base, "model"):
            base = base.model
        return base  
    
    @classmethod
    def compute_active_units(cls, means, threshold=0.005):

        var_mu = torch.var(means, dim=0)
        au = torch.sum(var_mu > threshold).item() 
        return au
    
    @classmethod
    def per_sample_loss(cls, rewards_chosen, rewards_rejected):
        return -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)

    def loss(self, rewards_chosen, rewards_rejected):
        return torch.mean(self.per_sample_loss(rewards_chosen, rewards_rejected))

    def compute_loss(self, wrapped_model, inputs, return_outputs=False):
        if isinstance(wrapped_model, SPLModel):
            model = wrapped_model  # .module
        else:
            model = wrapped_model.module
        device = model.llm_encoder.device
        batch_size = inputs["seq_start_end"].shape[0]

        if model.fixed_llm_embeddings:
            embeddings_chosen = torch.tensor(inputs["embeddings_chosen"]).to(device).bfloat16()
            embeddings_rejected = torch.tensor(inputs["embeddings_rejected"]).to(device).bfloat16()
        else:
            
            input_ids = torch.cat(
                [inputs["input_ids_chosen"], inputs["input_ids_rejected"]], dim=0
            )
            
            attention_mask = torch.cat(
                [inputs["attention_mask_chosen"], inputs["attention_mask_rejected"]], dim=0
            )
            
            backbone = self.get_llama_backbone(model.llm_encoder)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False, 
                    return_dict=True
                )
            embeddings = out.last_hidden_state 
            
            last_idx = attention_mask.sum(dim=1) - 1           # [B*2]
            batch_idx = torch.arange(last_idx.size(0), device=last_idx.device)
            embeddings = embeddings[batch_idx, last_idx]       # → [B*2, D]
            
            
            embeddings_chosen = embeddings[:batch_size]
            embeddings_rejected = embeddings[batch_size:]

        if model.fixed_contexts:
            contexts_embeddings_chosen = torch.tensor(inputs["contexts_embeddings_chosen"]).to(device).bfloat16()
            contexts_embeddings_rejected = torch.tensor(inputs["contexts_embeddings_rejected"]).to(device).bfloat16()
        else:
            input_ids_chosen = inputs["contexts_input_ids_chosen"]
            attention_mask_chosen = inputs["contexts_attention_mask_chosen"]
            token_length_chosen = torch.eq(input_ids_chosen,
                                           model.llm_contexts_encoder.config.pad_token_id).int().argmax(-1) - 1
            input_ids_rejected = inputs["contexts_input_ids_rejected"]
            attention_mask_rejected = inputs["contexts_attention_mask_rejected"]
            token_length_rejected = torch.eq(input_ids_rejected,
                                             model.llm_contexts_encoder.config.pad_token_id).int().argmax(-1) - 1


            with torch.no_grad():
                last_hidden_state_chosen = model.llm_contexts_encoder(
                    input_ids=input_ids_chosen,
                    attention_mask=attention_mask_chosen,
                    output_hidden_states=True
                ).hidden_states[-1]

                weights_for_non_padding_chosen = attention_mask_chosen * torch.arange(
                    start=1, end=last_hidden_state_chosen.shape[1] + 1
                ).unsqueeze(0).to(attention_mask_chosen.device).float()
                sum_embeddings = torch.sum(last_hidden_state_chosen * weights_for_non_padding_chosen.unsqueeze(-1),
                                           dim=1)
                num_of_none_padding_tokens_chosen = torch.sum(weights_for_non_padding_chosen, dim=-1).unsqueeze(-1)
                contexts_embeddings_chosen = sum_embeddings / num_of_none_padding_tokens_chosen
                
                last_hidden_state_rejected = model.llm_contexts_encoder(
                    input_ids=input_ids_rejected,
                    attention_mask=attention_mask_rejected,
                    output_hidden_states=True
                ).hidden_states[-1]

                weights_for_non_padding_rejected = attention_mask_rejected * torch.arange(
                    start=1, end=last_hidden_state_rejected.shape[1] + 1
                ).unsqueeze(0).to(attention_mask_rejected.device).float()
                sum_embeddings = torch.sum(last_hidden_state_rejected * weights_for_non_padding_rejected.unsqueeze(-1),
                                           dim=1)
                num_of_none_padding_tokens_rejected = torch.sum(weights_for_non_padding_rejected, dim=-1).unsqueeze(-1)
                contexts_embeddings_rejected = sum_embeddings / num_of_none_padding_tokens_rejected
        seq_start_end = inputs["seq_start_end"]
        user_type = torch.tensor(inputs["user_type"]).to(device).bfloat16()
        
        
        if model.use_iaf:
            rewards_chosen, rewards_rejected, mean, log_var, z, log_qz, last_mu, last_log_s, rc_noise, rr_noise, mean_swap, log_var_swap = model(
                embeddings_chosen,
                embeddings_rejected,
                contexts_embeddings_chosen,
                contexts_embeddings_rejected,
                seq_start_end,
                user_type,
                ground_truth_user_vector=False, 
            )
        else:
            rewards_chosen, rewards_rejected, mean, log_var, z, rc_noise, rr_noise, mean_swap, log_var_swap = model(
                    embeddings_chosen,
                    embeddings_rejected,
                    contexts_embeddings_chosen,
                    contexts_embeddings_rejected,
                    seq_start_end,
                    user_type,
                    ground_truth_user_vector=False,  
            )
            
            
        reconstruction_loss = self.loss(rewards_chosen, rewards_rejected)
        if self.kl_loss_weight == 0:
            loss = reconstruction_loss
            accuracy = torch.mean((rewards_chosen > rewards_rejected).float())
            if not return_outputs:
                self.log(
                    {
                        "train_recon_loss":reconstruction_loss.mean().item(),
                        "train_accuracy": accuracy.mean().item(),
                        "rewards_chosen": rewards_chosen.mean().item(),
                        "rewards_rejected": rewards_rejected.mean().item(),
                        "embeddings_chosen": embeddings_chosen.mean().item(),
                        "embeddings_rejected": embeddings_rejected.mean().item(),
                        "mean": mean.mean().item(),
                        "log_var": log_var.mean().item()
                    }
                )
        else:

            # SPL
            if model.use_iaf:

                log_pz = Normal(0,1).log_prob(z).sum(-1)

                raw_kld_per = (log_qz - log_pz)                     # [B]  ← per-sample KL

                free_nats = 1.0
                if free_nats > 0.0:
                    raw_kld_per = torch.where(
                        raw_kld_per < free_nats,
                        raw_kld_per.detach(),                        
                        raw_kld_per
                    )

                raw_kld = raw_kld_per.mean() 

                diff_mean = mean - mean_swap
                diff_log_var = log_var - log_var_swap

                log_var_rmse = torch.sqrt((diff_log_var ** 2).mean())
                mean_rmse = torch.sqrt((diff_mean ** 2).mean())
                
                if model.guiding:
                    eta = 0.1
                    def cos_eps(a, b, eps=1e-6):
                        a = a.float()
                        b = b.float()
                        dot = (a * b).sum(dim=-1)
                        norm_a = a.norm(p=2,dim=-1)
                        norm_b = b.norm(p=2,dim=-1)
                        return dot / ((norm_a + eps) * (norm_b + eps))
                    
                    mu_cos = cos_eps(mean, mean_swap.detach())
                    ell_cos = cos_eps(log_var, log_var_swap.detach())
                    
                    guide = 0.5 * (1.0+mu_cos) + eta * 0.5 * (1.0 - ell_cos)
                    guide_loss = guide.mean()
                    
            # Unimodal Gaussian
            else:
                raw_kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).mean()
                        
            if self.use_annealing:
                kld = self.annealer(raw_kld)
                self.annealer.step()
            else:
                kld = raw_kld
            kld = self.kl_loss_weight * kld
            
            # with guiding loss
            if model.guiding:
                loss = reconstruction_loss + kld + self.guiding_weight * guide_loss
                guide_ratio = guide_loss * self.guiding_weight / reconstruction_loss
            # w/o guiding loss
            else:
                loss = reconstruction_loss + kld
            
            accuracy = torch.mean((rewards_chosen > rewards_rejected).float())
            
            #z_noise_diff = self.loss(rc_noise, rr_noise) - reconstruction_loss
            
            if not return_outputs:
                if model.use_iaf:
                    self.log(
                        {
                            "log_qz": log_qz.mean().item(),
                        }
                    )

                if model.guiding:
                    self.log(
                        {
                            "guide_loss": guide_loss.mean().item(),
                            "guide_ratio": guide_ratio.mean().item(),
                        }
                    )
                self.log(
                    {
                        "train_recon_loss": reconstruction_loss.mean().item(),
                        "train_raw_kld": raw_kld.mean().item(),
                        "train_kld": kld.mean().item(),
                        "mean_rmse": mean_rmse.item(),
                        "log_var_rmse": log_var_rmse.item(),
                        "train_accuracy": accuracy.mean().item(),
                        "rewards_chosen": rewards_chosen.mean().item(),
                        "rewards_rejected": rewards_rejected.mean().item(),
                        "embeddings_chosen": embeddings_chosen.mean().item(),
                        "embeddings_rejected": embeddings_rejected.mean().item(),
                        "mean": mean.mean().item(),
                        "mean_swap": mean_swap.mean().item(),
                        "log_var": log_var.mean().item(),
                        "log_var_swap": log_var_swap.mean().item(),
                        #"z_noise_diff": z_noise_diff.mean().item(),
                    }
                )
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
                "mean": mean,
                "last_mean": last_mu,
                "log_var": log_var,
                "z": z,
                "user_type": user_type,
            }
        return loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.03 * num_training_steps),
            num_training_steps=num_training_steps
        )
        self.lr_scheduler = scheduler
        return scheduler

    @classmethod
    def compute_metrics(cls, eval_prediction: EvalPrediction):
        rewards_chosen, rewards_rejected, mean, last_mean, log_var, z, user_type = (
            eval_prediction.predictions
        )
        rewards_chosen = torch.from_numpy(rewards_chosen)
        rewards_rejected = torch.from_numpy(rewards_rejected)
        mean = torch.from_numpy(mean)
        last_mean = torch.from_numpy(last_mean)
        log_var = torch.from_numpy(log_var)
        z = torch.from_numpy(z)
        loss = cls.per_sample_loss(rewards_chosen, rewards_rejected)
        
        # MODIFY
        #kld = -torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
        au = cls.compute_active_units(last_mean)
        accuracy = torch.mean((loss < np.log(2)).float())

        def plot_latent_tsne(latent):
            from sklearn.manifold import TSNE
            z_embedding = TSNE(n_components=2, init='random', perplexity=20, learning_rate="auto").fit_transform(latent.numpy())
            import matplotlib.pyplot as plt
            colors = [f"C{int(i)}" for i in user_type]
            plt.scatter(z_embedding[:, 0], z_embedding[:, 1], c=colors)
            im = wandb.Image(plt)
            plt.close()
            return im
        
        def plot_latent_umap(latent):
            from umap import UMAP  # UMAP import
            import matplotlib.pyplot as plt
            z_embedding = UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42).fit_transform(latent.numpy())
            colors = [f"C{int(i)}" for i in user_type]
            plt.scatter(z_embedding[:, 0], z_embedding[:, 1], c=colors)
            im = wandb.Image(plt)
            plt.close()
            return im    
        
        #im1 = plot_latent_tsne(mean)
        im2 = plot_latent_tsne(z)
        #im3 = plot_latent_umap(mean)
        im4 = plot_latent_umap(z)

        return {
            "loss": loss.mean().item(),
            "accuracy": accuracy.item(),
            #"kld": kld.mean().item(),
            #"mean_embeddings_tsne": im1,
            "z_embeddings_tsne": im2,
            #"mean_embeddings_umap": im3,
            "z_embeddings_umap": im4,
            "active_units": au,
        }

    @classmethod
    def compute_metrics_fast(cls, eval_prediction: EvalPrediction):
        rewards_chosen, rewards_rejected, mean, last_mean, log_var, z, user_type = eval_prediction.predictions
        rewards_chosen = torch.from_numpy(rewards_chosen)
        rewards_rejected = torch.from_numpy(rewards_rejected)
        loss = cls.per_sample_loss(rewards_chosen, rewards_rejected)
        accuracy = torch.mean((loss < np.log(2)).float())
        au = cls.compute_active_units(torch.from_numpy(last_mean))
        return {
            "loss": loss.mean().item(),
            "accuracy": accuracy.item(),
            "active_units": au,
        }


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = "none"
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            y = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError(
                "Cyclical_setter method requires boolean argument (True/False)"
            )
        else:
            self.cyclical = value
        return


import time
from transformers import TrainerCallback

class TrainingPerfCallback(TrainerCallback):
    def __init__(self, measure_flops: bool = True, profile_steps: int = 1, warmup_steps: int = 5, log_to_wandb: bool = True):
        self.measure_flops = measure_flops
        self.profile_steps = profile_steps
        self.warmup_steps = warmup_steps
        self.log_to_wandb = log_to_wandb

        self._train_start = None
        self._step_start = None
        self._durations_ms = []
        self._prof = None
        self._profiled_steps = 0
        self.flops_per_step = None  # raw FLOPs (count), not FLOPS (rate)

        self._param_count = None
        self._eval_seconds = 0.0

    @staticmethod
    def _count_trainable_params(model):
        try:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        except Exception:
            return None

    @staticmethod
    def _world_size(args):
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_world_size()
        except Exception:
            pass
        return max(1, getattr(args, "world_size", 0) or getattr(args, "n_gpu", 0) or torch.cuda.device_count() or 1)

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start = time.perf_counter()
        model = kwargs.get("model", None)
        self._param_count = self._count_trainable_params(model) if model is not None else None

        if self._param_count is not None:
            print(f"[Perf] Trainable parameters: {self._param_count:,}")

        if self.log_to_wandb:
            try:
                import wandb
                if self._param_count is not None:
                    wandb.log({"perf/trainable_params": int(self._param_count)})
            except Exception:
                pass

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.perf_counter()

        if (
            self.measure_flops
            and self.flops_per_step is None
            and (state.global_step or 0) >= self.warmup_steps
            and self._profiled_steps < self.profile_steps
        ):
            try:
                import torch.profiler as profiler
                activities = [profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(profiler.ProfilerActivity.CUDA)
                self._prof = profiler.profile(activities=activities, record_shapes=False, with_flops=True)
                self._prof.__enter__()
            except Exception:
                self._prof = None
                self.measure_flops = False  

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is not None:
            dt_ms = (time.perf_counter() - self._step_start) * 1000.0
            self._durations_ms.append(dt_ms)

        if self._prof is not None:
            try:
                self._prof.step()
                self._prof.__exit__(None, None, None)

                total_flops = 0
                for evt in self._prof.key_averages():
                    f = getattr(evt, "flops", None)
                    if f is not None:
                        total_flops += f
                    if total_flops > 0:
                    self.flops_per_step = int(total_flops)
                self._profiled_steps += 1
                self._prof = None
            except Exception:
                self._prof = None

    # eval runtime 
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        try:
            if metrics and "eval_runtime" in metrics:
                self._eval_seconds += float(metrics["eval_runtime"])
        except Exception:
            pass

    def on_log(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is not None and not model.training:
            return
        
        if len(self._durations_ms) >= 1:
            import numpy as np
            avg_ms = float(np.mean(self._durations_ms[-min(50, len(self._durations_ms)):]))

            if self.log_to_wandb:
                try:
                    import wandb
                    data = {"perf/ms_per_step": avg_ms}
                    if self.flops_per_step is not None:
                        data["perf/FLOPs_per_step(G)"] = self.flops_per_step / 1e9
                    wandb.log(data)
                except Exception:
                    pass

    def on_train_end(self, args, state, control, **kwargs):
        wall_s = (time.perf_counter() - self._train_start) if self._train_start is not None else None
        avg_ms = float(np.mean(self._durations_ms)) if len(self._durations_ms) else None
        ws = self._world_size(args)

        train_wall_s = None
        if wall_s is not None:
            train_wall_s = max(0.0, wall_s - self._eval_seconds)

        total_gpu_hours = (wall_s / 3600.0) * ws if wall_s is not None else None
        train_gpu_hours = (train_wall_s / 3600.0) * ws if train_wall_s is not None else None

        print("===== Training Performance =====")
        if self._param_count is not None:
            print(f"- Trainable parameters: {self._param_count:,}")
        if avg_ms is not None:
            print(f"- Avg ms/step: {avg_ms:.2f} ms")
        if self.flops_per_step is not None:
            print(f"- FLOPs/step: {self.flops_per_step/1e9:.3f} GFLOPs")
        if total_gpu_hours is not None:
            print(f"- GPU-hours (total incl. eval): {total_gpu_hours:.3f} h")
        if train_gpu_hours is not None:
            print(f"- GPU-hours (train only): {train_gpu_hours:.3f} h")
        print("================================")

        if self.log_to_wandb:
            try:
                import wandb
                data = {}
                if self._param_count is not None: data["perf/trainable_params"] = int(self._param_count)
                if avg_ms is not None: data["perf/avg_ms_per_step"] = avg_ms
                if self.flops_per_step is not None: data["perf/FLOPs_per_step(G)"] = self.flops_per_step / 1e9
                if total_gpu_hours is not None: data["perf/GPU_hours_total"] = total_gpu_hours
                if train_gpu_hours is not None: data["perf/GPU_hours_train_only"] = train_gpu_hours
                if data: wandb.log(data)
            except Exception:
                pass
