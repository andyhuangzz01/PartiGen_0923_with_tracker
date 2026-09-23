import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderWrapper(nn.Module):
    def __init__(self, original_vae):
        super().__init__()
        self.decoder = original_vae.decoder
        self.decoder_latent_proj = original_vae.decoder_latent_proj
        self.skel_embedding = original_vae.skel_embedding
        self.final_layer = original_vae.final_layer
        self.query_pos_decoder = original_vae.query_pos_decoder
        self.arch = original_vae.arch
        self.latent_std = original_vae.latent_std

        self.h_dim = original_vae.h_dim

        
    def forward(self, z, history_motion):
        # 使用固定的 nfuture 或从输入张量获取
        bs = history_motion.shape[0]

        device = next(self.parameters()).device
        z = z.to(device)
        history_motion = history_motion.to(device)

        nfuture=8
        
        # 这部分与原始 decode 方法相同
        z = self.decoder_latent_proj(z)
        # device = z.device
        queries = torch.zeros(nfuture, bs, self.h_dim).to(device)
        history_embedding = self.skel_embedding(history_motion).permute(1, 0, 2).to(device)
        
        if self.arch == "all_encoder":
            xseq = torch.cat((z, history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq)[-nfuture:]
        elif self.arch == "encoder_decoder":
            xseq = torch.cat((history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(tgt=xseq, memory=z)
            output = output[-nfuture:]
            
        output = self.final_layer(output)
        feats = output.permute(1, 0, 2)
        return feats