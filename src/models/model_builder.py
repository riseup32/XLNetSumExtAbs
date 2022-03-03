
import copy

import torch
import torch.nn as nn
from pytorch_transformers import XLNetModel, XLNetConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)


    optim.set_parameters(list(model.named_parameters()))


    return optim


def build_optim_xlnet(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_xlnet, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_xlnet)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('xlnet.model')]
    optim.set_parameters(params)


    return optim


def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('xlnet.model')]
    optim.set_parameters(params)


    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


class Xlnet(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Xlnet, self).__init__()
        if(large):
            self.model = XLNetModel.from_pretrained('xlnet-large-cased', cache_dir=temp_dir)
        else:
            self.model = XLNetModel.from_pretrained('xlnet-base-cased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.xlnet = Xlnet(args.large, args.temp_dir, args.finetune_xlnet)

        self.ext_layer = ExtTransformerEncoder(self.xlnet.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        if (args.encoder == 'baseline'):
            xlnet_config = XLNetConfig(self.xlnet.model.config.vocab_size, d_model=args.ext_hidden_size,
                                       n_layer=args.ext_layers, n_head=args.ext_heads,
                                       d_inner=args.ext_ff_size)
            self.xlnet.model = XLNetModel(xlnet_config)
            self.ext_layer = Classifier(self.xlnet.model.config.hidden_size)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.xlnet.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.xlnet.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.xlnet.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.xlnet.model.embeddings.position_embeddings = my_pos_embeddings


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.xlnet(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, xlnet_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.xlnet = Xlnet(args.large, args.temp_dir, args.finetune_xlnet)

        if xlnet_from_extractive is not None:
            self.xlnet.model.load_state_dict(
                dict([(n[12:], p) for n, p in xlnet_from_extractive.items() if n.startswith('xlnet.model')]), strict=True)

        if (args.encoder == 'baseline'):
            xlnet_config = XLNetConfig(self.xlnet.model.config.vocab_size, d_model=args.enc_hidden_size,
                                     n_layer=args.enc_layers, n_head=8,
                                     d_inner=args.enc_ff_size,
                                     dropout=args.enc_dropout,
                                     summary_last_dropout=args.enc_dropout)
            self.xlnet.model = XLNetModel(xlnet_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.xlent.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.xlnet.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.xlnet.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.xlnet.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.xlnet.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.xlnet.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            # tgt_embeddings.weight = copy.deepcopy(self.xlnet.model.embeddings.word_embeddings.weight)
            tgt_embeddings.weight = copy.deepcopy(self.xlnet.model.word_embedding.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_xlnet_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.xlnet.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.xlnet.model.word_embedding.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.xlnet(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
