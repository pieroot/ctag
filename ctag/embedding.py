# Copyright (c) 2024
# Manuel Cherep <mcherep@mit.edu>
# Nikhil Singh <nsingh1@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import abc
import torch
import laion_clap
import numpy
import jax
import jax.numpy as jnp
import flax
from typing import Union, Iterable
from load_caco import load_caco
from util import load_from_list
from dataset import _dataset_process_map, DatasetConfig
from einops import rearrange


class BaseModel(abc.ABC):
    def embed_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Embeds audio into a latent space.

        Args:
            audio: Tensor of shape (batch_size, n_samples)

        Returns:
            Tensor of shape (batch_size, n_features)
        """
        raise NotImplementedError("embed_audio not implemented for this model.")

    def embed_text(self, text: Union[str, Iterable[str]]) -> torch.Tensor:
        """
        Embeds text into a latent space.

        Args:
            text: A string or list of strings

        Returns:
            Tensor of shape (batch_size, n_features)
        """
        raise NotImplementedError("embed_text not implemented for this model.")


class CLAPModel(BaseModel):
    def __init__(
        self,
        ckpt_path: str,
        enable_fusion: bool,
        amodel: str,
        tmodel: str,
        compile: bool = True,
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        self.device = device
        self.ckpt_path = ckpt_path
        self.model = laion_clap.CLAP_Module(
            enable_fusion=enable_fusion,
            amodel=amodel,
            tmodel=tmodel,
            device=device
        )
        self.compile = compile
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        self.model.model.load_state_dict(state_dict)

    def embed_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return self.model.get_audio_embedding_from_data(audio, use_tensor=True)

    def embed_text(self, text: Union[str, Iterable[str]]) -> torch.Tensor:
        return self.model.get_text_embedding(text, use_tensor=True)


class CacoModel(BaseModel):
    def __init__(
        self,
        ckpt_path: str,
        enable_fusion: bool,
        amodel: str,
        tmodel: str,
        compile: bool = True,
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        self.device = device
        self.ckpt_path = ckpt_path
        self.model_dict = load_caco(ckpt_path)
        self.caco_model = self.model_dict['caco_model']
        self.tokenizer = self.model_dict['tokenizer']
        self.params = flax.jax_utils.replicate(self.model_dict['caco_params'], devices=jax.local_devices())

        self.config = DatasetConfig(batch_size=1,
                                    patches_seq_len=5000,
                                    time_patch_size=16,
                                    freq_patch_size=16,
                                    max_text_len=100,
                                    synthetic_prob=0.8)
        self.seed = [42]
        
        def compute_audio_embedding(audio_batch):
            return self.caco_model.apply(
                {'params': self.params},
                audio_patches=audio_batch['audio_patches'],
                audio_time_inds=audio_batch['audio_time_inds'],
                audio_freq_inds=audio_batch['audio_freq_inds'],
                audio_mask=audio_batch['audio_mask'],
                deterministic=True,
                return_hidden_state=False,
                normalize=True,
                method=self.caco_model.get_audio_embedding,
            )
        
        def compute_text_embedding(text_batch):
            return self.caco_model.apply(
                {'params': self.params},
                text_input_ids=text_batch['text_input_ids'], 
                text_mask=text_batch['text_mask'],
                deterministic=True,
                return_hidden_state=False,
                normalize=True,
                method=self.caco_model.get_text_embedding,
            )
        
        self.a_apply = jax.pmap(compute_audio_embedding, axis_name='dp')
        self.t_apply = jax.pmap(compute_text_embedding, axis_name='dp')


    def embed_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        data_dict = load_from_list(audio, "", sample_rate)
        audio_batch = _dataset_process_map(data_dict, self.seed, self.config)
        embedding = self.a_apply(audio_batch)
        return torch.tensor(embedding.numpy())
    
    def embed_text(self, text: Union[str, Iterable[str]]) -> torch.Tensor:
        all_text_embeddings = []
        for class_text in text:
            tokenized = self.tokenizer("" + class_text, 
                                padding='max_length', 
                                truncation=True,
                                max_length=10, 
                                return_tensors='np')
            text_input_ids, text_mask = tokenized['input_ids'], tokenized['attention_mask']
            text_batch = dict(text_input_ids=text_input_ids,
                            text_mask=text_mask)
            text_batch = jax.tree_util.tree_map(
                lambda x: rearrange(jnp.asarray(x), '(d b) ... -> d b ...', d=jax.local_device_count()),
                text_batch
            )
            text_embedding = self.t_apply(text_batch)
            all_text_embeddings.append(text_embedding)
        all_text_embeddings = jnp.concatenate(all_text_embeddings, axis=0)
        all_text_embeddings = jnp.squeeze(all_text_embeddings, axis=1)

        return torch.from_numpy(jnp.array(all_text_embeddings))
    
    # def embed_text(self, text: Union[str, Iterable[str]]) -> torch.Tensor:
    #     embeddings = []
    #     for text_sample in text:
    #         tokenized = self.tokenizer(["" + text_sample], 
    #                             padding='max_length', 
    #                             truncation=True,
    #                             max_length=10, 
    #                             return_tensors='np')
    #         text_input_ids, text_mask = jnp.array(tokenized['input_ids']), jnp.array(tokenized['attention_mask'])
    #         text_batch = {
    #             'text_input_ids': text_input_ids,
    #             'text_mask': text_mask
    #         }
    #         embeddings.append(
    #             embedding = self.t_apply(text_batch)
    #         )
    #     return torch.stack(embeddings)