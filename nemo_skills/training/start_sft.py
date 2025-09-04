# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset as gpt_sft_chat_dataset
import torch.multiprocessing as mp
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import get_prompt_template_example
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.data.nlp.builders import build_dataloader, build_sft_dataset
from nemo_aligner.models.nlp.gpt.gpt_sft_model import GPTSFTModel
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_peft,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo
from omegaconf.omegaconf import OmegaConf, open_dict

"""Script to start SFT training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path=".", config_name="sft_config")
def main(cfg) -> None:
    cfg.model = load_and_override_model_config(cfg.model.restore_from_path, cfg.model)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # updating a few parameters based on num_checkpoints_to_save arg
    if cfg.trainer.sft.get("num_checkpoints_to_save", None) is not None:
        # if steps are > 0 using that
        if cfg.trainer.sft.max_steps > 0:
            num_steps = cfg.trainer.sft.max_steps
        else:
            # counting the steps per epoch
            # using wc -l since sft file might be large and we want to use optimized util
            data_size = int(os.popen(f'wc -l "{cfg.model.data.train_ds.file_path}"').read().split()[0])
            assert cfg.trainer.sft.max_epochs > 0
            num_steps = (data_size * cfg.trainer.sft.max_epochs) // cfg.model.data.train_ds.global_batch_size
        num_checkpoints = cfg.trainer.sft.num_checkpoints_to_save
        with open_dict(cfg):
            cfg.trainer.sft.max_epochs = 10000  # always using steps internally
            # rounding steps to make sure last checkpoint is not repeated
            cfg.trainer.sft.max_steps = (num_steps // num_checkpoints) * num_checkpoints
            cfg.trainer.sft.save_interval = num_steps // num_checkpoints
            cfg.trainer.sft.val_check_interval = num_steps // num_checkpoints
        logging.info(
            (
                "Adjusting config parameters in the following way:\n"
                "max_epochs: %d\nmax_steps: %d\nsave_interval: %d\nval_check_interval: %d"
            ),
            cfg.trainer.sft.max_epochs,
            cfg.trainer.sft.max_steps,
            cfg.trainer.sft.save_interval,
            cfg.trainer.sft.val_check_interval,
        )

    trainer = resolve_and_create_trainer(cfg, "sft")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    ptl_model = load_from_nemo(
        GPTSFTModel,
        cfg,
        trainer,
        strict=True,
        restore_path=cfg.model.restore_from_path,
        return_updated_cfg=False,
    )

    init_peft(ptl_model, cfg.model)

    # # ============================================================================
    # # CUSTOM TOKENS HANDLING
    # # ============================================================================
    # logging.info("=" * 80)
    # logging.info("🔧 CHECKING AND ADDING CUSTOM TOKENS")
    # logging.info("=" * 80)
    
    # custom_tokens = ["<tool_call>", "</tool_call>", "<locations>", "</locations>"]
    # tokenizer = ptl_model.tokenizer
    # original_vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else getattr(tokenizer, 'vocab_size', 'unknown')
    
    # logging.info(f"📝 Original tokenizer vocab size: {original_vocab_size}")
    # logging.info(f"🔍 Checking for custom tokens: {custom_tokens}")
    # logging.info("-" * 60)
    
    # # Check which tokens exist and which need to be added
    # existing_tokens = []
    # tokens_to_add = []
    
    # for token in custom_tokens:
    #     try:
    #         token_exists = False
    #         token_id = None
            
    #         # Try different methods to check token existence
    #         if hasattr(tokenizer, 'token_to_id'):
    #             token_id = tokenizer.token_to_id(token)
    #             token_exists = token_id is not None
    #         elif hasattr(tokenizer, 'text_to_ids'):
    #             # For NeMo tokenizers, check if token is split
    #             ids = tokenizer.text_to_ids(token)
    #             token_exists = len(ids) == 1
    #             if token_exists:
    #                 token_id = ids[0]
            
    #         if token_exists:
    #             existing_tokens.append((token, token_id))
    #             logging.info(f"  ✅ Found: '{token}' (ID: {token_id})")
    #         else:
    #             tokens_to_add.append(token)
    #             logging.info(f"  ❌ Missing: '{token}' - will be added")
                
    #     except Exception as e:
    #         tokens_to_add.append(token)
    #         logging.info(f"  ❌ Missing: '{token}' - will be added (error checking: {e})")
    
    # logging.info("-" * 60)
    # logging.info(f"📊 Summary: {len(existing_tokens)} existing, {len(tokens_to_add)} to add")
    
    # if tokens_to_add:
    #     logging.info("=" * 40)
    #     logging.info("🔨 ADDING MISSING TOKENS")
    #     logging.info("=" * 40)
    #     logging.info(f"➕ Adding {len(tokens_to_add)} custom tokens: {tokens_to_add}")
        
    #     # Add tokens to tokenizer
    #     add_success = False
    #     if hasattr(tokenizer, 'add_special_tokens'):
    #         # For HF-style tokenizers
    #         num_added = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    #         add_success = True
    #         logging.info(f"  ✅ Added {num_added} tokens using add_special_tokens()")
    #     elif hasattr(tokenizer, 'add_tokens'):
    #         # For some tokenizer types
    #         num_added = tokenizer.add_tokens(tokens_to_add)
    #         add_success = True
    #         logging.info(f"  ✅ Added {num_added} tokens using add_tokens()")
    #     else:
    #         logging.error("  ❌ Could not add custom tokens - tokenizer type not supported")
    #         logging.error(f"  📝 Tokenizer type: {type(tokenizer)}")
        
    #     if add_success:
    #         new_vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else getattr(tokenizer, 'vocab_size', 'unknown')
    #         logging.info(f"  📝 New tokenizer vocab size: {new_vocab_size}")
            
    #         # Resize model embeddings to match new vocab size
    #         logging.info("🔧 Resizing model embeddings...")
    #         resize_success = False
            
    #         # Try standard HuggingFace-style methods first
    #         if hasattr(ptl_model, 'resize_token_embeddings'):
    #             ptl_model.resize_token_embeddings(new_vocab_size)
    #             resize_success = True
    #             logging.info(f"  ✅ Resized model embeddings to {new_vocab_size} using ptl_model.resize_token_embeddings()")
    #         elif hasattr(ptl_model.model, 'resize_token_embeddings'):
    #             ptl_model.model.resize_token_embeddings(new_vocab_size)
    #             resize_success = True
    #             logging.info(f"  ✅ Resized model embeddings to {new_vocab_size} using ptl_model.model.resize_token_embeddings()")
    #         else:
    #             # Manual embedding resize for NeMo models
    #             logging.info("  🔧 Attempting manual embedding resize for NeMo model...")
    #             try:
    #                 import torch
    #                 import torch.nn as nn
                    
    #                 # Find the embedding layer in the model
    #                 embedding_layer = None
    #                 embedding_path = None
                    
    #                 # Common paths for NeMo model embeddings
    #                 possible_paths = [
    #                     'model.embedding.word_embeddings',
    #                     'model.language_model.embedding.word_embeddings', 
    #                     'model.module.embedding.word_embeddings',
    #                     'model.module.language_model.embedding.word_embeddings',
    #                     'model.model.embedding.word_embeddings',
    #                     'model.model.language_model.embedding.word_embeddings',
    #                     'model.decoder.embeddings.word_embeddings',
    #                     'model.decoder.embed_tokens',
    #                     'model.embed_tokens',
    #                     'embedding.word_embeddings',
    #                     'language_model.embedding.word_embeddings'
    #                 ]
                    
    #                 # Also try to inspect the model structure dynamically
    #                 logging.info(f"  🔍 Inspecting model structure...")
    #                 try:
    #                     # First, let's examine the top-level model structure
    #                     logging.info(f"  📝 Model type: {type(ptl_model)}")
    #                     top_level_attrs = [attr for attr in dir(ptl_model) if not attr.startswith('_') and not callable(getattr(ptl_model, attr, None))]
    #                     logging.info(f"  📝 Top-level attributes: {top_level_attrs[:10]}...")  # Show first 10
                        
    #                     def find_embedding_layers(obj, path="", max_depth=4):
    #                         """Recursively find embedding layers in the model."""
    #                         if max_depth <= 0:
    #                             return []
                            
    #                         # Skip problematic attributes that cause warnings
    #                         skip_attrs = {
    #                             'H', 'T', 'mH', 'mT', 'real', 'imag', 'shape', 'data', 'grad', 'device', 'dtype',
    #                             'requires_grad', 'is_leaf', 'grad_fn', 'names', 'ndim', 'size', 'stride'
    #                         }
                            
    #                         found_embeddings = []
    #                         if hasattr(obj, '__dict__') or hasattr(obj, '__class__'):
    #                             for attr_name in dir(obj):
    #                                 if (attr_name.startswith('_') or 
    #                                     attr_name in skip_attrs or
    #                                     attr_name.startswith('is_') or
    #                                     attr_name.endswith('_')):
    #                                     continue
                                    
    #                                 try:
    #                                     attr_value = getattr(obj, attr_name)
    #                                     current_path = f"{path}.{attr_name}" if path else attr_name
                                        
    #                                     # Check if this is an embedding layer
    #                                     if isinstance(attr_value, nn.Embedding):
    #                                         found_embeddings.append((current_path, attr_value))
    #                                         logging.info(f"    🎯 Found embedding: {current_path} ({attr_value.num_embeddings} x {attr_value.embedding_dim})")
    #                                     # Continue searching in modules and non-callable attributes
    #                                     elif (hasattr(attr_value, '__dict__') and 
    #                                           not callable(attr_value) and 
    #                                           not isinstance(attr_value, (torch.Tensor, str, int, float, bool, list, dict, tuple))):
    #                                         found_embeddings.extend(find_embedding_layers(attr_value, current_path, max_depth-1))
    #                                 except Exception as e:
    #                                     # Skip attributes that cause issues
    #                                     continue
                                        
    #                         return found_embeddings
                        
    #                     found_embeddings = find_embedding_layers(ptl_model)
    #                     if found_embeddings:
    #                         logging.info(f"  📋 Found {len(found_embeddings)} embedding layers:")
    #                         for path, emb in found_embeddings:
    #                             logging.info(f"    - {path}: {emb.num_embeddings} x {emb.embedding_dim}")
    #                         # Add the found paths to our search list
    #                         for path, emb in found_embeddings:
    #                             if path not in possible_paths:
    #                                 possible_paths.append(path)
    #                     else:
    #                         logging.info(f"  ❌ No embedding layers found in dynamic search")
    #                         # Try to show model structure for debugging
    #                         if hasattr(ptl_model, 'model'):
    #                             logging.info(f"  📝 ptl_model.model type: {type(ptl_model.model)}")
    #                             if hasattr(ptl_model.model, '__dict__'):
    #                                 model_attrs = [attr for attr in dir(ptl_model.model) if not attr.startswith('_')][:10]
    #                                 logging.info(f"  📝 ptl_model.model attributes: {model_attrs}")
    #                 except Exception as e:
    #                     logging.info(f"  ⚠️  Dynamic search failed: {e}")
                    
    #                 for path in possible_paths:
    #                     try:
    #                         embedding_layer = ptl_model
    #                         for attr in path.split('.'):
    #                             embedding_layer = getattr(embedding_layer, attr)
    #                         if isinstance(embedding_layer, nn.Embedding):
    #                             embedding_path = path
    #                             break
    #                     except AttributeError:
    #                         continue
                    
    #                 if embedding_layer is not None and isinstance(embedding_layer, nn.Embedding):
    #                     old_vocab_size = embedding_layer.num_embeddings
    #                     embedding_dim = embedding_layer.embedding_dim
                        
    #                     logging.info(f"  📝 Found embedding layer at: {embedding_path}")
    #                     logging.info(f"  📝 Current size: {old_vocab_size} → Target size: {new_vocab_size}")
    #                     logging.info(f"  📝 Embedding dimension: {embedding_dim}")
                        
    #                     if new_vocab_size > old_vocab_size:
    #                         # Create new embedding layer with expanded size
    #                         new_embedding = nn.Embedding(new_vocab_size, embedding_dim)
                            
    #                         # Copy existing weights
    #                         with torch.no_grad():
    #                             new_embedding.weight[:old_vocab_size] = embedding_layer.weight.data
    #                             # Initialize new token embeddings with small random values
    #                             nn.init.normal_(new_embedding.weight[old_vocab_size:], mean=0.0, std=0.02)
                            
    #                         # Replace the embedding layer
    #                         parent_obj = ptl_model
    #                         path_parts = embedding_path.split('.')
    #                         for attr in path_parts[:-1]:
    #                             parent_obj = getattr(parent_obj, attr)
    #                         setattr(parent_obj, path_parts[-1], new_embedding)
                            
    #                         logging.info(f"  ✅ Successfully resized embeddings from {old_vocab_size} to {new_vocab_size}")
    #                         logging.info(f"  📝 New tokens initialized with random values (std=0.02)")
    #                         resize_success = True
    #                     else:
    #                         logging.info(f"  ℹ️  No resize needed (current: {old_vocab_size}, target: {new_vocab_size})")
    #                         resize_success = True
    #                 else:
    #                     logging.error("  ❌ Could not find embedding layer in model")
    #                     logging.error(f"  📝 Searched paths: {possible_paths}")
                        
    #             except Exception as e:
    #                 logging.error(f"  ❌ Manual embedding resize failed: {e}")
    #                 logging.error(f"  📝 Exception type: {type(e).__name__}")
            
    #         if resize_success:
    #             logging.info("  🎉 Custom tokens successfully added and embeddings resized!")
    #         else:
    #             logging.warning("  ⚠️  Custom tokens added but embedding resize failed - this may cause issues")
    #             logging.warning("  💡 Training may still work but new tokens will have undefined behavior")
    # else:
    #     logging.info("✅ All custom tokens already present in tokenizer - no action needed")
    
    # logging.info("=" * 80)
    # logging.info("🏁 CUSTOM TOKENS HANDLING COMPLETE")
    # logging.info("=" * 80)

    with open_dict(cfg):
        # overwrite the model config with the config from the checkpoint
        cfg.model.encoder_seq_length = ptl_model.cfg.encoder_seq_length

    # monkey-patching the system token to allow training with llama format
    # TODO: remove when this is properly supported in nemo
    if cfg.model.data.chat:
        # not using default to avoid accidental errors by mistyping or misplacing this value in the config
        gpt_sft_chat_dataset.SYSTEM_TOKEN = cfg.model.data["system_token"]

    # pull values from checkpoint
    trainer_restore_path = trainer.ckpt_path

    # TODO: log this restore path
    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples = custom_trainer_state_dict["consumed_samples"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    train_data_cfg = cfg.model.data.train_ds
    val_data_cfg = cfg.model.data.validation_ds

    if cfg.model.data.get("sample", False):
        # if it is negative, num_samples is None
        if cfg.trainer.sft.max_steps < 0:
            num_samples = None
        else:
            num_samples = cfg.trainer.sft.max_steps * train_data_cfg.global_batch_size
    else:
        num_samples = None
    train_ds = build_sft_dataset(
        train_data_cfg,
        ptl_model.tokenizer,
        num_samples,
        answer_only_loss=True,
        is_chat=cfg.model.data.chat,
        special_tokens=cfg.model.data.chat_prompt_tokens,
    )
    if cfg.model.data.get("sample", False):
        num_samples = cfg.trainer.sft.limit_val_batches * val_data_cfg.global_batch_size
    else:
        num_samples = None
    validation_ds = build_sft_dataset(
        val_data_cfg,
        ptl_model.tokenizer,
        num_samples,
        answer_only_loss=True,
        is_chat=cfg.model.data.chat,
        special_tokens=cfg.model.data.chat_prompt_tokens,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=consumed_samples,
        mbs=train_data_cfg.micro_batch_size,
        gbs=train_data_cfg.global_batch_size,
        collate_fn=train_ds.collate_fn,
        drop_last=train_data_cfg.drop_last,
        pad_samples_to_global_batch_size=not train_data_cfg.drop_last,
        load_gbs=True,
    )

    val_dataloader = build_dataloader(
        cfg=cfg,
        dataset=validation_ds,
        consumed_samples=0,
        mbs=val_data_cfg.micro_batch_size,
        gbs=val_data_cfg.global_batch_size,
        collate_fn=validation_ds.collate_fn,
        drop_last=val_data_cfg.drop_last,
        pad_samples_to_global_batch_size=not val_data_cfg.drop_last,
        load_gbs=True,
        use_random_sampler=False,
    )

    init_using_ptl(trainer, ptl_model, train_dataloader, train_ds)
    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))
    timer = Timer(cfg.exp_manager.get("max_time_per_run") if cfg.exp_manager else None)

    sft_trainer = SupervisedTrainer(
        cfg=cfg.trainer.sft,
        model=ptl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
    )

    if custom_trainer_state_dict is not None:
        sft_trainer.load_state_dict(custom_trainer_state_dict)

    sft_trainer.fit()


if __name__ == "__main__":
    main()
