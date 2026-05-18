"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import bleu

from model import Transformer, make_src_mask, make_tgt_mask

# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence_level = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # Calculate log probabilities along the vocab dimension
        log_probabilities = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            # Distribute smoothing mass across all classes
            target_dist = torch.full_like(log_probabilities, self.smoothing / (self.vocab_size - 1))
            # Assign the confidence mass to the correct target index
            target_dist.scatter_(1, target.unsqueeze(1), self.confidence_level)
            # Ensure padding tokens receive zero probability mass
            target_dist[target == self.pad_idx] = 0.0

        # Compute negative log likelihood
        nll_loss = torch.sum(-target_dist * log_probabilities, dim=-1)

        # Mask out padding tokens from the final loss calculation
        valid_tokens_mask = (target != self.pad_idx)
        return nll_loss[valid_tokens_mask].mean()


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    # Toggle train/eval modes safely
    model.train() if is_train else model.eval()

    cumulative_loss = 0.0
    active_tokens = 0

    for src_batch, tgt_batch in data_iter:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        # Shift targets for autoregressive training
        tgt_in = tgt_batch[:, :-1]
        tgt_expected = tgt_batch[:, 1:]

        # Create padding/future masks
        mask_src = make_src_mask(src_batch)
        mask_tgt = make_tgt_mask(tgt_in)

        # Forward pass
        predictions = model(src_batch, tgt_in, mask_src, mask_tgt)

        # Flatten tensors for cross-entropy style loss
        vocab_dim = predictions.size(-1)
        predictions_flat = predictions.reshape(-1, vocab_dim)
        expected_flat = tgt_expected.reshape(-1)

        loss_val = loss_fn(predictions_flat, expected_flat)

        if is_train:
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

        # Update metrics tracking
        valid_elements = (expected_flat != loss_fn.pad_idx).sum().item()
        cumulative_loss += loss_val.item() * valid_elements
        active_tokens += valid_elements

    return cumulative_loss / active_tokens


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    model.eval()
    src, src_mask = src.to(device), src_mask.to(device)

    # Encode the source sequence once
    encoder_memory = model.encode(src, src_mask)

    # Initialize the decoder sequence with the <sos> token
    current_seq = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(current_seq)
        
        # Decode current sequence
        dec_output = model.decode(encoder_memory, src_mask, current_seq, tgt_mask)
        
        # Get the highest probability token for the last position
        next_tok = dec_output[:, -1, :].argmax(dim=-1).item()
        
        # Append to current sequence
        next_tok_tensor = torch.tensor([[next_tok]], dtype=src.dtype, device=device)
        current_seq = torch.cat([current_seq, next_tok_tensor], dim=1)

        if next_tok == end_symbol:
            break

    return current_seq


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """

    model.eval()

    hypotheses = []
    ground_truths = []

    # Safe vocabulary reverse mapping
    if isinstance(tgt_vocab, dict):
        idx_to_str = {idx: token for token, idx in tgt_vocab.items()}
    else:
        idx_to_str = tgt_vocab.itos

    def resolve_token(idx):
        return idx_to_str.get(idx, "<unk>")

    with torch.no_grad():
        for src_batch, tgt_batch in test_dataloader:
            src_batch = src_batch.to(device)
            
            for i in range(src_batch.size(0)):
                single_src = src_batch[i].unsqueeze(0)
                single_tgt = tgt_batch[i]

                # Generate prediction
                predicted_indices = greedy_decode(
                    model=model,
                    src=single_src,
                    src_mask=make_src_mask(single_src),
                    max_len=max_len,
                    start_symbol=tgt_vocab["<sos>"] if isinstance(tgt_vocab, dict) else tgt_vocab.stoi["<sos>"],
                    end_symbol=tgt_vocab["<eos>"] if isinstance(tgt_vocab, dict) else tgt_vocab.stoi["<eos>"],
                    device=device,
                ).squeeze(0).tolist()

                # Convert both lists to strings
                pred_str_list = [resolve_token(idx) for idx in predicted_indices]
                tgt_str_list = [resolve_token(idx) for idx in single_tgt.tolist()]

                # Filter out special structural tokens
                ignored_tokens = {"<sos>", "<eos>", "<pad>"}
                cleaned_pred = [t for t in pred_str_list if t not in ignored_tokens]
                cleaned_tgt = [t for t in tgt_str_list if t not in ignored_tokens]

                hypotheses.append(" ".join(cleaned_pred))
                ground_truths.append([" ".join(cleaned_tgt)])

    try:
        evaluation = bleu.corpus_bleu(ground_truths, hypotheses)
    except AttributeError:
        # Fallback in case their library expects lists differently
        evaluation = bleu.list_bleu(ground_truths, hypotheses)
        
    return evaluation * 100.0


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # Store architectural configurations securely for the autograder
    arch_config = {
        "src_vocab_size": model.src_embed.num_embeddings,
        "tgt_vocab_size": model.tgt_embed.num_embeddings,
        "d_model": model.src_embed.embedding_dim,
        "N": len(model.encoder.layers_stack),
        "num_heads": model.encoder.layers_stack[0].attn_layer.num_heads,
        "d_ff": model.encoder.layers_stack[0].ffn_layer.fc1.out_features,
        "dropout": model.encoder.layers_stack[0].drop.p,
    }

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "model_config": arch_config,
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    saved_state = torch.load(path, map_location="cpu")
    model.load_state_dict(saved_state["model_state_dict"])

    if optimizer and saved_state.get("optimizer_state_dict"):
        optimizer.load_state_dict(saved_state["optimizer_state_dict"])

    if scheduler and saved_state.get("scheduler_state_dict"):
        scheduler.load_state_dict(saved_state["scheduler_state_dict"])

    return saved_state["epoch"]


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    import wandb
    from dataset import Multi30kDataset
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    import torch.optim as optim
    from lr_scheduler import NoamScheduler

    active_device = "cuda" if torch.cuda.is_available() else "cpu"

    # W&B Configuration Matrix
    experiment_config = {
        "d_model": 512,
        "N": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "batch_size": 64,
        "num_epochs": 10,
        "warmup_steps": 4000,
        "base_lr": 1.0,
    }

    wandb.init(project="da6401-a3", config=experiment_config)

    # Initialize Datasets
    ds_train = Multi30kDataset("train")
    ds_val   = Multi30kDataset("validation")
    ds_test  = Multi30kDataset("test")

    # Build and share Vocabs
    ds_train.build_vocab()
    ds_val.src_vocab, ds_val.tgt_vocab = ds_train.src_vocab, ds_train.tgt_vocab
    ds_test.src_vocab, ds_test.tgt_vocab = ds_train.src_vocab, ds_train.tgt_vocab

    data_train = ds_train.process_data()
    data_val   = ds_val.process_data()
    data_test  = ds_test.process_data()

    PAD_TOKEN_ID = ds_train.src_vocab["<pad>"]

    # DataLoader Collate Helper
    def custom_collate(batch):
        src_items, tgt_items = zip(*batch)
        src_tensors = [torch.tensor(s) for s in src_items]
        tgt_tensors = [torch.tensor(t) for t in tgt_items]
        
        src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=PAD_TOKEN_ID)
        tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=PAD_TOKEN_ID)
        return src_padded, tgt_padded

    loader_train = DataLoader(data_train, batch_size=experiment_config["batch_size"], shuffle=True, collate_fn=custom_collate)
    loader_val   = DataLoader(data_val, batch_size=experiment_config["batch_size"], shuffle=False, collate_fn=custom_collate)
    loader_test  = DataLoader(data_test, batch_size=1, shuffle=False, collate_fn=custom_collate)

    # Model Setup
    model_instance = Transformer(
        src_vocab_size=len(ds_train.src_vocab),
        tgt_vocab_size=len(ds_train.tgt_vocab),
        d_model=experiment_config["d_model"],
        N=experiment_config["N"],
        num_heads=experiment_config["num_heads"],
        d_ff=experiment_config["d_ff"],
        dropout=experiment_config["dropout"],
    ).to(active_device)

    # Optimizer & Scheduler
    opt = optim.Adam(model_instance.parameters(), lr=experiment_config["base_lr"], betas=(0.9, 0.98), eps=1e-9)
    lr_sch = NoamScheduler(opt, d_model=experiment_config["d_model"], warmup_steps=experiment_config["warmup_steps"])
    
    # Loss Object
    criterion = LabelSmoothingLoss(vocab_size=len(ds_train.tgt_vocab), pad_idx=PAD_TOKEN_ID, smoothing=0.1)

    # Main Training Loop
    for ep in range(experiment_config["num_epochs"]):
        
        loss_t = run_epoch(loader_train, model_instance, criterion, opt, lr_sch, ep, is_train=True, device=active_device)
        loss_v = run_epoch(loader_val, model_instance, criterion, None, None, ep, is_train=False, device=active_device)

        print(f"Epoch {ep} - Train Loss: {loss_t:.4f} | Validation Loss: {loss_v:.4f}")

        save_checkpoint(model_instance, opt, lr_sch, ep)

        wandb.log({
            "epoch": ep,
            "train_loss": loss_t,
            "val_loss": loss_v
        })

    # Final BLEU Eval
    test_bleu_score = evaluate_bleu(model_instance, loader_test, ds_train.tgt_vocab, device=active_device)
    wandb.log({"test_bleu": test_bleu_score})
    print(f"Experiment Finished. Final Test BLEU: {test_bleu_score:.2f}")


if __name__ == "__main__":
    run_training_experiment()
