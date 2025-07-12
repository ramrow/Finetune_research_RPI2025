"""
from torch.utils.data import( 
    DataLoader,
    IterableDataset,
    
    )
from transformers import (
    DataCollatorForSeq2Seq,
    get_scheduler,
    
    )
import torch
import logging


def prepare_for_multi_train(model, tokenizer, train_data, valid_data, accelerator, batch_size=8):
    # Dataloaders
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="max_length", max_length=model.config.n_positions)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer)

    # Accelerate them
    train_dataloader = accelerator.prepare(train_dataloader)
    valid_dataloader = accelerator.prepare(valid_dataloader)
    # log_dataloading(train_dataloader, accelerator)
    model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler
    )
    return train_dataloader, valid_dataloader, model, optimizer, lr_scheduler

def load_model_tok_data(accelerator, config_dict):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Tokenizer
        tokenizer = tokops.get_trained_tokenizer(config_dict, making_dirs=True)
        vocab_size = len(tokenizer)

        # Data
        train_kwargs = {
            "tokenizer": tokenizer,
            "json_data_dir": config_dict["data_dir"],
            "split": "train",
            "data_format": config_dict["data_format"]
        }
        train_data = IterableDataset.from_generator(
            tokops.t5_tokked_model_inputs,
            gen_kwargs=train_kwargs
        )
        valid_kwargs = train_kwargs.copy()
        valid_kwargs["split"] = "valid"
        valid_data = IterableDataset.from_generator(
            tokops.t5_tokked_model_inputs,
            gen_kwargs=valid_kwargs
        )

        # Model
        model = get_model(config_dict, vocab_size)
    else:
        tokenizer = None
        train_data, valid_data = None, None
        model = None

    accelerator.wait_for_everyone()
    tokenizer = broadcast_object_list([tokenizer])[0]
    train_data = broadcast_object_list([train_data])[0]
    valid_data = broadcast_object_list([valid_data])[0]
    model = broadcast_object_list([model])[0]    
    logging.info(f"{accelerator.process_index}: Successfully broadcasted data, the evidence is that the type of model is {type(model)}")
    return model, tokenizer, train_data, valid_data

def record(metrics, locals, accelerator, save_file="metrics.json"):
    accelerator.wait_for_everyone()
    local_stats = torch.tensor([locals["loss_sum"], locals["corrects_sum"], locals["valid_toks"], locals["train_step"]], device=accelerator.device)
    global_loss, global_corrects_sum, global_valid_toks, global_train_step = accelerator.reduce(local_stats, reduction="sum")
    if accelerator.is_main_process:
        avg_loss = global_loss.item() / global_train_step.item()
        metrics["loss"].append(avg_loss)
        metrics["accuracy"].append(global_corrects_sum.item() / global_valid_toks.item())
        metrics["steps"].append(global_train_step.item())
        logging.info(f"Current step's ({locals["train_step"]}) average loss is {avg_loss:.4f}")
        dicts.save_as_json(metrics, save_file)
    accelerator.wait_for_everyone()
    return metrics

def validate(model, dataloader, epoch, accelerator):
    model.eval()
    process_idx = accelerator.process_index
    metrics = {"loss": [], "accuracy": [], "steps": []}
    locals = {"loss_sum": 0.0, "corrects_sum": 0, "valid_toks": 0, "train_step": 0}
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            outputs = model(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"], 
                labels=batch["labels"]
            )
            loss = outputs.loss

            # Metrics
            predictions = outputs.logits.argmax(dim=-1)
            valids_mask = batch["labels"] != -100 # tokenizer.pad_token_id
            corrects = (predictions[valids_mask] == batch["labels"][valids_mask]).sum().item()
            locals["corrects_sum"] += corrects
            locals["valid_toks"] += valids_mask.sum().item()
            locals["loss_sum"] += loss.item()
            locals["train_step"] = batch_idx + 1

            if batch_idx % 1000 == 0:
                logging.info(f"{process_idx}: valid step number {batch_idx}")
                metrics = record(metrics, locals, accelerator, save_file=f"valid_metrics{epoch}.json")
    metrics = record(metrics, locals, accelerator, save_file=f"valid_metrics{epoch}.json")

def train(model, dataloader, optimizer, lr_scheduler, epoch, config_dict, accelerator):
    model.train()
    process_idx = accelerator.process_index
    metrics = {"loss": [], "accuracy": [], "steps": []}
    locals = {"loss_sum": 0.0, "corrects_sum": 0, "valid_toks": 0, "train_step": 0}
    for batch_idx, batch in enumerate(dataloader):
        # model.forward() and loss calculation
        outputs = model(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            labels=batch["labels"]
        )
        loss = outputs.loss
        log_nan_loss(loss, batch_idx, accelerator)
        
        # Backpropagation
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        # Metrics
        predictions = outputs.logits.argmax(dim=-1)
        valids_mask = batch["labels"] != -100
        corrects = (predictions[valids_mask] == batch["labels"][valids_mask]).sum().item()
        locals["corrects_sum"] += corrects
        locals["valid_toks"] += valids_mask.sum().item()
        locals["loss_sum"] += loss.item()
        locals["train_step"] = batch_idx + 1

        # progress feedback
        if batch_idx % 1000 == 0:
            logging.info(f"{process_idx}: train step number {batch_idx}")
            metrics = record(metrics, locals, accelerator, save_file=f"train_metrics{epoch}.json")
        if batch_idx % 10000 == 0 and batch_idx > 0:
                if accelerator.is_main_process:
                    _, models_dir = save_ops.get_dirs(config_dict)
                    save_ops.save_in(accelerator.unwrap_model(model), models_dir)
        accelerator.wait_for_everyone()

    logging.info(f"{process_idx}: Total number of batches was {batch_idx + 1}")
    logging.info(f"{process_idx}: Final learning rate was: {lr_scheduler.get_last_lr()[0]}")
    _ = record(metrics, locals, accelerator, save_file=f"valid_metrics{epoch}.json")
            
def do_epochs(train_dataloader, valid_dataloader, model, optimizer, lr_scheduler, accelerator, config_dict):
    num_epochs = config_dict["num_epochs"]
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1} of {num_epochs}")
        train(model, train_dataloader, optimizer, lr_scheduler, epoch, config_dict, accelerator)
        if accelerator.is_main_process:
            _, models_dir = save_ops.get_dirs(config_dict)
            save_ops.save_in(accelerator.unwrap_model(model), models_dir)
            logging.info(f"Finished training loop. Checkpoint saved for epoch {epoch}.")
        accelerator.wait_for_everyone()
        validate(model, valid_dataloader, epoch, accelerator)
    if accelerator.is_main_process:
        logging.info("Training complete.")

def main(accelerator, config_dict):
    model, tokenizer, train_data, valid_data = load_model_tok_data(accelerator, config_dict)

    train_dataloader, valid_dataloader, model, optimizer, lr_scheduler = prepare_for_multi_train(model, tokenizer, train_data, valid_data, accelerator, batch_size=config_dict["hf_training_arguments"]["per_device_train_batch_size"])

    do_epochs(train_dataloader, valid_dataloader, model, optimizer, lr_scheduler, accelerator, config_dict)
"""