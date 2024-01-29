import torch
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path


import wandb

def train_epoch(config, model, device, data_loader, epoch, optimizer, scaler):

    model.train()
    train_iterator = tqdm(data_loader, desc=f"Training epoch {epoch}", total=len(data_loader))
    total_loss = 0
    batch_loss = 0
    total_acc = 0
    accumulation_counter = 0
    
    for batch in train_iterator:
        x,y = batch

        # Forward Pass
        with torch.cuda.amp.autocast():
            if isinstance(y, list):
                for i in range(len(y)):
                    y[i] = y[i].to(device)
            else:
                y = y.to(device)
            acc, loss = model(x.to(device), y)
        train_iterator.set_postfix(loss=f"{loss.item():.3f}")
        total_loss += loss.item()
        batch_loss += loss.item()
        total_acc += acc.item()
        
        # Tensorboard Logging
        # wandb.log({"loss/train": loss.item()})

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        accumulation_counter += 1
        if accumulation_counter == config.accumulation_steps:
            batch_loss /= config.accumulation_steps
            wandb.log({"loss/train": batch_loss})
            # Update the weights with scaled gradients
            scaler.step(optimizer)
            # Update the scale for the next iteration
            scaler.update()
            # Zero the gradients only after accumulation steps
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0
            batch_loss = 0

    total_loss /= len(data_loader)
    total_acc /= len(data_loader)
    
    # Save the model
    # optimizer.zero_grad(set_to_none=True)
    Path(config.weights_folder).mkdir(parents=True, exist_ok=True)
    file_name = config.epochs_save.format(config.model_name, epoch)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            # "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss
        },
        Path(config.weights_folder) / file_name
    )
    wandb.save(str(Path(config.weights_folder) / file_name))

    return total_loss, total_acc
    

def val_epoch(model, device, val_loader, epoch):
    model.eval()
    val_iterator = tqdm(val_loader, desc=f"Validating epoch {epoch}", total=len(val_loader))
    total_loss = 0
    total_acc = 0
    with torch.inference_mode():
        for batch in val_iterator:
            x,y = batch

            # Forward Pass
            with torch.cuda.amp.autocast():
                if isinstance(y, list):
                    for i in range(len(y)):
                        y[i] = y[i].to(device)
                else:
                    y = y.to(device)
                acc, loss = model(x.to(device), y)
            val_iterator.set_postfix(loss=f"{loss.item():.3f}")
            total_loss += loss.item()
            total_acc += acc.item()

    total_loss /= len(val_loader)
    total_acc /= len(val_loader)
    
    return total_loss, total_acc

def train(config, model, train_dl, val_dl):
    
    model = model.to(config.device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr, eps=1e-9)
    # Scaler
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, config.epochs+1):
        train_loss, train_acc = train_epoch(config, model, config.device, train_dl, epoch, optimizer, scaler)
        val_loss, val_acc = val_epoch(model, config.device, val_dl, epoch)
        wandb.log({"loss/train_epoch": train_loss, "loss/val_epoch": val_loss, "acc/train_epoch": train_acc, "acc/val_epoch": val_acc, "epoch":epoch})