import logging
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss

# Optional: AMP (Automatic Mixed Precision) for faster training
try:
    from torch.amp import autocast, GradScaler
    HAS_AMP = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        HAS_AMP = True
    except ImportError:
        HAS_AMP = False

def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)
    
def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    # Optimization settings
    accumulation_steps = getattr(args, 'accumulation_steps', 0) or 1
    use_amp = getattr(args, 'amp_opt_level', 'O0') != 'O0' and HAS_AMP
    num_workers = getattr(args, 'num_workers', 2)
    use_checkpoint = getattr(args, 'use_checkpoint', False)
    
    # Print optimization status
    print("\n" + "="*80)
    print("TRAINING CONFIG:")
    print(f"  Batch Size: {batch_size}, Num Workers: {num_workers}")
    print(f"  AMP: {('O'+getattr(args, 'amp_opt_level', '0')[-1]) if use_amp else 'Disabled'}")
    print(f"  Gradient Checkpointing: {'Enabled' if use_checkpoint else 'Disabled'}")
    print(f"  Gradient Accumulation Steps: {accumulation_steps}")
    print("="*80 + "\n")
    
    # Dataset with optimized augmentation
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",
                             transform=transforms.Compose(
                                 [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    # Optimized DataLoader settings
    prefetch_factor = 3 if num_workers > 0 else None
    persistent_workers = num_workers > 0
    
    train_loader = DataLoader(
        db_train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        db_train, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    # AMP setup
    if use_amp:
        try:
            scaler = GradScaler('cuda')
        except TypeError:
            # Fallback for older PyTorch versions
            scaler = GradScaler()
    else:
        scaler = None
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10
    epoch_start_time = time.time()
    
    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        
        optimizer.zero_grad()  # Zero at epoch start for accumulation
        
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", 
                                           total=len(train_loader), leave=False, ncols=70):
            # Non-blocking async GPU transfer
            image_batch = sampled_batch['image'].cuda(non_blocking=True)
            label_batch = sampled_batch['label'].cuda(non_blocking=True)
            
            # Forward pass with optional AMP
            if use_amp:
                try:
                    with autocast('cuda'):
                        outputs = model(image_batch)
                        loss_ce = ce_loss(outputs, label_batch[:].long())
                        loss_dice = dice_loss(outputs, label_batch, softmax=True)
                        loss = 0.4 * loss_ce + 0.6 * loss_dice
                        loss = loss / accumulation_steps
                except TypeError:
                    # Fallback for older PyTorch versions
                    with autocast():
                        outputs = model(image_batch)
                        loss_ce = ce_loss(outputs, label_batch[:].long())
                        loss_dice = dice_loss(outputs, label_batch, softmax=True)
                        loss = 0.4 * loss_ce + 0.6 * loss_dice
                        loss = loss / accumulation_steps
                
                # Backward with scaling
                scaler.scale(loss).backward()
            else:
                outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.4 * loss_ce + 0.6 * loss_dice
                loss = loss / accumulation_steps
                loss.backward()
            
            # Gradient accumulation step
            if (i_batch + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                optimizer.zero_grad()
            
            # Learning rate scheduling (poly decay)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item() * accumulation_steps, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce.item() * accumulation_steps, iter_num)

            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            
            if iter_num % 20 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min() + 1e-5)
                writer.add_image('train/Image', image, iter_num)
                outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[0, ...].float() * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs.float(), iter_num)
        
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        
        epoch_time = time.time() - epoch_start_time
        samples_per_sec = int(len(db_train) / epoch_time)
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f [%d samples/sec]' % (
            epoch_num, batch_loss, batch_ce_loss, batch_dice_loss, samples_per_sec))
        
        writer.add_scalar('epoch/train_loss', batch_loss, epoch_num)
        writer.add_scalar('epoch/samples_per_sec', samples_per_sec, epoch_num)
        
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            batch_dice_loss = 0
            batch_ce_loss = 0
            val_count = 0
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                                   total=len(val_loader), leave=False, ncols=70):
                    image_batch = sampled_batch['image'].cuda(non_blocking=True)
                    label_batch = sampled_batch['label'].cuda(non_blocking=True)
                    
                    if use_amp:
                        try:
                            with autocast('cuda'):
                                outputs = model(image_batch)
                                loss_ce = ce_loss(outputs, label_batch[:].long())
                                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                        except TypeError:
                            with autocast():
                                outputs = model(image_batch)
                                loss_ce = ce_loss(outputs, label_batch[:].long())
                                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    else:
                        outputs = model(image_batch)
                        loss_ce = ce_loss(outputs, label_batch[:].long())
                        loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    
                    batch_dice_loss += loss_dice.item()
                    batch_ce_loss += loss_ce.item()
                    val_count += 1

                # Only compute metrics if there's validation data
                if val_count > 0:
                    batch_ce_loss /= val_count
                    batch_dice_loss /= val_count
                    batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
                    logging.info('Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
                        epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
                    
                    writer.add_scalar('epoch/val_loss', batch_loss, epoch_num)
                    
                    if batch_loss < best_loss:
                        save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                        torch.save(model.state_dict(), save_mode_path)
                        best_loss = batch_loss
                    else:
                        save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                        torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                else:
                    # No validation data, just save last model
                    save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("No validation data available, saved checkpoint to {}".format(save_mode_path))
        
        epoch_start_time = time.time()

    writer.close()
    return "Training Finished!"
