import os
import json
import gc
import torch
import wandb
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate


def main(args):
    seed = args.seed
    now = datetime.now()
    now_time = now.strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(
        project=f"{args.project}",
        name=f"{args.dataset}_{args.model_name}_numtoken{args.num_graph_token}_queryaware{args.query_aware}_seed{seed}_time{now_time}",
        config=args
    )
    seed_everything(seed=args.seed)
    print(args)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()
    train_dataset = [dataset[i] for i in idx_split['train']]
    val_dataset = [dataset[i] for i in idx_split['val']]
    test_dataset = [dataset[i] for i in idx_split['test']]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph_type=dataset.graph_type, args=args, init_prompt=dataset.prompt)
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': params, 'lr': args.lr, 'weight_decay': args.wd}], betas=(0.9, 0.95))

    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(batch, args)
            loss.backward()
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)
                optimizer.step()
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({'Lr': lr, 'Accum Loss': accum_loss / args.grad_steps})
                accum_loss = 0.
            epoch_loss += loss.item()
            accum_loss += loss.item()
            progress_bar.update(1)

        print(f"Epoch: {epoch} | Train Loss: {epoch_loss / len(train_loader)}")
        wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

        val_loss = 0.
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = model(batch, args)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
            wandb.log({'Val Loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
        path = (
            f'{args.output_dir}/{args.dataset}/{epoch}'
            f'_model_name_{args.model_name}'
            f'_llm_model_name_{args.llm_model_name}'
            f'_llm_frozen_{args.llm_frozen}'
            f'_max_txt_len_{args.max_txt_len}'
            f'_max_new_tokens_{args.max_new_tokens}'
            f'_gnn_model_name_{args.gnn_model_name}'
            f'_patience_{args.patience}'
            f'_num_epochs_{args.num_epochs}'
            f'_numgt{args.num_graph_token}'
            f'_queryaware{args.query_aware}'
            f'_pooling{args.pooling}'
            f'_seed{seed}_{now_time}.csv'
        )
        print(f'path: {path}')

        model.eval()
        progress_bar_test = tqdm(range(len(test_loader)))
        with open(path, "w") as f:
            for step, batch in enumerate(test_loader):
                with torch.no_grad():
                    output = model.inference(batch, args)
                    df = pd.DataFrame(output)
                    for _, row in df.iterrows():
                        f.write(json.dumps(dict(row)) + "\n")
                progress_bar_test.update(1)

        acc = eval_funcs[args.dataset](path)
        print(f'Test Acc {acc}')
        wandb.log({'Test Acc': acc})


if __name__ == "__main__":
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
