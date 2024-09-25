import os
import time

import torch
import uuid
import wandb

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.eval import eval_single_dataset
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.utils import LabelSmoothing, cosine_lr


def finetune(rank, args, group):
    setup_ddp(rank, args.world_size, port=args.port)

    run = wandb.init(config=vars(args),
                        project=f"{args.model}_{args.train_dataset}_{args.finetuning_mode}_orth_strict",
                        entity='###',
                        name=f"process_{rank}",
                        group=group, 
                        )

    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    print(args.train_datasets_to_orth)

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    # Check if checkpoints already exist
    ft_path = (
        os.path.join(args.save, train_dataset, f"linear_finetuned.pt")
        if linearized_finetuning
        else os.path.join(args.save, train_dataset, f"finetuned.pt")
    )
    zs_path = (
        os.path.join(args.save, train_dataset, f"linear_zeroshot.pt")
        if linearized_finetuning
        else os.path.join(args.save, train_dataset, f"zeroshot.pt")
    )
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."

    if args.load is not None and args.load.endswith("pt"):
        image_encoder = (
            LinearizedImageEncoder.load(args.load)
            if linearized_finetuning
            else ImageEncoder.load(args.load)
        )
    else:
        print("Building image encoder.")
        image_encoder = (
            LinearizedImageEncoder(args, keep_lang=False)
            if linearized_finetuning
            else ImageEncoder(args)
        )

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    preprocess_fn = model.train_preprocess
    print_every = 10

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)

    data_loaders_to_orth = []
    for d in args.train_datasets_to_orth:
        dataset_to_orth = get_dataset(
            d,
            preprocess_fn,
            location=args.data_location,
            batch_size=args.orth_batch_size,
        )
        data_loaders_to_orth.append(get_dataloader(dataset_to_orth, is_train=True, args=args, image_encoder=None))

    len_orth_datasets = len(args.train_datasets_to_orth)

    num_batches = len(dataset.train_loader)

    # Distribute the data and model across the GPUs.
    ddp_loader = distribute_loader(data_loader)
    ddp_loaders_to_orth = [distribute_loader(loader) for loader in data_loaders_to_orth]
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
        output_device=rank,
    )

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    # Saving zero-shot model
    if args.save is not None and is_main_process():
        os.makedirs(ckpdir, exist_ok=True)
        model_path = (
            os.path.join(ckpdir, f"linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, f"zeroshot.pt")
        )
        ddp_model.module.image_encoder.save(model_path)

    ddp_loader_iters_to_orth = [iter(loader) for loader in ddp_loaders_to_orth]

    for epoch in range(args.epochs):
        ddp_model.train()

        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()
            data_time = time.time() - start_time

            logits = ddp_model(inputs)
            loss = loss_fn(logits, labels)

            penalty = torch.tensor(0)

            # Perform the penalty calculation after a certain number of iterations
            if step > args.penalty_iter:
                penalties = []  # List to store penalties for each dataset

                # Loop through all datasets in ddp_loaders_to_orth
                for i in range(len_orth_datasets):
                    ddp_loader_to_orth = ddp_loader_iters_to_orth[i]
                    
                    try:
                        # Get the next batch from the loader
                        batch_to_orth = next(ddp_loader_to_orth)
                    except StopIteration:
                        # Reset the iterator if it has reached the end
                        ddp_loader_iters_to_orth[i] = iter(ddp_loaders_to_orth[i])
                        ddp_loader_to_orth = ddp_loader_iters_to_orth[i]
                        batch_to_orth = next(ddp_loader_to_orth)

                    # Process the batch
                    batch_to_orth = maybe_dictionarize(batch_to_orth)
                    inputs_to_orth = batch_to_orth["images"].cuda()

                    # Compute tau_jacob and dp_norms for the current dataset
                    tau_jacob = ddp_model.module.image_encoder.model.dp(inputs_to_orth)
                    dp_norms = torch.norm(tau_jacob, dim=1)

                    # Append the penalty for the current dataset to the list
                    penalties.append(dp_norms.mean())

                # Compute the average penalty across all datasets
                if penalties:
                    penalty = torch.stack(penalties).mean()

            total_loss = loss + args.penalty * penalty
            total_loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                print("Saving checkpoint.")
                model_path = (
                    os.path.join(ckpdir, f"linear_checkpoint_{step}.pt")
                    if linearized_finetuning
                    else os.path.join(ckpdir, f"checkpoint_{step}.pt")
                )
                ddp_model.module.image_encoder.save(model_path)

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / len(ddp_loader)

                _, preds = torch.max(logits, 1)
                correct = torch.sum(preds == labels).item()
                accuracy = correct / labels.size(0)

                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    f"Acc: {accuracy}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True,
                )
                run.log({
                    'step': step,
                    'total_loss': total_loss.item(),
                    'train_accuracy': accuracy, 
                    'penalty': penalty.item(), 
                    'loss': loss.item(), 
                })

    # FIXME: Make this work with DDP.
    if is_main_process():
        # We only need to evaluate the model on the first GPU.
        image_encoder = ddp_model.module.image_encoder
        eval_single_dataset(image_encoder, train_dataset, args)

    if args.save is not None and is_main_process():
        zs_path = (
            os.path.join(ckpdir, f"linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, f"zeroshot.pt")
        )
        ft_path = (
            os.path.join(ckpdir, f"linear_finetuned.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, f"finetuned.pt")
        )
        image_encoder.save(ft_path)
        return zs_path, ft_path

    cleanup_ddp()


if __name__ == "__main__":
    train_datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
    ]
    epochs = {
        "Cars": 35,
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SUN397": 14,
        "SVHN": 4,
    }

    for dataset in train_datasets:
        if dataset != "SVHN":
            continue
        args = parse_arguments()

        args.lr = 1e-5
        args.epochs = epochs[dataset]
        args.train_dataset = dataset + "Val"

        args.train_datasets_to_orth = [d + "Val" for d in train_datasets if d != dataset]
        args.train_datasets_to_orth.append("ImageNetVal")
        

        # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
        args.batch_size = 16 if args.model == "ViT-L-14" else 128
        args.orth_batch_size = 4 if args.model == "ViT-L-14" else 16
        args.num_grad_accumulation = 8 if args.model == "ViT-L-14" else 1

        if args.seed is not None:
            args.save = f"/checkpoints_reg_strict_{args.seed}/{args.model}"
        else:
            args.save = f"/checkpoints_reg_strict/{args.model}"
        print("=" * 100)
        print(f"Finetuning {args.model} on {dataset}")
        print("=" * 100)

        group = "{}_{}".format(time.strftime('%Y%m%d-%H%M%S'), str(uuid.uuid4()))
    
        torch.multiprocessing.spawn(finetune, args=(args, group), nprocs=args.world_size)
