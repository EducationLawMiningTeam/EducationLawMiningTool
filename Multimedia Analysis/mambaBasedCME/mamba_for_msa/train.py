from __future__ import absolute_import, division, print_function

import argparse
from pytorch_transformers.modeling_roberta import RobertaConfig
import torch
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.nn import L1Loss, MSELoss, SmoothL1Loss, KLDivLoss
from pytorch_transformers import WarmupLinearSchedule , AdamW
from networks.mamba.model import MambaTextClassification
from utils.databuilder import set_up_data_loader
from utils.set_seed import set_random_seed, seed
from utils.metric import score_model
# from config.global_configs import DEVICE
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0")
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosei")
    parser.add_argument("--data_path", type=str, default='./dataset/MOSEI_16_sentilare_unaligned_data.pkl')
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--dev_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=70)
    parser.add_argument("--beta_shift", type=float, default=1.0)
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    parser.add_argument(
        "--model",
        type=str,
        choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base"],
        default="roberta-base")
    parser.add_argument("--model_name_or_path", default="../gpt-neox-20b", type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--mamba_path", type=str, default="../790m")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--test_step", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=int, default=2)
    parser.add_argument("--warmup_proportion", type=float, default=0.4)
    parser.add_argument("--seed", type=seed, default=9, help="integer or 'random'") #6758
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    return parser.parse_args()

def prep_for_training(args, num_train_optimization_steps: int):
    model = MambaTextClassification.from_pretrained(args.mamba_path)
    model.to(DEVICE)
    #Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    study_more = ["classfication_head"]
    CME_params = ['CME']
    optimizer_grouped_parameters = [
        {
            "params": [
                 p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay
        }
        # {"params": model.roberta.encoder.CME.parameters(), 'lr':args.learning_rate, "weight_decay": args.weight_decay},
        # {
        #     "params": [
        #         p for n, p in param_optimizer if any(nd in n for nd in no_decay)  and not any(nd in n for nd in CME_params)
        #     ],
        #     "weight_decay": 0.0,
        # },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        t_total=num_train_optimization_steps,
    )
    return model, optimizer, scheduler

def train_epoch(args, model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    preds = []
    labels = []
    tr_loss = 0

    nb_tr_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        outputs = model(
            input_ids,
            visual,
            acoustic,
            visual_ids,
            acoustic_ids,
            pos_ids, senti_ids, polarity_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids, 
        )
        logits = outputs[0]
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1))
        
        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step
        # loss2 = loss_fct(outputs[1].view(-1), label_ids.view(-1))
        # loss3 = loss_fct(outputs[2].view(-1), label_ids.view(-1))
        # loss = loss1 + loss2 + loss3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        tr_loss += loss.item()
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.detach().cpu().numpy()
        logits = np.squeeze(logits).tolist()
        label_ids = np.squeeze(label_ids).tolist()
        preds.extend(logits)
        labels.extend(label_ids)

    preds = np.array(preds)
    labels = np.array(labels)

    return tr_loss / nb_tr_steps, preds, labels

def evaluate_epoch(args, model: nn.Module, dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []
    loss = 0
    nb_dev_examples, nb_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                visual_ids,
                acoustic_ids,
                pos_ids, senti_ids, polarity_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            # logits = outputs[0][0]
            logits = outputs[0]
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step


            # loss2 = loss_fct(outputs[1].view(-1), label_ids.view(-1))
            # loss3 = loss_fct(outputs[2].view(-1), label_ids.view(-1))
            # loss = loss1 + loss2 + loss3
            loss += loss.item()
            nb_steps += 1
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()
            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return loss / nb_steps, preds, labels

def train(
    args,
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    optimizer,
    scheduler,):
    train_accuracies = []
    valid_losses = []
    test_accuracies = []
    test_f_scores = []
    non0_test_accs = []
    non0_test_f_scores = []
    test_maes = []
    test_corrs = []
    for epoch_i in range(int(args.n_epochs)):
        train_loss, train_pre, train_label = train_epoch(args, model, train_dataloader, optimizer, scheduler)
        valid_loss, valid_pre, valid_label = evaluate_epoch(args, model, validation_dataloader)
        test_loss, test_pre, test_label = evaluate_epoch(args, model, test_data_loader)
        train_acc, train_mae, train_corr, train_f_score = score_model(train_pre, train_label, "train")
        test_acc, test_mae, test_corr, test_f_score = score_model(test_pre, test_label, "test")
        non0_test_acc, _, _, non0_test_f_score = score_model(test_pre, test_label, "test", use_zero=True)
        valid_acc, valid_mae, valid_corr, valid_f_score = score_model(valid_pre, valid_label, "valid")
        print(
            "[epoch-{}/{}] : (loss / acc) train : {:.4f} / {:.2f}%       valid : {:.4f} / {:.2f}%       test : {:.4f} / {:.2f}%".format(
                epoch_i + 1, args.n_epochs, train_loss, train_acc * 100, valid_loss, valid_acc * 100, test_loss, test_acc * 100
            )
        )
        train_accuracies.append(train_acc)
        valid_losses.append(valid_loss.cpu())
        test_accuracies.append(test_acc)
        test_f_scores.append(test_f_score)
        non0_test_accs.append(non0_test_acc)
        non0_test_f_scores.append(non0_test_f_score)
        test_maes.append(test_mae)
        test_corrs.append(test_corr)
        # wandb.log(
        #     (
        #         {
        #             "train_loss": train_loss,
        #             "valid_loss": valid_loss,
        #             "train_acc": train_acc,
        #             "train_corr": train_corr,
        #             "valid_acc":valid_acc,
        #             "valid_corr":valid_corr,
        #             "test_loss":test_loss,
        #             "test_acc": test_acc,
        #             "test_mae": test_mae,
        #             "test_corr": test_corr,
        #             "test_f_score": test_f_score,
        #             "non0_test_acc": non0_test_acc,
        #             "non0_test_f_score": non0_test_f_score,
        #             "best_valid_loss": min(valid_losses),
        #             "best_test_acc": max(test_accuracies),
        #         }
        #     )
        # )
        wandb.log(
            (
                {
                    "best_test_mae": min(test_maes),
                    "best_test_corr": max(test_corrs),
                    "best_valid_loss": min(valid_losses),
                    "best_non0_test_f_score": max(non0_test_f_scores),
                    "best_non0_test_acc": max(non0_test_accs),
                    "best_test_f_score": max(test_f_scores),
                    "best_test_acc": max(test_accuracies),
                }
            )
        )
    x = np.arange(len(valid_losses))
    for i in x:
        i += 1
    plt.figure()
    plt.plot(x, valid_losses, label='Validation Loss',  color='blue')
    plt.legend()
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.show()
    plt.clf()
    max_tt_value = np.array(test_accuracies).max()
    max_tt_index = np.array(test_accuracies).argmax()
    plt.plot(x, train_accuracies, label="Train Accuracies", color = "blue")
    plt.plot(x, test_accuracies, label='Test Accuracies', color = "red")
    plt.annotate(
        f'Top Acc: {max_tt_value}',
        xy=(max_tt_index, max_tt_value),
        xytext=(max_tt_index, max_tt_value + 0.1),  # 将文本放在最高点的上方一点
        ha='center',  # 水平对齐方式
        clip_on=False,
        arrowprops=dict(facecolor='brown', shrink=0.05, width=0.5)  # 箭头属性
    )
    plt.axhline(y=max_tt_value, color='r', linestyle='--')
    plt.legend()
    plt.xlabel('Epoches')
    plt.ylabel('Acc')
    plt.show()

def main():
    args = parser_args()
    wandb.init(project="CENet", reinit=True)

    set_random_seed(args.seed)
    wandb.config.update(args)

    (train_data_loader,
    dev_data_loader,
    test_data_loader,
    num_train_optimization_steps,
    ) = set_up_data_loader(args)

    model, optimizer, scheduler = prep_for_training(args, num_train_optimization_steps)

    train(
        args,
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )

if __name__ == "__main__":
    main()