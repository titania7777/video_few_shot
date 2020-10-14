import torch
import os
import sys
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from utils import path_check, args_print_save, printer
from models import R2Plus1D, Resnet
from UCF101 import UCF101, CategoriesSampler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-path", type=str, default="../Data/UCF101/UCF101_frames/")
    parser.add_argument("--labels-path", type=str, default="./UCF101_few_shot_labels/")
    parser.add_argument("--save-path", type=str, default="./save/train1/")
    parser.add_argument("--tensorboard-path", type=str, default="./tensorboard/train1")
    parser.add_argument("--frame-size", type=str, default=112)
    parser.add_argument("--num-epochs", type=int, default=40)
    # =============================================================
    # pad options
    parser.add_argument("--random-pad-sample", action="store_true")
    parser.add_argument("--pad-option", type=str, default="default")
    # frame options
    parser.add_argument("--uniform-frame-sample", action="store_true")
    parser.add_argument("--random-start-position", action="store_true")
    parser.add_argument("--max-interval", type=int, default=7)
    parser.add_argument("--random-interval", action="store_true")
    # =============================================================
    parser.add_argument("--sequence-length", type=int, default=35)
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--scheduler-step-size", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=1)
    args = parser.parse_args()

    # path check
    path_check(args.save_path)
    # make tensorboard
    writer = SummaryWriter(args.tensorboard_path)
    # print args and save
    args_print_save(args)

    train_dataset = UCF101(
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        setname='train',
        # pad options
        random_pad_sample=args.random_pad_sample,
        pad_option=args.pad_option,
        # frame sample options
        uniform_frame_sample=args.uniform_frame_sample,
        random_start_position=args.random_start_position,
        max_interval=args.max_interval,
        random_interval=args.random_interval,
    )

    val_dataset = UCF101(
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        setname='val',
        # pad options
        random_pad_sample=False,
        pad_option='default',
        # frame sample options
        uniform_frame_sample=True,
        random_start_position=False,
        max_interval=7,
        random_interval=False,
    )
    train_sampler = CategoriesSampler(train_dataset.classes, 100, args.way, args.shot, args.query)
    val_sampler = CategoriesSampler(val_dataset.classes, 200, args.way, args.shot, args.query)
    
    # in windows has some issue when try to use DataLoader in pytorch, i don't know why..
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=0 if os.name == 'nt' else 4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=0 if os.name == 'nt' else 4, pin_memory=True)

    assert args.model in ['resnet', 'r2plus1d'], "'{}' model is invalid".format(setname)
    if args.model == 'resnet':
        model = Resnet(
            way=args.way,
            shot=args.shot,
            query=args.query,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            bidirectional=args.bidirectional,
        )

    if args.model == 'r2plus1d':
        model = R2Plus1D(
            way=args.way,
            shot=args.shot,
            query=args.query,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    best = 0
    total_loss = 0
    total_acc = 0
    n_iter_train = 0
    n_iter_val = 0
    print("train... {}-way {}-shot {}-query".format(args.way, args.shot, args.query))
    for e in range(1, args.num_epochs+1):
        train_acc = []
        train_loss = []
        for i, (datas, _) in enumerate(train_loader):
            model.train()
            datas = datas.to(device)
            pivot = args.way * args.shot
            
            shot, query = datas[:pivot], datas[pivot:]
            labels = torch.arange(args.way).repeat(args.query).to(device)
            # one_hot_labels = Variable(torch.zeros(args.way*args.query, args.way).scatter_(1, labels.view(-1, 1), 1)).to(device)

            pred = model(shot, query)

            # calculate loss
            loss = F.cross_entropy(pred, labels)
            train_loss.append(loss.item())
            total_loss = sum(train_loss)/len(train_loss)

            # update weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            acc = (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            train_acc.append(acc)
            total_acc = sum(train_acc)/len(train_acc)

            # print result
            printer("train", e, args.num_epochs, i+1, len(train_loader), loss.item(), total_loss, acc * 100, total_acc * 100)

            # tensorboard
            writer.add_scalar("Loss/train", loss.item(), n_iter_train)
            writer.add_scalar("Accuracy/train", acc, n_iter_train)
            n_iter_train += 1

        print("")
        val_acc = []
        val_loss = []
        for i, (datas, _) in enumerate(val_loader):
            model.eval()
            datas = datas.to(device)
            pivot = args.way * args.shot
            
            shot, query = datas[:pivot], datas[pivot:]
            labels = torch.arange(args.way).repeat(args.query).to(device)
            # one_hot_labels = Variable(torch.zeros(args.way*args.query, args.way).scatter_(1, labels.view(-1, 1), 1)).to(device)

            pred = model(shot, query)

            # calculate loss
            loss = F.cross_entropy(pred, labels)
            val_loss.append(loss.item())
            total_loss = sum(val_loss)/len(val_loss)

            # calculate accuracy
            acc = (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            val_acc.append(acc)
            total_acc = sum(val_acc)/len(val_acc)

            # print result
            printer("val", e, args.num_epochs, i+1, len(val_loader), loss.item(), total_loss, acc * 100, total_acc * 100)

            # tensorboard
            writer.add_scalar("Loss/val", loss.item(), n_iter_val)
            writer.add_scalar("Accuracy/val", acc, n_iter_val)
            n_iter_val += 1

        if total_acc > best:
            best = total_acc
            torch.save(model.state_dict(), os.path.join(args.save_path, "best.pth"))
        torch.save(model.state_dict(), os.path.join(args.save_path, "last.pth"))
        print(" Best: {:.2f}%".format(best * 100))

        lr_scheduler.step()