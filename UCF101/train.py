import torch
import os
import sys
import argparse
import torch.nn.functional as F
from models import Model
from torch.utils.data import DataLoader
from torch.autograd import Variable
from UCF101 import UCF101, CategoriesSampler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_path", type=str, default="../datas/UCF101_frames/")
    parser.add_argument("--labels_path", type=str, default="./UCF101_few_shot_labels/")
    parser.add_argument("--frame_size", type=str, default=224)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--sequence_length", type=int, default=40)
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=15)
    args = parser.parse_args()

    train_dataset = UCF101(
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        setname='train',
        # pad option
        random_pad_sample=True,
        pad_option='default',
        # frame sampler option
        uniform_frame_sample=True,
        random_start_position=True,
        max_interval=7,
        random_interval=True,
    )

    val_dataset = UCF101(
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        setname='val',
        # pad option
        random_pad_sample=False,
        pad_option='default',
        # frame sampler option
        uniform_frame_sample=True,
        random_start_position=False,
        max_interval=7,
        random_interval=False,
    )
    train_sampler = CategoriesSampler(train_dataset.classes, 100, args.way, args.shot, args.query)
    val_sampler = CategoriesSampler(val_dataset.classes, 200, args.way, args.shot, args.query)
    
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=0 if os.name == 'nt' else 4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=0 if os.name == 'nt' else 4, pin_memory=True)
        
    model = Model(
        way=args.way,
        shot=args.shot,
        query=args.query,
        num_layers=1,
        hidden_size=1024,
        bidirectional=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best = 0
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

            loss = F.cross_entropy(pred, labels)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = 100 * (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            train_acc.append(acc)

            sys.stdout.write(
                "\rtrain-[Epoch {}/{}] [Batch {}/{}] [Loss: {:.2f} (mean: {:.2f}), Acc: {:.2f}% (mean: {:.2f}%)]".format
                (
                    e,
                    args.num_epochs,
                    i+1,
                    len(train_loader),
                    loss.item(),
                    sum(train_loss)/len(train_loss),
                    acc,
                    sum(train_acc)/len(train_acc),
                )
            )

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

            loss = F.cross_entropy(pred, labels)
            val_loss.append(loss.item())

            acc = 100 * (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            val_acc.append(acc)

            sys.stdout.write(
                "\rval-[Epoch {}/{}] [Batch {}/{}] [Loss: {:.2f} (mean: {:.2f}), Acc: {:.2f}% (mean: {:.2f}%), Best: {:.2f}%]".format
                (
                    e,
                    args.num_epochs,
                    i+1,
                    len(val_loader),
                    loss.item(),
                    sum(val_loss)/len(val_loss),
                    acc,
                    sum(val_acc)/len(val_acc),
                    best
                )
            )
        
        print("")
        if sum(val_acc)/len(val_acc) > best:
            best = sum(val_acc)/len(val_acc)
        
        lr_scheduler.step()
