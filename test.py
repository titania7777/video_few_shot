import torch
import os
import sys
import argparse
import torch.nn.functional as F
from models import R2Plus1D, Resnet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from UCF101 import UCF101, CategoriesSampler

def printer(status, epoch, num_epochs, batch, num_batchs, loss, loss_mean, acc, acc_mean):
    sys.stdout.write("\r[{}]-[Epoch {}/{}] [Batch {}/{}] [Loss: {:.2f} (mean: {:.2f}), Acc: {:.2f}% (mean: {:.2f}%)]".format(
            status,
            epoch,
            num_epochs,
            batch,
            num_batchs,
            loss,
            loss_mean,
            acc,
            acc_mean
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-path", type=str, default="../Data/UCF101_frames/")
    parser.add_argument("--labels-path", type=str, default="./UCF101_few_shot_labels/")
    parser.add_argument("--save-path", type=str, default="./save/train1")
    parser.add_argument("--use-best", action="store_true")
    parser.add_argument("--frame-size", type=str, default=112)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=35)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--model", type=str, default='resnet')
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=1)
    args = parser.parse_args()

    test_dataset = UCF101(
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        setname='test',
        # pad option
        random_pad_sample=False,
        pad_option='default',
        # frame sampler option
        uniform_frame_sample=True,
        random_start_position=False,
        max_interval=7,
        random_interval=False,
    )
    test_sampler = CategoriesSampler(test_dataset.classes, 600, args.way, args.shot, args.query)
    
    # in windows has some issue when try to use DataLoader in pytorch, i don't know why..
    test_loader = DataLoader(dataset=test_dataset, batch_sampler=test_sampler, num_workers=0 if os.name == 'nt' else 4, pin_memory=True)
        
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
    
    if args.use_best:
        save_path = os.path.join(args.save_path, "best.pth")
    else:
        save_path = os.path.join(args.save_path, "last.pth")
    
    assert os.path.exists(save_path), "'{}' file is not exists !!".format(save_path)
    model.load_state_dict(torch.load(save_path))
    
    model.eval()
    best = 0
    print("test... {}-way {}-shot {}-query".format(args.way, args.shot, args.query))
    for e in range(1, args.num_epochs+1):
        test_acc = []
        test_loss = []
        for i, (datas, _) in enumerate(test_loader):
            datas = datas.to(device)
            pivot = args.way * args.shot
            
            shot, query = datas[:pivot], datas[pivot:]
            labels = torch.arange(args.way).repeat(args.query).to(device)
            # one_hot_labels = Variable(torch.zeros(args.way*args.query, args.way).scatter_(1, labels.view(-1, 1), 1)).to(device)

            pred = model(shot, query)

            loss = F.cross_entropy(pred, labels)
            test_loss.append(loss.item())

            acc = 100 * (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            test_acc.append(acc)

            printer("test", e, args.num_epochs, i+1, len(test_loader), loss.item(), sum(test_loss)/len(test_loss), acc, sum(test_acc)/len(test_acc))
        print("")