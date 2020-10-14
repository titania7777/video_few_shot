import os
import sys
import shutil
import torch.nn as nn

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

def path_check(path):
    if os.path.exists(path):
        while True:
            print("'{}' this path is already exist, do you want continue after remove ? [y/n]".format(path))
            response = input()
            if response == 'y' or response == 'n':
                break
        
        if response == 'y':
            shutil.rmtree(path)
            os.makedirs(path)

        if response == 'n':
            print("this script was terminated by user")
            sys.exit()
    else:
        os.makedirs(path)

def args_print_save(args):
    # print
    print("=================================================")
    [print("{}:{}".format(arg, getattr(args, arg))) for arg in vars(args)]
    print("=================================================")

    # save
    with open(os.path.join(args.save_path, "args.txt"), "w") as f:
        f.write("=================================================\n")
        [f.write("{}:{}\n".format(arg, getattr(args, arg))) for arg in vars(args)]
        f.write("=================================================\n")

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm2d) :
            module.eval()

def initialize(model):
    if type(model) == nn.Linear:
        nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)