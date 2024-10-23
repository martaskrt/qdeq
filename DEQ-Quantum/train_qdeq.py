# coding: utf-8
# This code was largely based off https://github.com/locuslab/deq. We would like to thank the authors. 

import argparse
import time
import math
import os, sys
import itertools
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('../')

from models.qdeq_model import QDEQCircuit
from lib.solvers import anderson, broyden
from lib import radam
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel
from torch.utils.tensorboard import SummaryWriter

from load_mnist import MNIST
from load_cifar import CIFAR

import wandb

parser = argparse.ArgumentParser(description='PyTorch QDEQ Model')
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=['fourier', "mnist", "fashion_mnist", "cifar10"],
                    help='dataset name')
parser.add_argument('--num_classes', type=int, default=10,
                    choices=[2,4,10],
                    help='number of mnist classes')
parser.add_argument('--n_wires', type=int, default=4,
                    choices=[4,10],
                    help='number of qubits=wires in circuit')
parser.add_argument('--amplitude_encoder', type=bool, default=True,
                     help='choice whether to choose amplitude or angle encoder')
parser.add_argument('--mode', type=str, default='implicit',
                    choices=['implicit', 'direct'])
parser.add_argument('--n_layer', type=int, default=-1,
                    help='number of layers in direct solver')
# Dropouts
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate (default: 0.05)')

# Initializations
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')

# Optimizers
parser.add_argument('--optim', default='Adam', type=str,
                    choices=['Adam', 'SGD', 'Adagrad', 'RMSprop', 'RAdam'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='the number of steps to warm up the learning rate to its lr value')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')

# Gradient updates
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=200000,
                    help='upper epoch limit (at least 200K for WT103 or PTB)')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')


# DEQ related [Bai et al. 2019]
parser.add_argument('--f_solver', default='anderson', type=str,
                    choices=['anderson', 'broyden'],
                    help='forward solver to use (only anderson and broyden supported now)')
parser.add_argument('--b_solver', default='broyden', type=str,
                    choices=['anderson', 'broyden', 'None'],
                    help='backward solver to use (if None, then set it to f_solver)')
parser.add_argument('--stop_mode', type=str, default="rel",
                    choices=['abs', 'rel'],
                    help='stop criterion absolute or relative')
parser.add_argument('--rand_f_thres_delta', type=int, default=0,
                    help='use (f_thres + U(-delta, 0)) for forward threshold (delta default to 0)')
parser.add_argument('--f_thres', type=int, default=40,
                    help='forward pass Broyden threshold')
parser.add_argument('--b_thres', type=int, default=40,
                    help='backward pass Broyden threshold')

# Jacobian regularization related [Bai et al. 2021]
parser.add_argument('--jac_loss_weight', type=float, default=0.0,
                    help='jacobian regularization loss weight (default to 0)')
parser.add_argument('--jac_loss_freq', type=float, default=0.0,
                    help='the frequency of applying the jacobian regularization (default to 0)')
parser.add_argument('--jac_incremental', type=int, default=0,
                    help='if positive, increase jac_loss_weight by 0.1 after this many steps')
parser.add_argument('--spectral_radius_mode', action='store_true',
                    help='compute spectral radius at validation time')

# Training techniques
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--wnorm', action='store_true',
                    help='apply WeightNorm to the weights')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=4000,
                    help='evaluation interval')
parser.add_argument('--work_dir', default='QC', type=str,
                    help='experiment directory.')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--pretrain_steps', type=int, default=0,
                    help='number of pretrain steps (default to 0')
parser.add_argument('--start_train_steps', type=int, default=0,
                    help='starting training step count (default to 0)')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--load', type=str, default='',
                    help='path to load weight')
parser.add_argument('--name', type=str, default='N/A',
                    help='name of the trial')

args = parser.parse_args()
args.tied = not args.not_tied
args.pretrain_steps += args.start_train_steps
args.work_dir += "deq"

print("args", args)
wandb.init(project='qdeq')
wandb.init(settings=wandb.Settings(code_dir=".."), tags=[args.dataset, f"{args.num_classes}C"])
wandb.run.log_code("..")
wandb.config.update(args)

assert args.batch_size % args.batch_chunk == 0
if args.mode == "direct" or args.pretrain_steps >0:
    assert args.n_layer > 0

args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
timestamp = time.strftime('%Y%m%d-%H%M%S')
if args.restart_dir:
    timestamp = args.restart_dir.split('/')[1]
args.name += "_" + timestamp
args.work_dir = os.path.join(args.work_dir, args.name)
logging = create_exp_dir(args.work_dir,
    scripts_to_save=['train_qdeq.py', 'models/qdeq_model.py', '../lib/solvers.py'], debug=args.debug)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda:0' if args.cuda else 'cpu')
device='cuda:0'

###############################################################################
# Load data
###############################################################################
if "mnist" in args.dataset:
    if args.dataset == "mnist":
        fashion=False
        if args.num_classes == 2:
            classes = [3,6]
        elif args.num_classes == 4:
            classes = [0,3,6,9]
        elif args.num_classes == 10:
            classes = list(range(10))
        fashion=False
    elif args.dataset == "fashion_mnist":
        if args.num_classes == 2:
            classes = [3,6]
        elif args.num_classes == 4:
            classes = [0,3,6,9]
        elif args.num_classes == 10:
            classes = list(range(10))
        fashion=True

    dataset = MNIST(
            root='./mnist_data',
            train_valid_split_ratio=[0.8, 0.2],
            digits_of_interest=classes,
            #n_test_samples=1000,
            device=device,
            fashion=fashion
        )
elif "cifar10" in args.dataset:
    assert args.num_classes == 10
    classes = list(range(10))

    dataset = CIFAR(
            root='./cifar_data',
            train_valid_split_ratio=[0.8, 0.2],
            digits_of_interest=classes,
            device=device,
            )

dataflow = dict()
device='cuda:0'
if args.dataset in ["fourier", "mnist", "fashion_mnist", "cifar10"]:
    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split], generator=torch.Generator(device=device))
        num_workers=8
        pin_memory=True
        if args.cuda:
            num_workers=0
            pin_memory=False
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            )
else:
    dataflow['train'] = [None]
    dataflow['valid'] = [None]
    dataflow['test'] = [None]
###############################################################################
# Build the model
###############################################################################

if args.dataset == "thermal":
    model = QDEQCircuit_Thermal(args.dataset, args.mode, n_layer=args.n_layer, pretrain_steps=args.pretrain_steps, device=device, f_solver=eval(args.f_solver), b_solver=eval(args.b_solver), stop_mode=args.stop_mode, logging=logging).to(device)

else:
    model = QDEQCircuit(args.dataset, args.mode, n_layer=args.n_layer, pretrain_steps=args.pretrain_steps, device=device, f_solver=eval(args.f_solver), b_solver=eval(args.b_solver), stop_mode=args.stop_mode, logging=logging, num_classes=args.num_classes, n_wires=args.n_wires, amplitude_encoder=args.amplitude_encoder).to(device)

#### optimizer
optimizer = getattr(optim if args.optim != 'RAdam' else radam, args.optim)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if not args.debug:
    writer = SummaryWriter(log_dir='log/', flush_secs=5)
else:
    writer = None

pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("pytorch_train_params", pytorch_train_params)

#### scheduler
if args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_step, eta_min=args.eta_min)


###############################################################################
# Training code
###############################################################################

def evaluate(data_subset,test=False):
    global train_step
    model.eval()

    # Evaluation
    val_loss, val_acc,val_res, val_step = 0, 0, 0, 0
    with torch.no_grad():
        mems = []
        device='cuda:0'
        for batch, data in enumerate(data_subset):
            if args.dataset in ['fourier', 'mnist', "fashion_mnist", "cifar10"]:
                x = data['x'].to(device)
                target = data['y'].to(device)
            else:
                x, target = None, None
            ret = model(x, target, mems, dataset=args.dataset, train_step=train_step, f_thres=args.f_thres, b_thres=args.b_thres, compute_jac_loss=False, writer=writer, debug=test)
            loss, acc, res, jac_loss, _, mems = ret[0], ret[1], ret[2], ret[3], ret[4], ret[5:]
            loss = loss.mean()
            val_loss += loss.float().item() * len(x)
            val_acc += acc * len(x)
            val_res += res * len(x)
            #val_step += 1
            val_step += len(x)
    model.train()
    return val_loss / val_step, val_acc / val_step, val_res / val_step


results_dict = {}
def train():
    global train_step, train_loss, train_acc, train_res, train_jac_loss, best_val_loss, eval_start_time, log_start_time, results_dict, device
    model.train()

    mems = []
    device='cuda:0'
    total_samples = 0
    for batch, data in enumerate(dataflow['train']):
        if args.dataset != "thermal":
            x = data['x'].to(device)
            target = data['y'].to(device)
        else:
            x, target = None, None

        train_step += 1
        model.zero_grad()

        # For DEQ:
        compute_jac_loss = np.random.uniform(0,1) < args.jac_loss_freq
        f_thres = args.f_thres + (random.randint(-args.rand_f_thres_delta,0) if args.rand_f_thres_delta > 0 else 0)
        b_thres = args.b_thres

        # Mode 2: Normal training with one batch per iteration
        ret = model(x, target, mems, dataset=args.dataset, train_step=train_step, f_thres=f_thres, b_thres=b_thres, compute_jac_loss=compute_jac_loss, writer=writer)
        loss, acc, res, jac_loss, _, mems = ret[0], ret[1], ret[2], ret[3], ret[4], ret[5:]
        loss = loss.float().mean().type_as(loss)
        jac_loss = jac_loss.float().mean().type_as(loss)
        if compute_jac_loss:
            (loss + jac_loss * args.jac_loss_weight).backward()
            train_jac_loss.append(jac_loss.float().item())
        else:
            loss.backward()

        lenx = len(x) 
        train_loss += loss.float().item() * lenx
        train_acc += acc * lenx
        train_res += res * lenx
        total_samples += lenx

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        max_memory = torch.cuda.max_memory_allocated()
        optimizer.step()

        # Step-wise learning rate annealing according to some scheduling (we ignore 'constant' scheduling)
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)

        # Logging of training progress
        if train_step % args.log_interval == 0:
            cur_loss = train_loss / total_samples #args.log_interval
            cur_acc = train_acc / total_samples #args.log_interval
            cur_res = train_res / total_samples #args.log_interval
            cur_ppl = math.exp(cur_loss)
            cur_jac_loss = np.mean(train_jac_loss)
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                    '| ms/batch {:5.2f} | jac {:5.4f} | loss {:5.7f} | acc {:5.7f} | res {:5.7f} | ppl {:9.3f} | max_mem {:.10f}'.format(
                epoch, train_step, batch+1, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_jac_loss, cur_loss, cur_acc, cur_res, cur_ppl, max_memory)

            wandb.log({"train_loss": cur_jac_loss,
                       "train_acc": cur_acc,
                       "train_res": cur_res,
                       "epoch": epoch,
                       "train_step": train_step,
                       "mode": 0 if args.mode =="direct" or train_step <= args.pretrain_steps else 1})

            logging(log_str)
            train_loss = 0
            train_acc = 0
            train_res =0
            total_samples=0
            train_jac_loss = []
            log_start_time = time.time()

            results_dict[epoch]=[cur_loss, cur_acc, cur_res]

            if writer is not None:
                writer.add_scalar('result/train_loss', cur_loss, train_step)
                writer.add_scalar('result/train_ppl', cur_ppl, train_step)

        # Enter evaluation/inference mode once in a while and save the model if needed
        if train_step % args.eval_interval == 0:
            val_loss, val_acc, val_res = evaluate(dataflow['valid'])

            results_dict[epoch]+=[val_loss, val_acc, val_res]

            val_ppl = math.exp(val_loss)
            logging('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.7f} | valid acc {:5.7f} | valid res {:5.7f} | valid ppl {:5.7f}'.format(
                train_step // args.eval_interval, train_step,
                (time.time() - eval_start_time), val_loss, val_acc, val_res, val_ppl)

            wandb.log({"val_loss": val_loss,
                       "val_acc": val_acc,
                       "val_res": val_res,
                       "epoch": epoch,
                       "train_step": train_step,
                       "mode": 0 if args.mode =="direct" or train_step <= args.pretrain_steps else 1})

            logging(log_str)
            logging('-' * 100)

            if writer is not None:
                writer.add_scalar('result/valid_loss', val_loss, train_step)
                writer.add_scalar('result/valid_ppl', val_ppl, train_step)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if not args.debug:
                    with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                        print(f'Saved Model! Experiment name: {args.name}')
                        torch.save(model, f)
                        model.save_weights(path=args.work_dir, name='model_state_dict')
                    with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)
                best_val_loss = val_loss
                wandb.log({"best_val_loss": best_val_loss,
                           "best_val_acc": val_acc,
                           "best_val_res": val_res,
                           "epoch": epoch,
                           "train_step": train_step,
                           "mode": 0 if args.mode =="direct" or train_step <= args.pretrain_steps else 1})

            eval_start_time = time.time()

        if train_step == args.pretrain_steps:
            print("You are using pre-training, which has completed :-)")
            model.save_weights(args.work_dir, f"pretrain_{train_step}_{args.name}")
            torch.cuda.empty_cache()
            best_val_loss = 0
        if train_step == args.max_step:
            break

        if train_step > max(0, args.pretrain_steps) and args.jac_loss_weight > 0 and args.jac_loss_freq > 0 and \
           args.jac_incremental > 0 and train_step % args.jac_incremental == 0:
            logging(f"Adding 0.1 to jac. regularization weight after {train_step} steps")
            args.jac_loss_weight += 0.1

# Loop over epochs.
train_step = 0
train_loss = 0
train_acc = 0
train_res = 0
train_jac_loss = []
best_val_loss = None

log_start_time = time.time()
eval_start_time = time.time()
start_time,end_time = 0,0
start_time = time.time()

try:
    for epoch in itertools.count(start=1):
        train()
        if train_step == args.max_step:
            logging('-' * 100)
            logging('End of training')
            break
except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')

#emissions: float = tracker.stop()
end_time = time.time()
# Load the best saved model.
print("loading best model...")
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
para_model = model.to(device)

# Run on test data.
test_loss, test_acc, test_res = evaluate(dataflow['test'])

if epoch not in results_dict:
    results_dict[epoch] = []
results_dict[epoch] += [test_loss, test_acc, test_res]
logging('=' * 100)
logging('| End of training | test loss {:5.7f} | test acc {:5.7f} | test res {:5.7f} | test ppl {:9.3f}'.format(test_loss, test_acc, test_res, math.exp(test_loss)))

wandb.log({"test_loss": test_loss,
           "test_acc": test_acc,
           "test_res": test_res})

logging('=' * 100)
with open(os.path.join(args.work_dir + "_results.csv"), 'w') as f:
    f.write("epoch,train_loss,train_acc,train_res,val_loss,val_acc,val_res,test_loss,test_acc,test_res\n")
    for key in range(1, epoch+1):
        curr_str = str(key)
        num_vals = 6
        if key == epoch:
            num_vals += 3
        curr_str += '\n'
        f.write(curr_str)
print("RUNTIME:::{:5.5f}".format(end_time - start_time))
