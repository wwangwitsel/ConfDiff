import numpy as np
import os
import torch
from utils.utils_data import generate_pretrain_loaders, generate_binary_pretrain_data, gen_index_dataset, train_test_data_gen, gen_confdiff_train_loader
import argparse
from utils.utils_models import linear_model, mlp_model
from utils.utils_loss import logistic_loss
from utils.utils_algo import get_model, accuracy_check, train_data_confidence_gen
from cifar_models import resnet
from algorithms import pretrainLR, ConfDiffUnbiased, ConfDiffReLU, ConfDiffABS


parser = argparse.ArgumentParser()

parser.add_argument('-lr', help='optimizer\'s learning rate', default=1e-3, type=float)
parser.add_argument('-pretrain_bs', help='batch_size of ordinary labels.', default=256, type=int)
parser.add_argument('-bs', help='batch_size of ordinary labels.', default=256, type=int)
parser.add_argument('-ds', help='specify a dataset', default='mnist', type=str, required=False)
parser.add_argument('-mo', help='model name', default='mlp', choices=['linear', 'mlp', 'resnet'], type=str, required=False)
parser.add_argument('-me', help='specify a method', default='ConfDiffABS', type=str, choices=['ConfDiffUnbiased','ConfDiffReLU', 'ConfDiffABS'], required=False)
parser.add_argument('-pretrain_ep', help='number of pretrain epochs', type=int, default=10)
parser.add_argument('-ep', help='number of ConfDiff epochs', type=int, default=200)
parser.add_argument('-n', help = 'number of unlabeled data pairs', default=15000, type=int, required=False)
parser.add_argument('-prior', help='the class prior of the data set', type=float, default=0.5)
parser.add_argument('-wd', help='weight decay', default=1e-5, type=float)
parser.add_argument('-lo', help='specify a loss function', default='logistic', type=str, choices=['logistic'], required=False)
parser.add_argument('-uci', help = 'Is UCI datasets?', default=0, type=int, choices=[0,1], required=False)
parser.add_argument('-gpu', help = 'used gpu id', default='0', type=str, required=False)
parser.add_argument('-seed', help = 'Random seed', default=1, type=int, required=False)
parser.add_argument('-run_times', help='random run times', default=5, type=int, required=False)

args = parser.parse_args()
device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")

if args.lo == 'logistic':
    loss_fn = logistic_loss

acc_run_list = torch.zeros(args.run_times)

save_pretrain_dir = "./result/pretrain"
save_total_dir = "./result/total"
save_detail_dir = "./result/detail"
if not os.path.exists(save_pretrain_dir):
    os.makedirs(save_pretrain_dir)
if not os.path.exists(save_total_dir):
    os.makedirs(save_total_dir)
if not os.path.exists(save_detail_dir):
    os.makedirs(save_detail_dir)

save_pretrain_name = "Res_pretrain_ds_{}_prior_{}_me_{}_mo_{}_lr_{}_wd_{}_pretrain_bs_{}_pretrain_ep_{}_seed_{}_n_{}.csv".format(args.ds, args.prior, args.me, args.mo, args.lr, args.wd, args.pretrain_bs, args.pretrain_ep, args.seed, args.n)
save_total_name = "Res_total_ds_{}_prior_{}_me_{}_mo_{}_lr_{}_wd_{}_bs_{}_ep_{}_pretrain_bs_{}_pretrain_ep_{}_seed_{}_n_{}.csv".format(args.ds, args.prior, args.me, args.mo, args.lr, args.wd, args.bs, args.ep, args.pretrain_bs, args.pretrain_ep, args.seed, args.n)
save_detail_name = "Res_detail_ds_{}_prior_{}_me_{}_mo_{}_lr_{}_wd_{}_bs_{}_ep_{}_pretrain_bs_{}_pretrain_ep_{}_seed_{}_n_{}.csv".format(args.ds, args.prior, args.me, args.mo, args.lr, args.wd, args.bs, args.ep, args.pretrain_bs, args.pretrain_ep, args.seed, args.n)

save_pretrain_path = os.path.join(save_pretrain_dir, save_pretrain_name)
save_total_path = os.path.join(save_total_dir, save_total_name)
save_detail_path = os.path.join(save_detail_dir, save_detail_name)

if os.path.exists(save_pretrain_path):
    os.remove(save_pretrain_path)
if os.path.exists(save_total_path):
    os.remove(save_total_path)
if os.path.exists(save_detail_path):
    os.remove(save_detail_path)

if_write = True

if if_write:
    with open(save_pretrain_path, 'a') as f:
        f.writelines("epoch,train_loss,train_accuracy,test_accuracy\n")
    with open(save_total_path, 'a') as f:
        f.writelines("run_idx,acc,std\n")
    with open(save_detail_path, 'a') as f:
        f.writelines("epoch,train_loss,test_accuracy\n")

for run_idx in range(args.run_times):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed); 
    torch.cuda.manual_seed_all(args.seed);
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.seed = args.seed + 1
    print('the {}-th random round'.format(run_idx))
    
    positive_pretrain_data, negative_pretrain_data, positive_pretrain_label, negative_pretrain_label, positive_pretrain_test_data, negative_pretrain_test_data, positive_pretrain_test_label, negative_pretrain_test_label, dim = generate_binary_pretrain_data(args.uci, args.ds)
    pretrain_loader, pretrain_test_loader, pretrain_eval_loader, pretrain_data, pretrain_label = generate_pretrain_loaders(positive_pretrain_data, negative_pretrain_data, positive_pretrain_label, negative_pretrain_label, positive_pretrain_test_data, negative_pretrain_test_data, positive_pretrain_test_label, negative_pretrain_test_label, args.pretrain_bs)
    pretrain_model = get_model(args.ds, args.mo, dim, device)


    avg_pretrain_test_acc, pretrain_model = pretrainLR(pretrain_model, pretrain_loader, pretrain_test_loader, pretrain_eval_loader, args, loss_fn, device, if_write=if_write, save_path=save_pretrain_path)
    print("Average test accuracy for pretrain: ", avg_pretrain_test_acc)

    train_data1, train_data2, train_label1, train_label2, data1_loader, data2_loader, test_loader = train_test_data_gen(positive_pretrain_data, negative_pretrain_data, positive_pretrain_test_data, negative_pretrain_test_data, args.n, args.prior, args.pretrain_bs)

    data1_confidence = torch.zeros(train_data1.shape[0])
    data1_confidence = data1_confidence.to(device)
    data1_confidence, start_idx1 = train_data_confidence_gen(data1_loader, pretrain_model, device, data1_confidence)


    data2_confidence = torch.zeros(train_data2.shape[0])
    data2_confidence = data2_confidence.to(device)
    data2_confidence, start_idx2 = train_data_confidence_gen(data2_loader, pretrain_model, device, data2_confidence)
 

    pcomp_confidence = (data2_confidence - data1_confidence).cpu()
    #print(pcomp_confidence[:10])
    confdiff_train_loader = gen_confdiff_train_loader(train_data1, train_data2, pcomp_confidence, train_label1, train_label2, args.bs)

    model = get_model(args.ds, args.mo, dim, device)
    if args.me == 'ConfDiffUnbiased':
        res_acc = ConfDiffUnbiased(model, confdiff_train_loader, test_loader, args, loss_fn, device, if_write=if_write, save_path=save_detail_path)
        print('ConfDiffUnbiased_acc: ', res_acc)
    elif args.me == 'ConfDiffABS':
        res_acc = ConfDiffABS(model, confdiff_train_loader, test_loader, args, loss_fn, device, if_write=if_write, save_path=save_detail_path)
        print('ConfDiffABS_acc: ', res_acc)   
    elif args.me == 'ConfDiffReLU':
        res_acc = ConfDiffReLU(model, confdiff_train_loader, test_loader, args, loss_fn, device, if_write=if_write, save_path=save_detail_path)
        print('ConfDiffReLU_acc: ', res_acc)
    acc_run_list[run_idx] = res_acc
    print('\n')
    if if_write:
        with open(save_total_path, "a") as f:
            f.writelines("{},{:.6f},None\n".format(run_idx + 1, res_acc))        
    
print('Avg_acc:{}    std_acc:{}'.format(acc_run_list.mean(), acc_run_list.std()))
if if_write:
    with open(save_total_path, "a") as f:
        f.writelines("in total,{:.6f},{:.6f}\n".format(acc_run_list.mean(), acc_run_list.std())) 
print('method:{}    lr:{}    wd:{}'.format(args.me, args.lr, args.wd))
print('loss:{}    prior:{}'.format(args.lo, args.prior))
print('model:{}    dataset:{}'.format(args.mo, args.ds))
print('num of sample:{}'.format(args.n))
print('\n')
