import argparse
import os
import random
import numpy as np
import torch
import torch.nn.modules as nn
import data_process_test as data_process
import model
import test_process
from tqdm import tqdm
from transformers import BertTokenizer
from model import ModelParam
from datetime import datetime


def random_seed(seed: int = 2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-run_type', type=int,
                       default=1, help='1: train, 2: debug train, 3: dev, 4: test')
    parse.add_argument('-save_model_path', type=str,
                       default='checkpoint', help='save the good model.pth path')
    parse.add_argument('-add_note', type=str, default='', help='Additional instructions when saving files')
    parse.add_argument('-gpu_num', type=str, default='0', help='gpu index')
    parse.add_argument('-gpu0_bsz', type=int, default=0,
                       help='the first GPU batch size')
    parse.add_argument('-epoch', type=int, default=10, help='train epoch num')
    parse.add_argument('-batch_size', type=int, default=16,
                       help='batch size number')
    parse.add_argument('-acc_grad', type=int, default=1, help='Number of steps to accumulate gradient on '
                                                              '(divide the batch_size and accumulate)')
    parse.add_argument('-lr', type=float, default=2e-5, help='learning rate')
    parse.add_argument('-fuse_lr', type=float, default=1e-3, help='learning rate')
    parse.add_argument('-min_lr', type=float,
                       default=1e-9, help='the minimum lr')
    parse.add_argument('-warmup_step_epoch', type=float,
                       default=2, help='warmup learning step')
    parse.add_argument('-num_workers', type=int, default=0,
                       help='loader dataset thread number')
    parse.add_argument('-l_dropout', type=float, default=0.1,
                       help='classify linear dropout')
    parse.add_argument('-train_log_file_name', type=str,
                       default='train_correct_log.txt', help='save some train log')
    parse.add_argument('-dis_ip', type=str, default='tcp://localhost:23456',
                       help='init_process_group ip and port')
    parse.add_argument('-optim_b1', type=float, default=0.9,
                       help='torch.optim.Adam betas_1')
    parse.add_argument('-optim_b2', type=float, default=0.999,
                       help='torch.optim.Adam betas_1')
    parse.add_argument('-data_path_name', type=str, default='10-flod-1',
                       help='train, dev and test data path name')
    # 混合模式
    parse.add_argument('-data_type', type=str, default='MVSA-multiple',
                       help='Train data type: MVSA-single and MVSA-multiple and HFM')
    parse.add_argument('-word_length', type=int,
                       default=200, help='the sentence\'s word length')
    parse.add_argument('-save_acc', type=float, default=-1, help='The default ACC threshold')
    parse.add_argument('-save_F1', type=float, default=-1, help='The default F1 threshold')
    parse.add_argument('-text_model', type=str, default='bert-base', help='language model')
    parse.add_argument('-loss_type', type=str, default='CE', help='Type of loss function')
    parse.add_argument('-optim', type=str, default='adam', help='Optimizer:adam, sgd, adamw')
    parse.add_argument('-activate_fun', type=str, default='gelu', help='Activation function')

    # 图像模型
    parse.add_argument('-image_model', type=str, default='convnext_base',
                       help='Image model: resnet-18, resnet-34, resnet-50, resnet-101, resnet-152')

    parse.add_argument('-image_size', type=int, default=224, help='Image dim')
    parse.add_argument('-image_output_type', type=str, default='all',
                       help='"all" represents the overall features and regional features of the picture, and "CLS" represents the overall features of the picture')
    parse.add_argument('-text_length_dynamic', type=int, default=1, help='1: Dynamic length; 0: fixed length')
    parse.add_argument('-fuse_type', type=str, default='max', help='att, ave, max')
    parse.add_argument('-tran_dim', type=int, default=768,
                       help='Input dimension of text and picture encoded transformer')
    parse.add_argument('-tran_num_layers', type=int, default=4, help='The layer of transformer')
    parse.add_argument('-image_num_layers', type=int, default=2, help='The layer of image transformer')
    parse.add_argument('-train_fuse_model_epoch', type=int, default=5,
                       help='The number of epoch of the model that only trains the fusion layer')
    parse.add_argument('-cl_loss_alpha', type=int, default=1, help='Weight of contrastive learning loss value')
    parse.add_argument('-cl_self_loss_alpha', type=int, default=1, help='Weight of contrastive learning loss value')
    parse.add_argument('-temperature', type=float, default=0.07,
                       help='Temperature used to calculate contrastive learning loss')

    # 布尔类型的参数
    parse.add_argument('-cuda', action='store_true', default=True,
                       help='if True: use cuda. if False: use cpu')
    parse.add_argument('-fixed_image_model', action='store_true', default=False, help='是否固定图像模型的参数')
    
    # 添加my_type参数
    parse.add_argument("-my_type", type=int, default=0, help='执行模式')

    opt = parse.parse_args()

    random_seed(2023)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_num)
    opt.epoch = opt.train_fuse_model_epoch + opt.epoch

    dt = datetime.now()
    opt.save_model_path = opt.save_model_path + '/' + dt.strftime(
        '%Y-%m-%d-%H-%M-%S') + '-'
    if opt.add_note != '':
        opt.save_model_path += opt.add_note + '-'
    print('\n', opt.save_model_path, '\n')

    assert opt.batch_size % opt.acc_grad == 0
    opt.acc_batch_size = opt.batch_size // opt.acc_grad

    cl_fuse_model = model.CLModel(opt)

    if opt.cuda is True:
        print("cuda")
        assert torch.cuda.is_available()
        if len(opt.gpu_num) > 1:
            if opt.gpu0_bsz > 0:
                cl_fuse_model = torch.nn.DataParallel(cl_fuse_model).cuda()
            else:
                print('multi-gpu')
                print('当前GPU编号：', opt.local_rank)
                torch.cuda.set_device(opt.local_rank)  # 在进行其他操作之前必须先设置这个
                torch.distributed.init_process_group(backend='nccl')
                cl_fuse_model = cl_fuse_model.cuda()
                cl_fuse_model = nn.parallel.DistributedDataParallel(cl_fuse_model, find_unused_parameters=True)
        else:
            cl_fuse_model = cl_fuse_model.cuda()

    print('Init Data Process:')
    tokenizer = None
    abl_path = ''
    if opt.text_model == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased/vocab.txt')

    data_path_root = abl_path + 'dataset/data/'
    test_data_path = data_path_root + 'test.json'
    photo_path = abl_path + 'dataset/data/' + 'image_data'
    image_coordinate = None
    data_translation_path = abl_path + 'dataset/data/' + 'translations_test.json'

    # data_type:标识数据的类型，1是训练数据，2是开发集，3是测试数据
    test_loader, opt.test_data_len = data_process.data_process(opt, test_data_path, tokenizer, photo_path, data_type=3,
                                                               data_translation_path=data_translation_path,
                                                               image_coordinate=image_coordinate)

    print(opt)

    print('\nTest Begin')
    
    # 注意，训练之后需要更改对应的model_path！
    model_path = "./checkpoint/多模态融合模型(预训练+微调)/07-11-19-42-22-Acc-0.75000.pth"
    cl_fuse_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)

    orgin_param = ModelParam()
    
    # 记录结果
    with torch.no_grad():
        cl_fuse_model.eval()
        test_dataset = test_loader.dataset
        outputs = []
        names = []
        for index, data in enumerate(test_loader):
            texts_origin, bert_attention_mask, image_origin, text_image_mask, labels, \
                texts_augment, bert_attention_mask_augment, image_augment, text_image_mask_augment, _, ids = data

            if opt.cuda is True:
                texts_origin = texts_origin.cuda()
                bert_attention_mask = bert_attention_mask.cuda()
                image_origin = image_origin.cuda()
                text_image_mask = text_image_mask.cuda()
                labels = labels.cuda()

            orgin_param.set_data_param(texts=texts_origin, bert_attention_mask=bert_attention_mask, images=image_origin,
                                       text_image_mask=text_image_mask)

            origin_res = cl_fuse_model(orgin_param)
            _, predicted = torch.max(origin_res, 1)
            outputs += predicted.cpu().tolist()
            names += ids

    emotion_labels = {
        0: "positive",
        1: "neutral",
        2: "negative",
    }
    
    with open('result.txt', 'w') as f:
        f.write('guid,tag\n')
        for guid, tag in zip(names, outputs):
            f.write('{},{}\n'.format(guid, emotion_labels[tag]))

