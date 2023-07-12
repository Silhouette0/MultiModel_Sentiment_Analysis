from util.write_file import WriteFile
from datetime import datetime
import torch
import os


def compare_to_save(last_value, now_value, opt, bert_model, train_log, dev_log, compare_target, threshold, add_new_note=None, last_model_name=None, add_enter=True):
    is_save_model = False
    set_threshold = threshold
    model_name = '' if add_new_note is None else last_model_name
    if last_value < now_value:
        if now_value > set_threshold:
            if add_new_note is not None:
                f = open(opt.save_model_path + '/' + model_name + '.txt', 'r+', encoding='utf-8')
                content = f.read()
                f.close()
                WriteFile(
                    opt.save_model_path, model_name + '-' + compare_target + '.txt', '这是依据%s标准进行保存的' % compare_target + '\n' + content, 'w')
                os.remove(opt.save_model_path + '/' + model_name + '.txt')
                save_content = '**%s高于上次 %.6f, 本次为了 %.6f, 已经存储模型为 %s' % (
                    compare_target, last_value, now_value, opt.save_model_path + '/' + model_name + '.pth')
                is_save_model = True
            else:
                dt = datetime.now()
                model_name = dt.strftime(
                    '%m-%d-%H-%M-%S') + '-' + compare_target + '-' + str('%.5f' % now_value)
                # a = bert_model.state_dict()
                torch.save(bert_model.state_dict(),
                           opt.save_model_path + '/' + model_name + '.pth')
                save_content = '这是依据%s标准进行保存的' \
                               '\nopt: %s \nepoch: %s ' \
                               '\ntrain_loss: %s \ntrain_accuracy: %s ' \
                               '\ntrain_F1_weighted: %s \ntrain_precision_weighted: %s ' \
                               '\ntrain_R_weighted: %s \ntrain_F1: %s' \
                               '\ntrain_R: %s \ntrain_precision: %s' \
                               '\ndev_loss: %s \ndev_accuracy: %s' \
                               '\ndev_F1_weighted: %s \ndev_precision_weighted: %s ' \
                               '\ndev_R_weighted: %s \ndev_F1: %s' \
                               '\ndev_R: %s \ndev_:precision %s \n' % \
                               (compare_target, str(opt), str(train_log['epoch']),
                                str(train_log['run_loss']), str(train_log['train_accuracy']),
                                str(train_log['train_F1_weighted']), str(train_log['train_precision_weighted']),
                                str(train_log['train_R_weighted']), str(train_log['train_F1']),
                                str(train_log['train_R']),str(train_log['train_precision']),
                                str(dev_log['dev_loss']), str(dev_log['dev_accuracy']),
                                str(dev_log['dev_F1_weighted']), str(dev_log['dev_precision_weighted']),
                                str(dev_log['dev_R_weighted']), str(dev_log['dev_F1']),
                                str(dev_log['dev_R']), str(dev_log['dev_precision']))

                WriteFile(
                    opt.save_model_path, model_name + '.txt', save_content, 'w')

                save_content = '**%s高于上次 %.6f, 本次为了 %.6f, 已经存储模型为 %s' % (
                    compare_target, last_value, now_value, opt.save_model_path + '/' + model_name + '.pth')
                is_save_model = True
        else:
            save_content = '**%s高于上次 %.6f, 本次为了 %.6f, 但低于%.6f, 不存储' % (
                compare_target, last_value, now_value, set_threshold)
        if add_enter is True:
            WriteFile(
                opt.save_model_path, opt.train_log_file_name, save_content + '\n\n', 'a+')
        else:
            WriteFile(
                opt.save_model_path, opt.train_log_file_name, save_content + '\n', 'a+')
        print(save_content, '\n')
        return now_value, is_save_model, model_name
    else:
        save_content = '%s低于上次 %.6f, 本次为了 %.6f' % (compare_target, last_value, now_value)
        if add_enter is True:
            WriteFile(
                opt.save_model_path, opt.train_log_file_name, save_content + '\n\n', 'a+')
        else:
            WriteFile(
                opt.save_model_path, opt.train_log_file_name, save_content + '\n', 'a+')
        print(save_content, '\n')
        return last_value, is_save_model, model_name
