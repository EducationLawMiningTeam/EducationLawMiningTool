import os
import numpy as np
import torch


def save_info_0(dataset,
        hidden_num,
        learning_rate,
        epochs,
        batch_size,
        seed,
        cv_num,
        q_num,
        length,
        time_spend,
        d_model,
        nhead,
        num_encoder_layers,
        dropout,
        gpu
        ,
        speed_cate,loss_rate
        ,info_file):

        params_list = (
            'dataset = %s\n' % dataset,
            'learning_rate = %f\n' % learning_rate,
            'length = %d\n' % length,
            'batch_size = %d\n' % batch_size,
            'seed = %d\n' % seed,
            'q_num = %d\n' % q_num,

            'length = %s\n' % length,
            'time_spend = %d\n' % time_spend,
            'd_model = %f\n' % d_model,
            'nhead = %d\n' % nhead,
            'num_encoder_layers = %d\n' % num_encoder_layers,
            'dropout = %f\n' % dropout,
            'speed_cate = %f\n' % speed_cate,
            'loss_rate = %f\n' % loss_rate           
            
        )
        info_file.write('%s%s%s%s%s%s%s%s%s%s%s%s%s%s' % params_list)



def save_info_1(cv,epoch,max_auc,valid_auc,valid_loss,valid_mse,valid_mae,valid_acc,train_auc,train_loss, train_mse,train_mae,train_acc,time_end,time_start,info_file,info_file1):
    print_list = (
        'cv:%-3d' % cv,
        'epoch:%-3d' % epoch,
        'max_auc:%-8.4f' % max_auc,
        'valid_auc:%-8.4f' % valid_auc,
        'valid_loss:%-8.4f' % valid_loss,
        'valid_mse:%-8.4f' % valid_mse,
        'valid_mae:%-8.4f' % valid_mae,
        'valid_acc:%-8.8f\n' % valid_acc,
        'train_auc:%-8.4f' % train_auc,
        'train_loss:%-8.4f' % train_loss,
        'train_mse:%-8.4f' % train_mse,
        'train_mae:%-8.4f' % train_mae,
        'train_acc:%-8.8f' % train_acc,
        'time:%-6.2fs' % (time_end - time_start)
    )            
    print_list1 = (
        'cv:%d' % cv,
        'epoch:%d' % epoch,
        'max_auc:%f' % max_auc,
        'valid_auc:%f' % valid_auc,
        'valid_loss:%f' % valid_loss,
        'valid_mse:%f' % valid_mse,
        'valid_mae:%f' % valid_mae,
        'valid_acc:%f\n' % valid_acc,
        'train_auc:%f' % train_auc,
        'train_loss:%f' % train_loss,
        'train_mse:%f' % train_mse,
        'train_mae:%f' % train_mae,
        'train_acc:%f' % train_acc,
        'time:%fs' % (time_end - time_start)
    )
    print('%s %s %s %s %s %s %s %s %s %s %s %s %s %s' % print_list)

    info_file.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % print_list)
    info_file1.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % print_list1)


# def save_info_2(cv,train_auc, test_auc, test_mse, test_mae,test_acc,test_loss,info_file):
#     print_list_test = (
#         'cv:%-3d' % cv,
#         'train_auc:%-8.4f' % train_auc,
#         'test_auc:%-8.4f' % test_auc,
#         'test_mse:%-8.4f' % test_mse,
#         'test_mae:%-8.4f' % test_mae,
#         'test_acc:%-8.8f' % test_acc,
#         'test_loss:%-8.4f' % test_loss
#     )

#     print('%s %s %s %s %s %s %s\n' % print_list_test)
#     info_file.write('%s %s %s %s %s %s %s\n' % print_list_test)

def save_info_2(cv,valid_auc, test_auc, test_mse, test_mae,test_acc,test_loss,info_file):
    print_list_test = (
        'cv:%-3d' % cv,
        'valid_auc:%-8.4f' % valid_auc,
        'test_auc:%-8.4f' % test_auc,
        'test_acc:%-8.8f' % test_acc,
        'test_mse:%-8.4f' % test_mse,
        'test_mae:%-8.4f' % test_mae,
        'test_loss:%-8.4f' % test_loss
    )

    print('%s %s %s %s %s %s %s\n' % print_list_test)
    info_file.write('%s %s %s %s %s %s %s\n' % print_list_test)

def save_info_4(cv,valid_auc, test_auc, test_mse, test_mae,test_acc,test_loss,info_file):
    print_list_test = (
        'cv:%-3d' % cv,
        'valid_auc:%-8.4f' % valid_auc,
        'test_auc:%-8.4f' % test_auc,
        'test_mse:%-8.4f' % test_mse,
        'test_mae:%-8.4f' % test_mae,
        'test_acc:%-8.8f' % test_acc,
        'test_loss:%-8.4f' % test_loss
    )

    print(' %s %s %s %s %s %s %s\n' % print_list_test)
    info_file.write(' %s %s %s %s %s %s %s\n' % print_list_test)


def save_info_3(average_train_auc, average_test_auc,average_test_mse, average_test_mae, average_test_acc,average_test_loss,info_file):
    print_result = (
    'valid_auc:%-8.4f' % average_train_auc,
    'test_auc:%-8.4f' % average_test_auc,
    'test_acc:%-8.8f' % average_test_acc,
    'test_mse:%-8.4f' % average_test_mse,
    'mae:%-8.4f' % average_test_mae,
    'test_loss:%-8.4f' % average_test_loss
    )
    print('%s %s %s %s %s %s\n' % print_result)
    info_file.write('%s %s %s %s %s %s\n' % print_result)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='   .pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss