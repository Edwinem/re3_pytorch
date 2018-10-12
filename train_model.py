#Python modules
import argparse
import logging
import json
import warnings
import os
import gc

#Pytorch modules
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim

import numpy as np
from datasets import ALOVDataset
import model
import pytorch_utils as util

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.INFO)


transform = transforms.Compose([util.Normalize(), util.ToTensor()])


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = util.RunningAverage()



    for i,data in enumerate(dataloader):
        optimizer.zero_grad()

        x1, x2, y = data['previmg'], data['currimg'], data['currbb']
        if params.cuda:
            x1, x2, y = Variable(x1.cuda()), Variable(x2.cuda()), Variable(y.cuda(), requires_grad=False)


        output = model(x1, x2)
        loss = loss_fn(output, y)

        loss.backward(retain_graph=True)

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output = output.data.cpu().numpy()
            # compute all metrics on this batch
            summary_batch = {}
            summary_batch['loss'] = loss.data[0]
            summ.append(summary_batch)
            logging.info('- Loss for iteration {} is {}'.format(i,loss.data[0]))

        # update the average loss
        loss_avg.update(loss.data[0])
        if i==100:
            break


        # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        util.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        # val_acc = val_metrics['accuracy']
        # is_best = val_acc>=best_val_acc
        #
        # # Save weights
        util.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                                is_best=False,
                               checkpoint=model_dir)
        #
        # # If best_eval, best_save_path
        # if is_best:
        #     logging.info("- Found new best accuracy")
        #     best_val_acc = val_acc
        #
        #     # Save best val metrics in a json file in the model directory
        #     best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
        #     util.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        util.save_dict_to_json(val_metrics, last_json_path)


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    dataset = dataloader.dataset

    total_loss = 0

    # compute metrics over the dataset
    for i in range(64):

        sample = dataset[i]
        sample['currimg'] = sample['currimg'][None, :, :, :]
        sample['previmg'] = sample['previmg'][None, :, :, :]
        x1, x2 = sample['previmg'], sample['currimg']
        y = sample['currbb']

        # move to GPU if available
        if params.cuda:
            x1 = Variable(x1.cuda())
            x2 = Variable(x2.cuda())
            y = Variable(y.cuda(), requires_grad=False)

        # compute model output
        output = model(x1, x2)
        loss = loss_fn(output[0], y)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output = output.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = dict()
        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean


if __name__ == '__main__':

    default_json = json.dumps([{"learning_rate": 1e-3,
                                "batch_size": 1,
                                "num_epochs": 100,
                                "dropout_rate": 0.8,
                                "num_channels": 32,
                                "save_summary_steps": 100,
                                "num_workers": 4}])

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default=None, help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'

    args = parser.parse_args()

    params=util.Params(default_json)

    model_dir=0
    if args.model_dir :
        params=util.Params()
        params.update(args.model_dir)
        model_dir=args.model_dir
    else:
        model_dir_path=os.path.join(".","model")
        if  not os.path.isdir(model_dir_path):
            os.mkdir(model_dir_path)
        model_dir=model_dir_path


    params.cuda = torch.cuda.is_available()


    alov = ALOVDataset('/large_storage/imagedata++', '/large_storage/alov300++_rectangleAnnotation_full', transform)

    dataloader = DataLoader(alov, batch_size=params.batch_size)

    use_gpu = torch.cuda.is_available()

    model = model.Re3Net().cuda() if use_gpu else model.Re3Net()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    net = 0

    loss_fn=model.loss_fn(params.cuda)


    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, dataloader, dataloader, optimizer, loss_fn, 0, params, model_dir,
                       args.restore_file)
