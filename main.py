import os
import sys
import logging
import argparse
from shutil import copyfile
import torch
from Codes import utils
from Codes.network import UNet
from Codes.training import *
from Codes.losses import MSELoss, ConnLoss3Proj
from Codes.dataset import SynthDataset, collate_fn
from Codes import utils
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def main(config_file="main.config"):

    torch.set_num_threads(1)

    __c__ = utils.yaml_read(config_file)
    
    output_path = __c__["output_path"]
    batch_size = __c__["batch_size"]
    ours = __c__["ours"]
    
    utils.mkdir(output_path)
    utils.config_logger(os.path.join(output_path, "main.log"))

    copyfile(config_file, os.path.join(output_path, "main.config"))
    copyfile(__file__, os.path.join(output_path, "main.py"))

    logger.info("Command line: {}".format(' '.join(sys.argv)))

    logger.info("Loading training dataset")
    dataset_training = SynthDataset(train=True, cropSize=__c__["crop_size"], th=__c__["threshold"])
    dataloader_training= DataLoader(dataset_training, batch_size=batch_size, num_workers=4, \
                                    shuffle=True, collate_fn=collate_fn)
    
    logger.info("Done. {} datapoints loaded.".format(len(dataset_training)))

    logger.info("Loading validation dataset")
    dataset_validation = SynthDataset(train=False, th=__c__["threshold"])
    dataloader_validation = DataLoader(dataset_validation, batch_size=1, num_workers=1, \
                                        shuffle=False)
    
    logger.info("Done. {} datapoints loaded.".format(len(dataset_validation)))
    
    training_step = TrainingEpoch(dataloader_training,
                                  ours,
                                  __c__["ours_start"],
                                  __c__["neg_coeff"],
                                  __c__["pos_coeff"])
    
    validation = Validation(tuple(__c__["crop_size_test"]),
                            tuple(__c__["margin_size"]),
                            dataloader_validation,
                            __c__["num_classes"],
                            output_path)

    logger.info("Creating model...")
    network = UNet(in_channels=__c__["in_channels"],
                   m_channels=__c__["m_channels"],
                   out_channels=__c__["num_classes"],
                   n_convs=__c__["n_convs"],
                   n_levels=__c__["n_levels"],
                   dropout=__c__["dropout"],
                   batch_norm=__c__["batch_norm"],
                   upsampling=__c__["upsampling"],
                   pooling=__c__["pooling"],
                   three_dimensional=__c__["three_dimensional"]).cuda()
        
    network.train(True)
    optimizer = torch.optim.Adam(network.parameters(), lr=__c__["lr"],
                                 weight_decay=__c__["weight_decay"])

    if __c__["lr_decay"]:
        lr_lambda = lambda it: 1/(1+it*__c__["lr_decay_factor"])
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        lr_scheduler = None
    
    base_loss = MSELoss().cuda()
    
    if ours:
        our_loss = ConnLoss3Proj(dmax=__c__["threshold"]).cuda()
    else:
        our_loss = None

    logger.info("Running...")

    trainer = Trainer(training_step=lambda iter: training_step(iter, network, optimizer,
                                                                lr_scheduler, base_loss, our_loss),
                         validation   =lambda iter: validation(iter, network, base_loss),
                         valid_every=__c__["valid_every"],
                         print_every=__c__["print_every"],
                         save_every=__c__["save_every"],
                         save_path=output_path,
                         save_objects={"network":network},
                         save_callback=None)

    trainer.run_for(__c__["num_iters"])

if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, default="main.config")

    args = parser.parse_args()

    main(**vars(args))
