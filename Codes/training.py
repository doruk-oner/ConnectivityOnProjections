import time
import numpy as np
import os
import logging
from . import utils
from skimage.morphology import skeletonize_3d
from .scores import correctness_completeness_quality

logger = logging.getLogger(__name__)

class Trainer(object):

    def __init__(self,training_step, validation=None, valid_every=None,
                 print_every=None, save_every=None, save_path=None, save_objects={},
                 save_callback=None):

        self.training_step = training_step
        self.validation = validation
        self.valid_every = valid_every
        self.print_every = print_every
        self.save_every = save_every
        self.save_path = save_path
        self.save_objects = save_objects
        self.save_callback = save_callback

        self.starting_iteration = 0

        if self.save_path is None or self.save_path in ["", ".", "./"]:
            self.save_path = os.getcwd()

        self.results = {"training":{"epochs":[], "results":[]},
                        "validation": {"epochs":[], "results":[]}}

    def save_state(self, iteration):

        for name, object in self.save_objects.items():
            utils.torch_save(os.path.join(self.save_path, "{}_final.pickle".format(name, iteration)),
                             object.state_dict())

        utils.pickle_write(os.path.join(self.save_path, "results_final.pickle".format(iteration)),
                           self.results)

        if self.save_callback is not None:
            self.save_callback(iteration)

    def run_for(self, iterations):

        start_time = time.time()
        block_iter_start_time = time.time()

        for iteration in range(self.starting_iteration, self.starting_iteration+iterations+1):

            train_step_results = self.training_step(iteration)
            self.results['training']["results"].append(train_step_results)
            self.results['training']["epochs"].append(iteration)

            if self.print_every is not None:
                if iteration%self.print_every==0:
                    elapsed_time = (time.time() - start_time)//60
                    block_iter_elapsed_time = time.time() - block_iter_start_time

                    loss_v1 = train_step_results["loss"] if "loss" in train_step_results.keys() else None
                    loss_v2 = train_step_results["loss_2"] if "loss_2" in train_step_results.keys() else 0
                    to_print = "[{:0.0f}min][{:0.2f}s] - Epoch: {} (Train-batch Loss: {:0.6f}, {:0.6f})"
                    to_print = to_print.format(elapsed_time, block_iter_elapsed_time, iteration, loss_v1, loss_v2)

                    logger.info(to_print)
                    block_iter_start_time = time.time()

            if self.validation is not None and self.valid_every is not None:
                if iteration%self.valid_every==0 and iteration!=self.starting_iteration:
                    logger.info("validation...")
                    start_valid = time.time()

                    validation_results = self.validation(iteration)

                    self.results['validation']["results"].append(validation_results)
                    self.results['validation']["epochs"].append(iteration)
                    logger.info("Validation time: {:.2f}s".format(time.time()-start_valid))

            if self.save_every is not None:
                if iteration%self.save_every==0 and iteration!=self.starting_iteration:
                    start_saving = time.time()
                    self.save_state(iteration)
                    logger.info("Saving time: {:.2f}s".format(time.time()-start_saving))

class TrainingEpoch(object):

    def __init__(self, dataloader, ours=False, ours_start=0, neg_coeff=1e-2, pos_coeff=1e-4):

        self.dataloader = dataloader
        self.ours = ours
        self.ours_start = ours_start
        self.neg_coeff = neg_coeff
        self.pos_coeff = pos_coeff

    def __call__(self, iterations, network, optimizer, lr_scheduler, base_loss, our_loss):
        
        mean_loss = 0
        mean_loss2 = 0
        for images, labels, graphs, slices in self.dataloader:

            images = images.cuda()
            labels = labels.cuda()
            
            preds = network(images.contiguous())
            
            loss = base_loss(preds, labels)
            loss_v = float(utils.from_torch(loss))
            loss_v2 = float(0)
            
            if self.ours and iterations >= self.ours_start:
                loss2 = our_loss(preds, labels, self.neg_coeff, self.pos_coeff)
                loss_v2 = float(utils.from_torch(loss2))
                loss += loss2

            if np.isnan(loss_v) or np.isinf(loss_v):
                return {"loss": loss_v,
                        "pred": utils.from_torch(preds),
                        "labels": utils.from_torch(labels)}
            
            mean_loss += loss_v
            mean_loss2 += loss_v2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

        return {"loss"   : float(mean_loss/len(self.dataloader)),
                "loss_2" : float(mean_loss2/len(self.dataloader))}
    
    
class Validation(object):

    def __init__(self, crop_size, margin_size, dataloader_val, out_channels, output_path):
        self.crop_size = crop_size
        self.margin_size = margin_size
        self.dataloader_val = dataloader_val
        self.out_channels = out_channels
        self.output_path = output_path

        self.bestqual = 0
        self.bestcomp = 0
        self.bestcorr = 0

    def __call__(self, iteration, network, loss_function):

        losses = []
        preds = []
        scores = {"corr":[], "comp":[], "qual":[]}

        network.train(False)
        with utils.torch_no_grad:
            for i, (image, label) in enumerate(self.dataloader_val):
        
                image  = image.cuda()[:,None]
                label  = label.cuda()[:,None]

                out_shape = (image.shape[0],self.out_channels,*image.shape[2:])
                pred = utils.to_torch(np.empty(out_shape, np.float32), volatile=True).cuda()
                pred = utils.process_in_chuncks(image, pred,
                                            lambda chunk: network(chunk),
                                            self.crop_size, self.margin_size)

                loss = loss_function(pred, label)
                loss_v = float(utils.from_torch(loss))
                losses.append(loss_v)

                pred_np = utils.from_torch(pred)[0]
                preds.append(pred_np)
                label_np = utils.from_torch(label)[0]
                
                pred_mask = skeletonize_3d((pred_np < 5)[0])//255
                label_mask = (label_np==0)

                corr, comp, qual = correctness_completeness_quality(pred_mask, label_mask, slack=3)
                
                scores["corr"].append(corr)
                scores["comp"].append(comp)
                scores["qual"].append(qual)
                
                # save the prediction here
                output_valid = os.path.join(self.output_path, "output_valid")
                utils.mkdir(output_valid)
                np.save(os.path.join(output_valid, "pred_{:06d}_final.npy".format(i,iteration)), pred_np)

        scores["qual"] = np.nan_to_num(scores["qual"])
        
        qual_total = np.mean(scores["qual"],axis=0)
        corr_total = np.mean(scores["corr"],axis=0)
        comp_total = np.mean(scores["comp"],axis=0)

        if self.bestqual < qual_total:
            self.bestqual = qual_total
            for i in range(len(self.dataloader_val)):
                np.save(os.path.join(output_valid, "pred_{:06d}_bestqual.npy".format(i,iteration)), preds[i])
            utils.torch_save(os.path.join(self.output_path, "network_bestqual.pickle"),
                             network.state_dict())
        

        logger.info("\tMean loss: {}".format(np.mean(losses)))
        logger.info("\tMean qual: {:0.3f}".format(qual_total))
        logger.info("Best quality score is {}".format(self.bestqual))
        
        network.train(True)

        return {"loss": np.mean(losses),
                "mean_corr": corr_total,
                "mean_comp": comp_total,
                "mean_qual": qual_total,
                "scores": scores}
    
    
    
