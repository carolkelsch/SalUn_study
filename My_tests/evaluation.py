import numpy as np 
import matplotlib as plt
import torch
from FromSalUn import *
from torchmetrics.multimodal.clip_score import CLIPScore



class EvaluationStats():

    methods = ["UnlearnAccuracy", "RemainingAccuracy", "TestingAccuracy", "CLIP-Score", "FID", "RunTimeEfficiency"]

    def __init__(self, model, data_loders, method=None):
        self.model = model
        self.data_loders = data_loders
        self.stats = {}
        if method != None:
            compute_acc(method)

    def get_acc():
        return self.stats
    
    '''From SalUn'''
    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    '''From SalUn'''
    def MIA(metric):
        shadow_train = torch.utils.data.Subset(self.data_loader["retain"], list(range(len(self.data_loader["test"]))))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        self.stats = {metric: SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=self.data_loader["test"],
            target_train=None,
            target_test=self.data_loader["forget"],
            model=model,
            )
        }

    def compute_acc(metric, loader):
        losses = np.zeros_like(len(loader))
        top1 = np.zeros_like(len(loader))

        # switch to evaluate mode
        self.model.eval()
        device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

        for i, (image, target) in enumerate(loader):

            image = image.cuda()
            target = target.cuda()

            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            prec1 = accuracy(output.data, target)[0]
            losses[i] = loss.item()
            top1[i] = prec1.item()

            if i % args.print_freq == 0:
                print(f"Test: [{i}/{len(loader)}]\t" + 
                f"Loss {loss[i]:.4f} ({np.mean(loss):.4f})\t" +
                f"Accuracy {top1[i]:.3f} ({np.mean(top1):.3f})")

            print(f"Accuracy {np.mean(top1):.3f}")

        self.stats = {metric: np.mean(top1)}

    def compute_metric(method):

        if not isinstance(method, list):
            if not isinstance(method, str):
                print(f"\033[31mInvalid method\r\n Please pick between {methods}!\033[0m")
        
        for m in method:
            if m == "UnlearnAccuracy":
                if "forget" in self.data_loders and self.data_loaders["forget"] is not None:
                    compute_acc(m, self.data_loders["forget"])

            elif m == "RemainingAccuracy":
                if "retain" in self.data_loders and self.data_loaders["retain"] is not None:
                    compute_acc(m, self.data_loders["retain"])

            elif m == "TestingAccuracy":
                if "test" in self.data_loders and self.data_loaders["test"] is not None:
                    compute_acc(m, self.data_loders["test"])

            elif m == "MIA":
                if "forget" in self.data_loders and self.data_loaders["forget"] is not None:
                    MIA("MIA")

            elif m == "RunTimeEfficiency":
                # TODO: check how to implement run time efficiency, probably in the implementation of the unlearning method
                return None
            
            elif m == "CLIP-Score":
                '''Only makes sense for generation models: measures the accuracy of a text describing an image 
                or an image generated from a text description'''

                #CLIPScore

                return None

            elif m == "FID":
                '''Only makes sense for generation models: measures the distance of two distributions,
                in this case the distance between two images distributions'''
                return None

            else:
                print(f"\033[31mInvalid method\r\n Please pick between {methods}!\033[0m")
                return None
    
        return self.stats
        
