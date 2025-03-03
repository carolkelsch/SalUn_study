import numpy as np 
import matplotlib as plt

class Evaluation_Stats():
    def __init__(self, model, data_loders, method=None):
        self.model = model
        self.data_loders = data_loders
        self.stats = {}
        if method != None:
            compute_acc(method)
    
    def get_acc():
        return self.stats
    
    def compute_acc(method):
        if method == "classification":
            return None
        elif method == "generation":
            return None
        else:
            print("\033[31mInvalid method\r\n Please pick between 'classification' or 'generation'!\033[0m")
            return None
    
    def evaluate(loader):
        losses = np.zeros_like(len(loader))
        top1 = np.zeros_like(len(loader))

        # switch to evaluate mode
        self.model.eval()
        device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        for i, (image, target) in enumerate(loader):

            image = image.cuda()
            target = target.cuda()

            # compute output
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(loader), loss=losses, top1=top1
                    )
                )

            print("valid_accuracy {top1.avg:.3f}".format(top1=top1))

        return top1.avg
        
