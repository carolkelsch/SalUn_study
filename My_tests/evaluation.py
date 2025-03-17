import numpy as np 
import matplotlib as plt
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
from torchmetrics.multimodal.clip_score import CLIPScore



class EvaluationStats():

    def __init__(self, model, data_loaders, method=None):
        self.methods = ["UnlearnAccuracy", "RemainingAccuracy", "TestingAccuracy", "CLIP-Score", "FID", "MIA", "RunTimeEfficiency"]

        self.model = model
        self.data_loaders = data_loaders
        self.stats = {}

        if method != None:
            self.compute_metric(method)
    
    '''From SalUn'''
    def accuracy(self, output, target, topk=(1,)):
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
    
    def MIA(self, metric):

        '''From SalUn'''
        def entropy(p, dim=-1, keepdim=False):
            return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


        def m_entropy(p, labels, dim=-1, keepdim=False):
            log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
            reverse_prob = 1 - p
            log_reverse_prob = torch.where(
                p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
            )
            modified_probs = p.clone()
            modified_probs[:, labels] = reverse_prob[:, labels]
            modified_log_probs = log_reverse_prob.clone()
            modified_log_probs[:, labels] = log_prob[:, labels]
            return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)


        def collect_prob(data_loader):
            if data_loader is None:
                return torch.zeros([0, 10]), torch.zeros([0])

            prob = []
            targets = []

            self.model.eval()
            with torch.no_grad():
                for batch in data_loader:
                    try:
                        batch = [tensor.to(next(self.model.parameters()).device) for tensor in batch]
                        data, target = batch
                    except:
                        print("UnknownErrorrrr")
                    with torch.no_grad():
                        output = self.model(data)
                        prob.append(F.softmax(output, dim=-1).data)
                        targets.append(target)

            return torch.cat(prob), torch.cat(targets)


        def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
            n_shadow_train = shadow_train.shape[0]
            n_shadow_test = shadow_test.shape[0]
            n_target_train = target_train.shape[0]
            n_target_test = target_test.shape[0]

            X_shadow = (
                torch.cat([shadow_train, shadow_test])
                .cpu()
                .numpy()
                .reshape(n_shadow_train + n_shadow_test, -1)
            )
            Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

            clf = SVC(C=3, gamma="auto", kernel="rbf")
            clf.fit(X_shadow, Y_shadow)

            accs = []

            if n_target_train > 0:
                X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
                acc_train = clf.predict(X_target_train).mean()
                accs.append(acc_train)

            if n_target_test > 0:
                X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
                acc_test = 1 - clf.predict(X_target_test).mean()
                accs.append(acc_test)

            return np.mean(accs)


        def SVC_MIA(shadow_train, target_train, target_test, shadow_test):
            shadow_train_prob, shadow_train_labels = collect_prob(shadow_train)
            shadow_test_prob, shadow_test_labels = collect_prob(shadow_test)

            target_train_prob, target_train_labels = collect_prob(target_train)
            target_test_prob, target_test_labels = collect_prob(target_test)

            shadow_train_corr = (
                torch.argmax(shadow_train_prob, axis=1) == shadow_train_labels
            ).int()
            shadow_test_corr = (
                torch.argmax(shadow_test_prob, axis=1) == shadow_test_labels
            ).int()
            target_train_corr = (
                torch.argmax(target_train_prob, axis=1) == target_train_labels
            ).int()
            target_test_corr = (
                torch.argmax(target_test_prob, axis=1) == target_test_labels
            ).int()

            shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
            shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
            target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None])
            target_test_conf = torch.gather(target_test_prob, 1, target_test_labels[:, None])

            shadow_train_entr = entropy(shadow_train_prob)
            shadow_test_entr = entropy(shadow_test_prob)

            target_train_entr = entropy(target_train_prob)
            target_test_entr = entropy(target_test_prob)

            shadow_train_m_entr = m_entropy(shadow_train_prob, shadow_train_labels)
            shadow_test_m_entr = m_entropy(shadow_test_prob, shadow_test_labels)
            if target_train is not None:
                target_train_m_entr = m_entropy(target_train_prob, target_train_labels)
            else:
                target_train_m_entr = target_train_entr
            if target_test is not None:
                target_test_m_entr = m_entropy(target_test_prob, target_test_labels)
            else:
                target_test_m_entr = target_test_entr

            acc_corr = SVC_fit_predict(
                shadow_train_corr, shadow_test_corr, target_train_corr, target_test_corr
            )
            acc_conf = SVC_fit_predict(
                shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf
            )
            acc_entr = SVC_fit_predict(
                shadow_train_entr, shadow_test_entr, target_train_entr, target_test_entr
            )
            acc_m_entr = SVC_fit_predict(
                shadow_train_m_entr, shadow_test_m_entr, target_train_m_entr, target_test_m_entr
            )
            acc_prob = SVC_fit_predict(
                shadow_train_prob, shadow_test_prob, target_train_prob, target_test_prob
            )
            metricstats = {
                "correctness": acc_corr,
                "confidence": acc_conf,
                "entropy": acc_entr,
                "m_entropy": acc_m_entr,
                "prob": acc_prob,
            }
            
            return metricstats

        self.stats = {metric: SVC_MIA(
            shadow_train=self.data_loaders["train_retain_mia"],
            shadow_test=self.data_loaders["test_retain"],
            target_train=None,
            target_test=self.data_loaders["train_forget"]
            )
        }

    def compute_acc(self, metric, loader):
        losses = np.zeros(len(loader))
        top1 = np.zeros(len(loader))
        
        criterion = torch.nn.CrossEntropyLoss()

        # switch to evaluate mode
        self.model.eval()
        device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

        for i, (image, target) in enumerate(loader):

            image = image.cuda()
            target = target.cuda()

            with torch.no_grad():
                output = self.model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            prec1 = self.accuracy(output.data, target)[0]
            losses[i] = loss.item()
            top1[i] = prec1.item()

            print(f"Test: [{i}/{len(loader)}]\t" + 
            f"Loss {losses[i]:.4f} ({np.mean(losses):.4f})\t" +
            f"Accuracy {top1[i]:.3f} ({np.mean(top1):.3f})")

            print(f"Accuracy {np.mean(top1):.3f}")

        self.stats = {metric: np.mean(top1)}

    def compute_metric(self, method):

        if not isinstance(method, list):
            if not isinstance(method, str):
                print(f"\033[31mInvalid method\r\n Please pick between {self.methods}!\033[0m")
        
        if isinstance(method, str):
            method = [method]
        
        for m in method:
            if m == "UnlearnAccuracy":
                if "forget" in self.data_loaders and self.data_loaders["forget"] is not None:
                    self.compute_acc(m, self.data_loaders["forget"])

            elif m == "RemainingAccuracy":
                if "retain" in self.data_loaders and self.data_loaders["retain"] is not None:
                    self.compute_acc(m, self.data_loaders["retain"])

            elif m == "TestingAccuracy":
                if "test" in self.data_loaders and self.data_loaders["test"] is not None:
                    self.compute_acc(m, self.data_loaders["test"])

            elif m == "MIA":
                if np.isin(np.array(['train_retain_mia', 'test_retain', 'train_forget']), np.array(list(self.data_loaders.keys()))).sum() != 3:
                    print(f"\033[31mCould not find the necessary data loaders for MIA computation.\r\n Please provide 'train_retain_mia', 'test_retain' and 'train_forget'!\033[0m")
                    return None
                
                if self.data_loaders["train_retain_mia"] is None or self.data_loaders["test_retain"] is None or self.data_loaders["train_forget"] is None:
                    print(f"\033[31mCheck data loaders for MIA computation.\r\n It seems like 'train_retain_mia' or 'test_retain' or 'train_forget' are empty!\033[0m")
                    return None

                else:
                    self.MIA(m)


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
                print(f"\033[31mInvalid method\r\n Please pick between {self.methods}!\033[0m")
                return None
    
        return self.stats
        
