import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from torch.nn import Parameter
import math
import util


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    
    def forward(self, feature_map):
        return F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)

class ImageClassifier(torch.nn.Module):
    def __init__(self, P):
        super(ImageClassifier, self).__init__()
        
        self.arch = P['arch']
        feature_extractor = torchvision.models.resnet50(pretrained=P['use_pretrained'])
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])
        

        # if P['freeze_feature_extractor']:
        #     for param in feature_extractor.parameters():
        #         param.requires_grad = False
        # else:
        for param in feature_extractor.parameters():
            param.requires_grad = True
        self.feature_extractor = feature_extractor
            
        self.avgpool = GlobalAvgPool2d()

        linear_classifier = torch.nn.Linear(P['feat_dim'], P['num_classes'], bias=True)
        self.linear_classifier = linear_classifier

    # def unfreeze_feature_extractor(self):
    #     for param in self.feature_extractor.parameters():
    #         param.requires_grad = True

    def get_cam(self, x):
        feats = self.feature_extractor(x)
        CAM = F.conv2d(feats, self.linear_classifier.weight.unsqueeze(-1).unsqueeze(-1))
        return CAM

    def foward_linearinit(self, x):
        x = self.linear_classifier(x)
        return x
        
    def forward(self, x):

        feats = self.feature_extractor(x)
        pooled_feats = self.avgpool(feats)
        logits = self.linear_classifier(pooled_feats)
    
        return logits

    def get_config_optim(self, P):
        feature_extractor_params = [param for param in list(self.feature_extractor.parameters()) if
                                    param.requires_grad]
        linear_classifier_params = [param for param in list(self.linear_classifier.parameters()) if
                                    param.requires_grad]
        opt_params = [
            {'params': feature_extractor_params, 'lr': P['lr']},
            {'params': linear_classifier_params, 'lr': P['lr_mult'] * P['lr']}
        ]
        return  opt_params



class ImageClassifier_Matrix(torch.nn.Module):
    def __init__(self, P):
        super(ImageClassifier_Matrix, self).__init__()

        self.arch = P['arch']
        feature_extractor = torchvision.models.resnet50(pretrained=P['use_pretrained'])
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])

        # if P['freeze_feature_extractor']:
        #     for param in feature_extractor.parameters():
        #         param.requires_grad = False
        for param in feature_extractor.parameters():
            param.requires_grad = True

        self.feature_extractor = feature_extractor

        self.avgpool = GlobalAvgPool2d()

        linear_classifier = torch.nn.Linear(P['feat_dim'], P['num_classes'], bias=True)
        self.linear_classifier = linear_classifier

        # self.w_matrix = torch.nn.Parameter(torch.zeros([P['num_classes'], P['num_classes']])).double().requires_grad_()
        # self.w_matrix = torch.nn.Parameter(torch.randn([P['num_classes'], P['num_classes']]).cuda())
        # self.apply = False

    # def set_w_matrix(self, matrix):
    #     self.w_matrix = torch.nn.Parameter(matrix, requires_grad=True)
    #     self.apply = True


    # def unfreeze_feature_extractor(self):
    #     for param in self.feature_extractor.parameters():
    #         param.requires_grad = True

    def get_cam(self, x):
        feats = self.feature_extractor(x)
        CAM = F.conv2d(feats, self.linear_classifier.weight.unsqueeze(-1).unsqueeze(-1))
        return CAM

    def foward_linearinit(self, x):
        x = self.linear_classifier(x)
        return x

    def forward(self, x):

        feats = self.feature_extractor(x)
        pooled_feats = self.avgpool(feats)
        logits = self.linear_classifier(pooled_feats)
        # if self.apply:
        #     logits = torch.matmul(logits,  self.w_matrix.T)

        return logits


def inverse_sigmoid(p):
    epsilon = 1e-5
    p = np.minimum(p, 1 - epsilon)
    p = np.maximum(p, epsilon)
    return np.log(p / (1-p))

class LabelEstimator(torch.nn.Module):

    def __init__(self, P, observed_label_matrix, estimated_labels):

        super(LabelEstimator, self).__init__()
        # print('initializing label estimator')

        # Note: observed_label_matrix is assumed to have values in {-1, 0, 1} indicating
        # observed negative, unknown, and observed positive labels, resp.

        num_examples = int(np.shape(observed_label_matrix)[0])
        observed_label_matrix = np.array(observed_label_matrix).astype(np.int8)
        total_pos = np.sum(observed_label_matrix == 1)
        total_neg = np.sum(observed_label_matrix == -1)
        # print('observed positives: {} total, {:.1f} per example on average'.format(total_pos, total_pos / num_examples))
        # print('observed negatives: {} total, {:.1f} per example on average'.format(total_neg, total_neg / num_examples))

        if estimated_labels is None:
            # initialize unobserved labels:
            w = 0.1
            q = inverse_sigmoid(0.5 + w)
            param_mtx = q * (2 * torch.rand(num_examples, P['num_classes']) - 1)

            # initialize observed positive labels:
            init_logit_pos = inverse_sigmoid(0.995)
            idx_pos = torch.from_numpy((observed_label_matrix == 1).astype(np.bool))
            param_mtx[idx_pos] = init_logit_pos

            # initialize observed negative labels:
            init_logit_neg = inverse_sigmoid(0.005)
            idx_neg = torch.from_numpy((observed_label_matrix == -1).astype(np.bool))
            param_mtx[idx_neg] = init_logit_neg
        else:
            param_mtx = inverse_sigmoid(torch.FloatTensor(estimated_labels))

        self.logits = torch.nn.Parameter(param_mtx)

    def get_estimated_labels(self):
        with torch.set_grad_enabled(False):
            estimated_labels = torch.sigmoid(self.logits)
        estimated_labels = estimated_labels.clone().detach().cpu().numpy()
        return estimated_labels

    def forward(self, indices):
        x = self.logits[indices, :]
        x = torch.sigmoid(x)
        return x


class MultilabelModel(torch.nn.Module):
    def __init__(self, P, observed_label_matrix, estimated_labels=None):
        super(MultilabelModel, self).__init__()

        self.f = ImageClassifier(P)
        self.g = LabelEstimator(P, observed_label_matrix, estimated_labels)

    def forward(self, batch):
        f_logits = self.f(batch['image'])
        g_preds = self.g(batch['idx'])  # oops, we had a sigmoid here in addition to
        return (f_logits, g_preds)

    def get_config_optim(self, P):
        feature_extractor_params = [param for param in list(self.f.feature_extractor.parameters()) if
                                    param.requires_grad]
        linear_classifier_params = [param for param in list(self.f.linear_classifier.parameters()) if
                                    param.requires_grad]
        label_estimator_params = [param for param in list(self.g.parameters()) if param.requires_grad]
        opt_params = [
            {'params': feature_extractor_params, 'lr': P['lr']},
            {'params': linear_classifier_params, 'lr': P['lr_mult'] * P['lr']},  ## ROLE은 lr_mult없게 구현되긴 함
            {'params': label_estimator_params, 'lr': P['lr_mult'] * P['lr']}
        ]
        return opt_params


class ConstrainedFFNNModel(nn.Module):
    """ C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss """

    def __init__(self, P):
        super(ConstrainedFFNNModel, self).__init__()
        input_dim, hidden_dim, output_dim = P['input_dim'], P['hidden_dim'], P['num_classes']
        self.nb_layers = P['num_layers']
        self.R = P['R']

        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers - 1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)

        self.drop = nn.Dropout(P['dropout'])

        self.sigmoid = nn.Sigmoid()
        if P['non_lin'] == 'tanh':
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()

    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers - 1:
                # x = self.sigmoid(self.fc[i](x))
                x = self.fc[i](x)
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        # if self.training:
        #     constrained_out = x
        # else:
        #     x = self.sigmoid(x)
        #     constrained_out = util.get_constr_out(x, self.R)
        # return constrained_out
        return x

    def get_config_optim(self, P):
        return [{'params': self.parameters(), 'lr':P['lr']}]
