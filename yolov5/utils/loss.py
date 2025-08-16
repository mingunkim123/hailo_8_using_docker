import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QFocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ComputeLoss:
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device
        h = model.hyp
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))
        g = h.get('fl_gamma', 0.0)
        if g and g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        det = de_parallel(model).model[-1]
        self.balance = {3: [4.0, 1.0, 0.4],
                        4: [4.0, 1.0, 0.25, 0.06],
                        5: [4.0, 1.0, 0.25, 0.06, 0.02]}.get(det.nl, [4.0, 1.0, 0.4])
        self.ssi = list(det.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in ('na', 'nc', 'nl', 'anchors'):
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)
        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 4], device=device)
            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sid = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sid], a[sid], gj[sid], gi[sid], score_iou[sid]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[torch.arange(n, device=device), tcls[i].long()] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / max(obji.detach().item(), 1e-9)
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        g = 0.5
        off = torch.tensor([[0, 0],
                            [1, 0],
                            [0, 1],
                            [-1, 0],
                            [0, -1]], device=targets.device).float() * g
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape, device=targets.device)[[3, 2, 3, 2]]
            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                t = t[j]
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j1, k1 = ((gxy % 1 < g) & (gxy > 1)).T
                l1, m1 = ((gxi % 1 < g) & (gxi > 1)).T
                jmask = torch.stack((torch.ones_like(j1), j1, k1, l1, m1))
                t = t.repeat((5, 1, 1))[jmask]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[jmask]
            else:
                t = targets[0]
                offsets = 0
            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            a = t[:, 6].long()
            gj = torch.clamp(gj, 0, int(gain[3].item()) - 1)
            gi = torch.clamp(gi, 0, int(gain[2].item()) - 1)
            indices.append((b, a, gj, gi))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
        return tcls, tbox, indices, anch
