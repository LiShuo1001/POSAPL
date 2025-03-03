import os
import re
import json
import torch
import random
import datetime
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from torchvision.transforms import InterpolationMode
from posapl_0_randaugment import RandomAugment
from posapl_0_utils import MetricLogger, SmoothedValue
from posapl_1_model import CreateModel


class TrainDataset(Dataset):

    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
                pass
            pass
        pass

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.pre_caption(ann['caption'], self.max_words)
        label = torch.tensor(ann['label'])

        pos = ann["pos"]
        if len(pos) == 0:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        return image, caption, self.img_ids[ann['image_id']], label, self.deal_pos(pos)

    @staticmethod
    def pre_caption(caption, max_words):
        caption = re.sub(r"([,.'!?\"()*#:;~])", '', caption.lower(),).replace(
            '-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(r"\s{2,}", ' ', caption)
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])
        if not len(caption):
            raise ValueError("pre_caption yields invalid text")
        return caption

    @staticmethod
    def deal_pos(pos):
        pos_str = " ".join(pos)
        return pos_str

    pass


class EvalDataset(Dataset):

    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.poss = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(TrainDataset.pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
            for i, pos in enumerate(ann['pos']):
                self.poss.append(TrainDataset.deal_pos(pos))
                pass
            pass
        pass

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

    pass


class CreateDataset(object):

    def __init__(self):
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                              (0.26862954, 0.26130258, 0.27577711))
        self.pretrain_transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_res, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(), self.normalize])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_res, scale=(0.5, 1.0),
                                         interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(), self.normalize])
        self.test_transform = transforms.Compose([transforms.Resize((config.image_res, config.image_res),
                                                                    interpolation=InterpolationMode.BICUBIC),
                                                  transforms.ToTensor(), self.normalize])

        self.train_dataset = TrainDataset(config.train_file, self.train_transform, config.image_root)
        self.test_dataset = EvalDataset(config.test_file, self.test_transform, config.image_root)
        self.val_dataset = EvalDataset(config.val_file, self.test_transform, config.image_root)

        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size_train,
                                       num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size_test,
                                      num_workers=4, pin_memory=True, shuffle=False, drop_last=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size_test,
                                     num_workers=4, pin_memory=True, shuffle=False, drop_last=False)
        pass

    pass


class POSAPLModel(nn.Module):

    def __init__(self):
        super().__init__()
        create_model = CreateModel(config=config)
        self.tokenize = create_model.tokenize
        self.model = create_model.create_model("ViT-B/32", pretrained="clip",
                                               pretrain_path=config.pretrain_path_open_clip)
        pass

    def get_vis_emb(self, image):
        img_emb = self.model.encode_image(image, normalize=True)
        return img_emb

    def get_txt_emb(self, text_ids, pos=None):
        txt_emb = self.model.encode_text(text_ids, pos=pos, normalize=True)
        return txt_emb

    def forward(self, image, text_ids, pos=None, idx=None, label=None):
        img_emb = self.get_vis_emb(image)
        txt_emb = self.get_txt_emb(text_ids, pos)
        loss_contrastive = self.get_contrastive_loss(img_emb, txt_emb, idx=idx)
        loss_triplet = self.get_triplet_loss(img_emb, txt_emb)
        return loss_contrastive, loss_triplet

    def get_contrastive_loss(self, image_feat, text_feat, idx=None):
        logits = image_feat @ text_feat.t()
        logits = self.model.logit_scale *logits
        if idx is None:
            labels = torch.arange(image_feat.shape[0], device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
        else:
            idx = idx.view(-1, 1)
            pos_idx = torch.eq(idx, idx.t()).float()
            labels = pos_idx / pos_idx.sum(dim=1, keepdim=True)
            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()
        return (loss_i2t + loss_t2i) / 2.0

    def get_triplet_loss(self, image_feat, text_feat, margin=0.1, gamma=2.0):
        _scores = image_feat @ text_feat.t()
        _diagonal = _scores.diag().view(image_feat.shape[0], 1)

        def get_cost(diagonal, scores):
            cost = (margin + scores - diagonal.expand_as(scores)).clamp(min=0)
            cost = cost.masked_fill_(Variable(torch.eye(scores.size(0)) > .5).to(scores.device), 0)
            return (cost ** gamma).sum()

        sum_cost_s = get_cost(_diagonal, _scores)
        sum_cost_im = get_cost(_diagonal.t(), _scores)
        return (sum_cost_s + sum_cost_im) / 2.0

    pass


class Runner(object):

    def __init__(self):
        self.device = torch.device(config.device)

        self.model = POSAPLModel()
        self.model = self.model.to(self.device)

        self.set_trainable(self.model)

        self.create_dataset = CreateDataset()

        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr,
                               weight_decay=config.weight_decay, eps=1e-8, betas=(0.9, 0.98))
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, config.epochs * len(self.create_dataset.train_loader))
        pass

    @staticmethod
    def set_trainable(model):
        for name, module in model.named_modules():
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            pass
        for name, module in model.named_modules():
            if 'adapter' in name:
                module.train()
                for param in module.parameters():
                    param.requires_grad = True
                    # Tools.print(name)
                    pass
                pass
            pass
        pass

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self):
        Tools.print("Start training")
        best_result = self.test()
        for epoch in range(0, config.epochs):
            train_stats = self.train_one_epoch(self.create_dataset.train_loader, epoch)
            test_result = self.test()

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_result.items()}, 'epoch': epoch}
            Tools.print(json.dumps(log_stats), txt_path=config.log_filename)
            if test_result['r_mean'] > best_result['r_mean']:
                best_result = test_result
                pass
        return best_result

    def train_one_epoch(self, data_loader, epoch):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('loss_contrastive', SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_triplet', SmoothedValue(window_size=1, fmt='{value:.4f}'))

        header = 'Train Epoch: [{}]'.format(epoch)

        self.model.train()
        for i, (image, text, idx, label, pos) in enumerate(metric_logger.log_every(data_loader, 50, header)):
            image = image.to(self.device, non_blocking=True)
            idx = idx.to(self.device, non_blocking=True)
            text_input = self.model.tokenize(text).to(self.device)
            pos_input = self.model.tokenize(pos).to(self.device)

            loss_contrastive, loss_triplet = self.model(image, text_input, pos=pos_input, idx=idx, label=label)
            loss = loss_contrastive + loss_triplet

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            metric_logger.update(loss_contrastive=loss_contrastive.item())
            metric_logger.update(loss_triplet=loss_triplet.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            pass
        return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

    def test(self):
        score_test_i2t, score_test_t2i = self.evaluation(self.create_dataset.test_loader)
        test_result = self.itm_eval(score_test_i2t, score_test_t2i, self.create_dataset.test_dataset.txt2img,
                                    self.create_dataset.test_dataset.img2txt)
        Tools.print(f"Start evaluating test_result={test_result}")
        return test_result

    @torch.no_grad()
    def evaluation(self, data_loader):
        self.model.eval()

        # Inference img features
        image_embeds = []
        for image, img_id in data_loader:
            image_embed = self.model.get_vis_emb(image.to(self.device))
            image_embeds.append(image_embed)
            pass

        # Inference text features
        text_embeds = []
        texts = data_loader.dataset.text
        poss = data_loader.dataset.poss
        num_text, text_bs = len(texts), config.batch_size_test_text  # 256
        for i in range(0, num_text, text_bs):
            text_input = self.model.tokenize(texts[i: min(num_text, i + text_bs)]).to(self.device)
            noun_input = self.model.tokenize(poss[i: min(num_text, i + text_bs)]).to(self.device)
            text_embed = self.model.get_txt_emb(text_input, noun_input)
            text_embeds.append(text_embed)
            pass

        # calculate similarity matrix
        image_embeds = torch.cat(image_embeds, dim=0)
        text_embeds = torch.cat(text_embeds, dim=0)
        sims_matrix = image_embeds @ text_embeds.t()

        score_matrix_i2t = sims_matrix
        score_matrix_t2i = sims_matrix.t()
        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    @torch.no_grad()
    def itm_eval(self, scores_i2t, scores_t2i, txt2img, img2txt):
        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            pass

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]
            pass

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        eval_result = {'txt_r1': round(tr1, 2),
                       'txt_r5': round(tr5, 2),
                       'txt_r10': round(tr10, 2),
                       'img_r1': round(ir1, 2),
                       'img_r5': round(ir5, 2),
                       'img_r10': round(ir10, 2),
                       'r_mean': round(r_mean, 2)}
        return eval_result

    pass


class ConfigCommon(object):

    def __init__(self):
        self.clean_gpu()
        torch.cuda.set_device(self.get_gpu_id())
        self.setup_seed(2024)

        self.device = "cuda"
        self.epochs = 10
        self.lr = 0.001
        self.weight_decay = 0.01

        ################ Model Abl      ###########################################################
        self.abl_param_down_dim = 128
        self.abl_model_has_adapter_image = True
        self.abl_model_has_adapter_text = True
        self.abl_model_has_adapter_pos = True
        self.abl_model_has_adapter_fpn_image = True
        self.abl_model_has_adapter_fpn_text = True
        ################ Model Abl      ###########################################################

        ################ Model setting  ###########################################################
        # Vision encoder setting
        self.image_res = 224  # no need modify
        self.patch_size = 32  # if use swin, set the patch_size to 32, else 16
        ############################################################################################

        ################ Training setting #########################################################
        self.pretrain_path_open_clip = "/data/alisure/2025_benke_data/2_rsitr/open_clip/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin"
        self.batch_size_train = 256
        self.batch_size_test = 128
        self.batch_size_test_text = 128
        ############################################################################################
        pass

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        pass

    @staticmethod
    def get_gpu_id():
        """
        torch.cuda.set_device(get_gpu_id())
        """
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_id, free = 0, 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            now_free = (info.free // 1048576) / 1024  # info.total, info.free, info.used
            if now_free > free:
                free = now_free
                gpu_id = i
            pass
        pynvml.nvmlShutdown()
        return gpu_id

    @staticmethod
    def clean_gpu(o=None):
        if o is not None:
            del o
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        pass

    pass


class Config_RSITMD_ViT(ConfigCommon):

    def __init__(self):
        super().__init__()

        self.output_dir = Tools.new_dir("./outputs/test_RSITMD_ViT")
        self.log_filename = os.path.join(self.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-log.txt"))

        self.model = 'vit'
        self.lr = 0.001
        self.abl_model_has_pos_side = False
        self.abl_model_is_fpn_end_add = False
        self.abl_param_down_dim_pos = 64

        ############## The train & val & test set root ############################################
        self.image_root = '/data/alisure/2025_benke_data/2_rsitr/Dataset/rsitmd'
        self.train_file = ['data/finetune_pos/rsitmd_train.json']  # Path to the training data file
        self.val_file = 'data/finetune_pos/rsitmd_val.json'  # Path to the validation data file
        self.test_file = 'data/finetune_pos/rsitmd_test.json'  # Path to the testing data file
        ############################################################################################
        pass

    pass


class Config_RSICD_ViT(ConfigCommon):

    def __init__(self):
        super().__init__()

        self.output_dir = Tools.new_dir("./outputs/test_RSICD_ViT")
        self.log_filename = os.path.join(self.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-log.txt"))

        self.model = 'vit'
        self.lr = 0.001
        self.abl_model_has_pos_side = True
        self.abl_model_is_fpn_end_add = True
        self.abl_param_down_dim_pos = 128

        ############## The train & val & test set root ############################################
        self.image_root = '/data/alisure/2025_benke_data/2_rsitr/Dataset/rsicd'
        self.train_file = ['data/finetune_pos/rsicd_train.json']  # Path to the training data file
        self.val_file = 'data/finetune_pos/rsicd_val.json'  # Path to the validation data file
        self.test_file = 'data/finetune_pos/rsicd_test.json'  # Path to the testing data file
        ############################################################################################
        pass

    pass


if __name__ == '__main__':
    result_list = []
    for ConfigCLS in [Config_RSITMD_ViT, Config_RSICD_ViT]:
        config = ConfigCLS()
        runner = Runner()
        best_result = runner.train()
        result_list.append(best_result)
        pass

    for result_one in result_list:
        Tools.print(result_one)
        pass
    pass
