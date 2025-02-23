import os

import matplotlib
from matplotlib import patches
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# import wandb
from tqdm import tqdm

from config.config import get_arguments
from dataset.PairedClearSyntheticDataset import PairedClearSyntheticDataset
from dataset.RealFogDataset import RealFogDataset
from models.feature_extractor import FeatureExtractor

from models.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1, FogPassFilterLoss
from models.temp_model import CNNModel

from yolov9_main.models.yolo import Model, DetectionModel
from yolov9_main.train import parse_opt
from yolov9_main.utils.dataloaders import create_dataloader
from yolov9_main.utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_img_size,
                                       check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                                       intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                                       one_cycle, one_flat_cycle, print_args, print_mutation, strip_optimizer,
                                       yaml_save, yaml_load)
from yolov9_main.utils.loss_tal import ComputeLoss
from yolov9_main.utils.torch_utils import smart_optimizer, ModelEMA, EarlyStopping

import sys
path = r"D:\UNI\LAB\FIOD_\yolov9_main"
sys.path.insert(0, path)


def gram_matrix(feature_map):
    channels, height, width = feature_map.size()
    features = feature_map.view(channels, height * width)
    gram = torch.mm(features, torch.t(features))
    return gram

def plot_losses(sf_losses, cw_losses, fsm_losses, con_losses, total_losses):
    # Detach tensors and convert to numpy arrays
    sf_losses_np = [loss.cpu().detach().numpy() for loss in sf_losses]
    cw_losses_np = [loss.cpu().detach().numpy() for loss in cw_losses]
    # fsm_losses_np = [loss.cpu().detach().numpy() for loss in fsm_losses]
    con_losses_np = [loss.cpu().detach().numpy() for loss in con_losses]
    total_losses_np = [loss.cpu().detach().numpy() for loss in total_losses]

    # Write losses to file
    with open('losses.txt', 'w') as f:
        f.write('sf_losses\n')
        f.write(str(sf_losses_np))
        f.write('\n\n')
        f.write('cw_losses\n')
        f.write(str(cw_losses_np))
        f.write('\n\n')
        f.write('con_losses\n')
        f.write(str(con_losses_np))
        f.write('\n\n')
        f.write('total_losses\n')
        f.write(str(total_losses_np))

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sf_losses)), sf_losses_np, label='SF Loss', color='r')
    plt.plot(range(len(cw_losses)), cw_losses_np, label='CW Loss', color='g')
    # plt.plot(range(len(fsm_losses)), fsm_losses_np, label='FSM Loss', color='b')
    plt.plot(range(len(con_losses)), con_losses_np, label='Con Loss', color='b')
    plt.plot(range(len(total_losses)), total_losses_np, label='Total Loss', color='k')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def create_infinite_iterator(loader):
    """
    Creates an infinite iterator that automatically resets when the dataset is exhausted.

    Args:
        loader: DataLoader instance

    Returns:
        A generator that yields (iteration_index, batch) pairs indefinitely
    """
    iteration = 0
    while True:
        for i, batch in enumerate(loader):
            yield iteration + i, batch
        iteration += len(loader)

def main():
    # wandb.init(mode="disabled")
    args = get_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ####### CHUẨN BỊ MODEL VÀ DATALOADER
    lr_fpf1 = 1e-3  # lr của fogpass filter
    lr_fpf2 = 1e-3

    FogPassFilter1 = FogPassFilter_conv1(528)
    FogPassFilter1_optimizer = torch.optim.Adamax([p for p in FogPassFilter1.parameters() if p.requires_grad == True],
                                                  lr=lr_fpf1)
    FogPassFilter1.to(device)
    FogPassFilter2 = FogPassFilter_res1(2080)
    FogPassFilter2_optimizer = torch.optim.Adamax([p for p in FogPassFilter2.parameters() if p.requires_grad == True],
                                                  lr=lr_fpf2)
    FogPassFilter2.to(device)
    fogpassfilter_loss = FogPassFilterLoss(margin=0.1)

    cwsf_dataset = PairedClearSyntheticDataset(args.sf_root, args.cw_root, set='train')
    cwsf_pair_loader = DataLoader(
        cwsf_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=cwsf_dataset.collate_fn
    )

    rf_dataset = RealFogDataset(args.rf_root, args.rf_list_file, mean=args.img_mean)
    rf_loader = DataLoader(
        rf_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=rf_dataset.collate_fn
    )

    cwsf_pair_loader_fogpass = DataLoader(
        cwsf_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=cwsf_dataset.collate_fn
    )
    rf_loader_fogpass = DataLoader(
        rf_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=rf_dataset.collate_fn
    )

    rf_loader_iter = create_infinite_iterator(rf_loader)
    cwsf_pair_loader_iter = create_infinite_iterator(cwsf_pair_loader)
    cwsf_pair_loader_iter_fogpass = create_infinite_iterator(cwsf_pair_loader_fogpass)
    rf_loader_iter_fogpass = create_infinite_iterator(rf_loader_fogpass)

    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    m = nn.Softmax(dim=1)
    log_m = nn.LogSoftmax(dim=1)
    mse = nn.MSELoss(reduction='mean').to(device)
    # model = CNNModel()


    ################# YOLOv9
    RANK = int(os.getenv('RANK', -1))
    hyp = yaml_load('yolov9_main/data/hyps/hyp.scratch-high.yaml')
    opt = parse_opt()

    model = Model(cfg='yolov9_main/models/detect/yolov9-s.yaml').to(device)

    # checkpoint = torch.load('yolov9-s.pt', map_location='cpu')
    # model.load_state_dict(checkpoint, strict = False)

    amp = check_amp(model)
    amp_device = "cuda" if amp else "cpu"
    scaler = torch.amp.GradScaler(enabled=amp)
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    stopper, stop = EarlyStopping(patience=opt.patience), False
    last_opt_step = -1
    accumulate = 16

    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], opt.epochs)  # cosine 1->hyp['lrf']
    elif opt.flat_cos_lr:
        lf = one_flat_cycle(1, hyp['lrf'], opt.epochs)  # flat cosine 1->hyp['lrf']
    elif opt.fixed_lr:
        lf = lambda x: 1.0
    else:
        lf = lambda x: (1 - x / opt.epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    ema = ModelEMA(model) if RANK in {-1, 0} else None

    compute_loss = ComputeLoss(model)

    #################
    criterion = nn.MSELoss()
    # optimization = optim.SGD(model.parameters(), lr=0.000001, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimization, step_size=30, gamma=0.1)
    extractor = FeatureExtractor(model, 8)

    sf_losses = []
    cw_losses = []
    fsm_losses = []
    con_losses = []
    total_losses = []

    # wandb.init(project="yolov9_training", name="experiment_1")
    # matplotlib.use('TkAgg')
    temp = 0
    for i_iter in tqdm(range(0, args.num_steps)):
        loss_detection_value = 0
        loss_fsm_value = 0
        loss_con_value = 0
        temp += 1

        # for sub_i in range(args.iter_size):
        #     # Train fog-pass filtering module
        #     model.eval()
        #     for param in model.parameters():
        #         param.requires_grad = False
        #     for param in FogPassFilter1.parameters():
        #         param.requires_grad = True
        #     for param in FogPassFilter2.parameters():
        #         param.requires_grad = True

        #     # Lấy batch dữ liệu
        #     _, batch = next(cwsf_pair_loader_iter_fogpass)
        #     foggy_image, clear_image, box, name = batch

        #     _, batch_rf = next(rf_loader_iter_fogpass)
        #     rf_img, rf_name = batch_rf

        #     # Move images to GPU and wrap in Variable
        #     realfog_images = rf_img.to(device)
        #     foggy_images = foggy_image.to(device)
        #     clear_images = clear_image.to(device)


        #     # Get feature maps for each image type
        #     realfog_features = extractor.get_feature_maps(realfog_images)
        #     foggy_features = extractor.get_feature_maps(foggy_images)
        #     clear_features = extractor.get_feature_maps(clear_images)

        #     feature_realfog0, feature_realfog1 = realfog_features[0], realfog_features[1]
        #     feature_foggy0, feature_foggy1 = foggy_features[0], foggy_features[1]
        #     feature_clear0, feature_clear1 = clear_features[0], clear_features[1]

        #     fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
        #     sf_features = {'layer0': feature_foggy0, 'layer1': feature_foggy1}
        #     cw_features = {'layer0': feature_clear0, 'layer1': feature_clear1}
        #     rf_features = {'layer0': feature_realfog0, 'layer1': feature_realfog1}

        #     total_fpf_loss = 0

        #     for idx, layer in enumerate(fsm_weights):
        #         cw_feature = cw_features[layer]
        #         sf_feature = sf_features[layer]
        #         rf_feature = rf_features[layer]

        #         # print(f"Layer {idx}, rf_feature shape: {rf_feature.shape}")
        #         #
        #         # if rf_feature.shape[0] != 4:
        #         #     print(f"Warning: Batch size of rf_feature in layer {idx} is {rf_feature.shape[0]}, not 4")
        #         #     print(f"temp(batch): {temp}")
        #         # else:
        #         #     print(f"Batch size of rf_feature in layer {idx} is 4")

        #         fogpassfilter = None
        #         fogpassfilter_optimizer = None
        #         if idx == 0:
        #             fogpassfilter = FogPassFilter1
        #             fogpassfilter_optimizer = FogPassFilter1_optimizer
        #         elif idx == 1:
        #             fogpassfilter = FogPassFilter2
        #             fogpassfilter_optimizer = FogPassFilter2_optimizer

        #         fogpassfilter.train()
        #         fogpassfilter_optimizer.zero_grad()

        #         sf_gram = [0] * args.batch_size
        #         cw_gram = [0] * args.batch_size
        #         rf_gram = [0] * args.batch_size
        #         vector_sf_gram = [0] * args.batch_size
        #         vector_cw_gram = [0] * args.batch_size
        #         vector_rf_gram = [0] * args.batch_size
        #         fog_factor_sf = [0] * args.batch_size
        #         fog_factor_cw = [0] * args.batch_size
        #         fog_factor_rf = [0] * args.batch_size

        #         for batch_idx in range(args.batch_size):
        #             sf_gram[batch_idx] = gram_matrix(sf_feature[batch_idx])
        #             cw_gram[batch_idx] = gram_matrix(cw_feature[batch_idx])
        #             rf_gram[batch_idx] = gram_matrix(rf_feature[batch_idx])
        #             vector_sf_gram[batch_idx] = sf_gram[batch_idx][
        #                 torch.triu(torch.ones_like(sf_gram[batch_idx])) == 1
        #                 ].detach().clone().requires_grad_()

        #             vector_cw_gram[batch_idx] = cw_gram[batch_idx][
        #                 torch.triu(torch.ones_like(cw_gram[batch_idx])) == 1
        #                 ].detach().clone().requires_grad_()

        #             vector_rf_gram[batch_idx] = rf_gram[batch_idx][
        #                 torch.triu(torch.ones_like(rf_gram[batch_idx])) == 1
        #                 ].detach().clone().requires_grad_()

        #             fog_factor_sf[batch_idx] = fogpassfilter(vector_sf_gram[batch_idx])
        #             fog_factor_cw[batch_idx] = fogpassfilter(vector_cw_gram[batch_idx])
        #             fog_factor_rf[batch_idx] = fogpassfilter(vector_rf_gram[batch_idx])

        #         fog_factor_embeddings = torch.cat((torch.unsqueeze(fog_factor_sf[0], 0),
        #                                            torch.unsqueeze(fog_factor_cw[0], 0),
        #                                            torch.unsqueeze(fog_factor_rf[0], 0),
        #                                            torch.unsqueeze(fog_factor_sf[1], 0),
        #                                            torch.unsqueeze(fog_factor_cw[1], 0),
        #                                            torch.unsqueeze(fog_factor_rf[1], 0),
        #                                            torch.unsqueeze(fog_factor_sf[2], 0),
        #                                            torch.unsqueeze(fog_factor_cw[2], 0),
        #                                            torch.unsqueeze(fog_factor_rf[2], 0),
        #                                            torch.unsqueeze(fog_factor_sf[3], 0),
        #                                            torch.unsqueeze(fog_factor_cw[3], 0),
        #                                            torch.unsqueeze(fog_factor_rf[3], 0)), 0)
        #         # fog_factor_embeddings = []
        #         # for i in range(args.batch_size):
        #         #     fog_factor_embeddings = torch.cat((fog_factor_embeddings,
        #         #                                        torch.cat((torch.unsqueeze(fog_factor_sf[i], 0),
        #         #                                                   torch.unsqueeze(fog_factor_cw[i], 0),
        #         #                                                   torch.unsqueeze(fog_factor_rf[i], 0))), 0))

        #         fog_factor_embeddings_norm = torch.norm(fog_factor_embeddings, p=2, dim=1).detach()
        #         size_fog_factor = fog_factor_embeddings.size()
        #         fog_factor_embeddings = fog_factor_embeddings.div(
        #             fog_factor_embeddings_norm.expand(size_fog_factor[1], 12).t())

        #         fog_factor_labels = torch.LongTensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        #         # fog_factor_labels = []
        #         # for i in range(args.batch_size):
        #         #     fog_factor_labels = torch.cat((fog_factor_labels, torch.LongTensor([0, 1, 2])), 0)

        #         fog_pass_filter_loss = fogpassfilter_loss(fog_factor_embeddings, fog_factor_labels)
        #         # fog_pass_filter_loss.backward()
        #         # fogpassfilter_optimizer.step()
        #         total_fpf_loss += fog_pass_filter_loss

        #         print("total_fpf_loss: ", total_fpf_loss);

        #         # wandb.log({f'layer{idx}/fpf loss': fog_pass_filter_loss}, step=i_iter)
        #         # wandb.log({f'layer{idx}/total fpf loss': total_fpf_loss}, step=i_iter)

        #     print(f'total_fpf_loss: {total_fpf_loss}')
        #     print(f'temp: {temp}')
        #     with torch.autograd.detect_anomaly():
        #         total_fpf_loss.backward(retain_graph=True)
        #     FogPassFilter1_optimizer.step()
        #     FogPassFilter2_optimizer.step()

        # Train model
        ####################
        model.train()
        optimizer.zero_grad()

        for param in model.parameters():
            param.requires_grad = True
        for param in FogPassFilter1.parameters():
            param.requires_grad = False
        for param in FogPassFilter2.parameters():
            param.requires_grad = False

        _, batch = cwsf_pair_loader_iter.__next__()
        sf_image, cw_image, box, name = batch

        ############### illustrate the batch size of rf_loader_iter

        print("^^^^^ TEST GROUND TRUTH!")

        # img = cw_image[0].cpu().numpy().transpose((1, 2, 0))  # Change to HWC format
        # img = (img * 255).astype(np.uint8)  # Scale to 0-255 if necessary

        # _, ax = plt.subplots(1)
        # ax.imshow(img)

        # boxes_for_image = box[box[:, 0] == 0] # chỉ chọn box của batch_id 0

        # for b in boxes_for_image:
        #     x, y, w, h = b[2:]
        #     x *= img.shape[1]
        #     y *= img.shape[0]
        #     w *= img.shape[1]
        #     h *= img.shape[0]
        #     rect = patches.Rectangle((x - w / 2, y - h / 2), w, h, linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)

        # plt.show()
        ##############

        _, batch_rf = rf_loader_iter.__next__()
        rf_img, rf_name = batch_rf
        sf_loss = torch.tensor(0.0, dtype=float, device=device)
        cw_loss = torch.tensor(0.0, dtype=float, device=device)
        loss_con = torch.tensor(0.0, dtype=float, device=device)

        if i_iter % 3 == 0:
            # Move images to GPU
            sf_images = sf_image.to(device, non_blocking=True).float() / 255
            cw_images = cw_image.to(device, non_blocking=True).float() / 255
            boxes = box.to(device)

            # Get predictions and features
            with torch.amp.autocast(amp_device):
                sf_predictions = model(sf_images)
                sf_loss, sf_loss_items = compute_loss(sf_predictions[0], boxes)
                cw_predictions = model(cw_images)
                cw_loss, cw_loss_items = compute_loss(cw_predictions[0], boxes)

            sf_features_list = extractor.get_feature_maps(sf_images)
            feature_sf0, feature_sf1 = sf_features_list[0], sf_features_list[1]

            cw_features_list = extractor.get_feature_maps(cw_images)
            feature_cw0, feature_cw1 = cw_features_list[0], cw_features_list[1]

            # loss_con = kl_loss(log_m(sf_predictions[0][0]), log_m(cw_predictions[0][0]))
            # sf_preds = torch.cat([p for p in sf_predictions[0]], 1)
            # cw_preds = torch.cat([p for p in cw_predictions[0]], 1)

            pd = len(sf_predictions[0]) # prediction heads
            for i in range(len(sf_predictions[1])):
                loss_con += mse(sf_predictions[1][i], cw_predictions[1][i])

            loss_con /= pd
            # loss_con.backward()

            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
            sf_features = {'layer0': feature_sf0, 'layer1': feature_sf1}
            cw_features = {'layer0': feature_cw0, 'layer1': feature_cw1}

        elif i_iter % 3 == 1:
            # Move images to GPU
            sf_images = sf_image.to(device, non_blocking=True).float() / 255
            rf_images = rf_img.to(device, non_blocking=True).float() / 255
            boxes = box.to(device)

            with torch.amp.autocast(amp_device):
                sf_predictions = model(sf_images)  # forward
                sf_loss, sf_loss_items = compute_loss(sf_predictions[0], boxes)
            sf_features_list = extractor.get_feature_maps(sf_images)
            feature_sf0, feature_sf1 = sf_features_list[0], sf_features_list[1]

            rf_predictions = model(rf_images)
            rf_features_list = extractor.get_feature_maps(rf_images)
            feature_rf0, feature_rf1 = rf_features_list[0], rf_features_list[1]

            rf_features = {'layer0': feature_rf0, 'layer1': feature_rf1}
            sf_features = {'layer0': feature_sf0, 'layer1': feature_sf1}
            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}

        else:  # i_iter % 3 == 2
            # Move images to GPU
            cw_images = cw_image.to(device, non_blocking=True).float() / 255
            rf_images = rf_img.to(device, non_blocking=True).float() / 255
            boxes = box.to(device)

            # Get predictions and features
            with torch.amp.autocast(amp_device):
                cw_predictions = model(cw_images)  # forward
                cw_loss, cw_loss_items = compute_loss(cw_predictions[0], boxes)
            cw_features_list = extractor.get_feature_maps(cw_images)
            feature_cw0, feature_cw1 = cw_features_list[0], cw_features_list[1]

            rf_predictions = model(rf_images)
            rf_features_list = extractor.get_feature_maps(rf_images)
            feature_rf0, feature_rf1 = rf_features_list[0], rf_features_list[1]

            rf_features = {'layer0': feature_rf0, 'layer1': feature_rf1}
            cw_features = {'layer0': feature_cw0, 'layer1': feature_cw1}
            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}

        sf_losses.append(sf_loss)
        cw_losses.append(cw_loss)
        con_losses.append(loss_con)

        loss_fsm = 0
        fog_pass_filter_loss = 0

        for idx, layer in enumerate(fsm_weights):
            a_feature = None
            b_feature = None
            # fog pass filter loss between different fog conditions a and b
            if i_iter % 3 == 0:
                a_feature = cw_features[layer]
                b_feature = sf_features[layer]
            if i_iter % 3 == 1:
                a_feature = rf_features[layer]
                b_feature = sf_features[layer]
            if i_iter % 3 == 2:
                a_feature = rf_features[layer]
                b_feature = cw_features[layer]

            layer_fsm_loss = 0
            na, da, ha, wa = a_feature.size()
            nb, db, hb, wb = b_feature.size()

            fogpassfilter = None
            fogpassfilter_optimizer = None

            if idx == 0:
                fogpassfilter = FogPassFilter1
                fogpassfilter_optimizer = FogPassFilter1_optimizer
            elif idx == 1:
                fogpassfilter = FogPassFilter2
                fogpassfilter_optimizer = FogPassFilter2_optimizer

            fogpassfilter.eval()

            for batch_idx in range(4):
                b_gram = gram_matrix(b_feature[batch_idx])
                a_gram = gram_matrix(a_feature[batch_idx])

                if i_iter % 3 == 1 or i_iter % 3 == 2:
                    a_gram = a_gram * (hb * wb) / (ha * wa)

                vector_b_gram = b_gram[torch.triu(
                    torch.ones(b_gram.size()[0], b_gram.size()[1])).requires_grad_() == 1].requires_grad_()
                vector_a_gram = a_gram[torch.triu(
                    torch.ones(a_gram.size()[0], a_gram.size()[1])).requires_grad_() == 1].requires_grad_()

                fog_factor_b = fogpassfilter(vector_b_gram)
                fog_factor_a = fogpassfilter(vector_a_gram)

                # print("a_feature.shape: ", a_feature.shape)
                # print("b_feature.shape: ", b_feature.shape)
                # print("fog_factor_b.shape: ", fog_factor_b.shape)
                # print("b_feature.size(0)", b_feature.size(0)) # batch size??

                half = int(fog_factor_b.shape[0] / 2)

                layer_fsm_loss += fsm_weights[layer] * torch.mean(
                    (fog_factor_b / (hb * wb) - fog_factor_a / (ha * wa)) ** 2) / half / b_feature.size(0)

            loss_fsm += layer_fsm_loss / 4.

        print("loss_con ", loss_con)
        total_loss = (
            sf_loss + cw_loss             # Detection Loss
            + args.weight_fsm * loss_fsm  # Fog Style Matching Loss
            + args.weight_con * loss_con  # Consistency Loss
        )

        total_loss = total_loss / args.iter_size
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
        scaler.step(optimizer)  # optimizer.step
        scaler.update()
        optimizer.zero_grad()
        if ema:
            ema.update(model)

        # if i_iter - last_opt_step >= accumulate:
        #     scaler.unscale_(optimizer)  # unscale gradients
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
        #     scaler.step(optimizer)  # optimizer.step
        #     scaler.update()
        #     optimizer.zero_grad()
        #     if ema:
        #         ema.update(model)
        #     last_opt_step = i_iter

        # optimizer.step()

        sf_losses.append(sf_loss)
        cw_losses.append(cw_loss)
        fsm_losses.append(args.weight_fsm * loss_fsm)
        total_losses.append(total_loss)

        if (sf_loss + cw_loss) != 0:
            detect_loss = sf_loss + cw_loss
            loss_detection_value += detect_loss.data.cpu().numpy() / args.iter_size
        if loss_fsm != 0:
            loss_fsm_value += loss_fsm.data.cpu().numpy() / args.iter_size
        if loss_con != 0:
            loss_con_value += loss_con.data.cpu().numpy() / args.iter_size

        # wandb.log({
        #     "detect_loss": loss_detection_value,
        #     # "class_loss": loss_cls_value,
        #     # "dfl_loss": loss_dfl_value,
        #     "fsm_loss": args.weight_fsm * loss_fsm_value,
        #     "consistency_loss": args.weight_con * loss_con_value,
        #     "total_loss": total_loss,
        # }, step=i_iter)

        print(
            f"epoch {i_iter + 1}:total_loss: {total_loss}, detect_loss: {loss_detection_value}, fsm_loss: {args.weight_fsm * loss_fsm_value}"
            f", consistency_loss: {args.weight_con * loss_con_value}"
            )

    scheduler.step()
    plot_losses(sf_losses, cw_losses, fsm_losses, con_losses, total_losses)
    print("end")
    # wandb.finish()

    return

if __name__ == '__main__':
    main()
