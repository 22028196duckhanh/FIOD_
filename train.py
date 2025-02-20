import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
# import wandb
from tqdm import tqdm

import yolov9_main.val as validate
from config.config import get_arguments
from dataset.PairedClearSyntheticDataset import PairedClearSyntheticDataset
from dataset.RealFogDataset import RealFogDataset
from models.feature_extractor import FeatureExtractor
from models.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1, FogPassFilterLoss
from yolov9_main.models.yolo import Model
from yolov9_main.train import parse_opt
from yolov9_main.utils.dataloaders import create_dataloader
from yolov9_main.utils.general import (check_amp, colorstr, one_cycle, one_flat_cycle, yaml_load)
from yolov9_main.utils.loss_tal import ComputeLoss
from yolov9_main.utils.metrics import fitness
from yolov9_main.utils.torch_utils import smart_optimizer, ModelEMA, EarlyStopping


def gram_matrix(feature_map):
    channels, height, width = feature_map.size()
    features = feature_map.view(channels, height * width)
    gram = torch.mm(features, torch.t(features))
    return gram


# def compute_yolo_loss(model, predictions, labels):
#     loss, loss_items = model.loss(predictions, labels)
#
#     box_loss = loss_items[0]  # Box loss (CIoU/GIoU)
#     class_loss = loss_items[1]  # Class loss (Cross-entropy)
#     dfl_loss = loss_items[2]  # Distribution Focal Loss (DFL)
#
#     return box_loss, class_loss, dfl_loss

def plot_losses(box_losses, cls_losses, dfl_losses, fsm_losses, total_losses):

    box_losses_np = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else np.array(loss) for loss in box_losses]
    cls_losses_np = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else np.array(loss) for loss in cls_losses]
    dfl_losses_np = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else np.array(loss) for loss in dfl_losses]
    fsm_losses_np = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else np.array(loss) for loss in fsm_losses]
    total_losses_np = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else np.array(loss) for loss in total_losses]

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(box_losses)), box_losses_np, label='Box Loss', color='r')
    plt.plot(range(len(cls_losses)), cls_losses_np, label='Class Loss', color='g')
    plt.plot(range(len(dfl_losses)), dfl_losses_np, label='DFL Loss', color='b')
    plt.plot(range(len(fsm_losses)), fsm_losses_np, label='FSM Loss', color='y')
    plt.plot(range(len(total_losses)), total_losses_np, label='Total Loss', color='k')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs('results', exist_ok=True)
    save_path = os.path.join('results', 'loss_plot.png')

    # Lưu hình vào file
    plt.savefig(save_path)
    plt.close()



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
    # loss của fpf
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

    rf_dataset = RealFogDataset(args.rf_root, args.rf_list_file)
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
    # model = CNNModel()

    ################# YOLOv9

    RANK = int(os.getenv('RANK', -1))
    hyp = yaml_load('yolov9_main/data/hyps/hyp.scratch-high.yaml')
    opt = parse_opt()
    nc = 2
    names = {0: 'person', 2: 'car'}

    model = Model(cfg='yolov9_main/models/detect/yolov9-s.yaml')

    # checkpoint = torch.load('yolov9-s.pt', map_location='cpu')
    # model.load_state_dict(checkpoint, strict = False)

    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    amp = check_amp(model)
    amp_device = "cuda" if amp else "cpu"
    scaler = torch.amp.GradScaler(enabled=amp)
    optimizer = smart_optimizer(model, 'Adam', hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
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

    best_fitness, start_epoch = 0.0, 0
    scheduler.last_epoch = start_epoch-1

    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    gs = max(int(model.stride.max()), 32)
    val_loader = create_dataloader("E:\\yolov9_modify_architecture\\val_merge\\images",
                                   640,
                                   args.batch_size,
                                   gs,
                                   single_cls = False,
                                   hyp=hyp,
                                   cache=None,
                                   rect=True,
                                   rank=-1,
                                   workers=args.num_workers,
                                   pad=0.5,
                                   prefix=colorstr('val: '))[0]

    model.half().float()

    #################
    criterion = nn.MSELoss()
    # optimization = optim.SGD(model.parameters(), lr=0.000001, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimization, step_size=30, gamma=0.1)
    extractor = FeatureExtractor(model, 8)

    box_losses = []
    cls_losses = []
    dfl_losses = []
    fsm_losses = []
    total_losses = []

    # wandb.init(project="yolov9_training", name="experiment_1")
    for i_iter in tqdm(range(0, args.num_steps)):
        loss_box_value = 0
        loss_cls_value = 0
        loss_dfl_value = 0
        loss_fsm_value = 0
        # loss_con_value = 0
        for sub_i in range(args.iter_size):
            # Train fog-pass filtering module
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            for param in FogPassFilter1.parameters():
                param.requires_grad = True
            for param in FogPassFilter2.parameters():
                param.requires_grad = True

            # Lấy batch dữ liệu
            _, batch = next(cwsf_pair_loader_iter_fogpass)
            foggy_image, clear_image, box, name = batch

            _, batch_rf = next(rf_loader_iter_fogpass)
            rf_img, rf_name = batch_rf

            # Move images to GPU and wrap in Variable
            realfog_images = rf_img.to(device)
            foggy_images = foggy_image.to(device)
            clear_images = clear_image.to(device)


            # Get feature maps for each image type
            realfog_features = extractor.get_feature_maps(realfog_images)
            foggy_features = extractor.get_feature_maps(foggy_images)
            clear_features = extractor.get_feature_maps(clear_images)

            feature_realfog0, feature_realfog1 = realfog_features[0], realfog_features[1]
            feature_foggy0, feature_foggy1 = foggy_features[0], foggy_features[1]
            feature_clear0, feature_clear1 = clear_features[0], clear_features[1]

            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
            sf_features = {'layer0': feature_foggy0, 'layer1': feature_foggy1}
            cw_features = {'layer0': feature_clear0, 'layer1': feature_clear1}
            rf_features = {'layer0': feature_realfog0, 'layer1': feature_realfog1}

            total_fpf_loss = 0
            fogpassfilter = None
            fogpassfilter_optimizer = None

            for idx, layer in enumerate(fsm_weights):
                cw_feature = cw_features[layer]
                sf_feature = sf_features[layer]
                rf_feature = rf_features[layer]

                if idx == 0:
                    fogpassfilter = FogPassFilter1
                    fogpassfilter_optimizer = FogPassFilter1_optimizer
                elif idx == 1:
                    fogpassfilter = FogPassFilter2
                    fogpassfilter_optimizer = FogPassFilter2_optimizer

                fogpassfilter.train()
                fogpassfilter_optimizer.zero_grad()

                sf_gram = [0] * args.batch_size
                cw_gram = [0] * args.batch_size
                rf_gram = [0] * args.batch_size
                vector_sf_gram = [0] * args.batch_size
                vector_cw_gram = [0] * args.batch_size
                vector_rf_gram = [0] * args.batch_size
                fog_factor_sf = [0] * args.batch_size
                fog_factor_cw = [0] * args.batch_size
                fog_factor_rf = [0] * args.batch_size

                for batch_idx in range(args.batch_size):
                    sf_gram[batch_idx] = gram_matrix(sf_feature[batch_idx])
                    cw_gram[batch_idx] = gram_matrix(cw_feature[batch_idx])
                    rf_gram[batch_idx] = gram_matrix(rf_feature[batch_idx])
                    vector_sf_gram[batch_idx] = sf_gram[batch_idx][
                        torch.triu(torch.ones_like(sf_gram[batch_idx])) == 1
                        ].detach().clone().requires_grad_()

                    vector_cw_gram[batch_idx] = cw_gram[batch_idx][
                        torch.triu(torch.ones_like(cw_gram[batch_idx])) == 1
                        ].detach().clone().requires_grad_()

                    vector_rf_gram[batch_idx] = rf_gram[batch_idx][
                        torch.triu(torch.ones_like(rf_gram[batch_idx])) == 1
                        ].detach().clone().requires_grad_()

                    fog_factor_sf[batch_idx] = fogpassfilter(vector_sf_gram[batch_idx])
                    fog_factor_cw[batch_idx] = fogpassfilter(vector_cw_gram[batch_idx])
                    fog_factor_rf[batch_idx] = fogpassfilter(vector_rf_gram[batch_idx])

                embeddings_list = []
                for batch_idx in range(args.batch_size):
                    embeddings_list.append(fog_factor_sf[batch_idx].unsqueeze(0))
                    embeddings_list.append(fog_factor_cw[batch_idx].unsqueeze(0))
                    embeddings_list.append(fog_factor_rf[batch_idx].unsqueeze(0))
                fog_factor_embeddings = torch.cat(embeddings_list, dim=0)

                fog_factor_embeddings_norm = torch.norm(fog_factor_embeddings, p=2, dim=1).detach()
                size_fog_factor = fog_factor_embeddings.size()
                fog_factor_embeddings = fog_factor_embeddings.div(
                    fog_factor_embeddings_norm.expand(size_fog_factor[1], args.batch_size * 3).t())
                fog_factor_labels = torch.arange(3, device=device).long().repeat(args.batch_size)
                fog_pass_filter_loss = fogpassfilter_loss(fog_factor_embeddings, fog_factor_labels)
                # fogpassfilter_optimizer.step()
                total_fpf_loss += fog_pass_filter_loss

                # wandb.log({f'layer{idx}/fpf loss': fog_pass_filter_loss}, step=i_iter)
                # wandb.log({f'layer{idx}/total fpf loss': total_fpf_loss}, step=i_iter)

            print(f'total_fpf_loss: {total_fpf_loss}')
            with torch.autograd.detect_anomaly():
                total_fpf_loss.backward()
            FogPassFilter1_optimizer.step()
            FogPassFilter2_optimizer.step()

            # Train model
            ####################
            model.train()

            optimizer.zero_grad()

            # Save model
            # if (not nosave) or (final_epoch and not evolve):  # if save
            #     ckpt = {
            #         'epoch': epoch,
            #         'best_fitness': best_fitness,
            #         'model': deepcopy(de_parallel(model)).half(),
            #         'ema': deepcopy(ema.ema).half(),
            #         'updates': ema.updates,
            #         'optimizer': optimizer.state_dict(),
            #         'opt': vars(opt),
            #         'git': GIT_INFO,  # {remote, branch, commit} if a git repo
            #         'date': datetime.now().isoformat()}
            #
            #     # Save last, best and delete
            #     torch.save(ckpt, last)
            #     if best_fitness == fi:
            #         torch.save(ckpt, best)
            #     if opt.save_period > 0 and epoch % opt.save_period == 0:
            #         torch.save(ckpt, w / f'epoch{epoch}.pt')
            #     del ckpt
            #     callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        for param in model.parameters():
            param.requires_grad = True
        for param in FogPassFilter1.parameters():
            param.requires_grad = False
        for param in FogPassFilter2.parameters():
            param.requires_grad = False

        _, batch = cwsf_pair_loader_iter.__next__()
        sf_image, cw_image, box, name = batch

        _, batch_rf = rf_loader_iter.__next__()
        rf_img, rf_name = batch_rf

        if i_iter % 3 == 0:
            # Move images to GPU
            sf_images = sf_image.to(device, non_blocking=True).float()
            cw_images = cw_image.to(device, non_blocking=True).float()
            boxes = box.to(device)

            # Get predictions and features
            with torch.amp.autocast(amp_device):
                sf_predictions = model(sf_images)  # forward
                sf_loss, sf_loss_items = compute_loss(sf_predictions[1], boxes)
                sf_box_loss, sf_class_loss, sf_dfl_loss = sf_loss_items

            sf_features_list = extractor.get_feature_maps(sf_images)
            feature_sf0, feature_sf1 = sf_features_list[0], sf_features_list[1]

            with torch.amp.autocast(amp_device):
                cw_predictions = model(cw_images)  # forward
                cw_loss, cw_loss_items = compute_loss(cw_predictions[1], boxes)
                cw_box_loss, cw_class_loss, cw_dfl_loss = cw_loss_items
            cw_features_list = extractor.get_feature_maps(cw_images)
            feature_cw0, feature_cw1 = cw_features_list[0], cw_features_list[1]

            # CONSISTENCY LOSS

            # loss_con = kl_loss(log_m(feature_sf0), m(feature_cw0))
            # loss_con.backward()


            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
            sf_features = {'layer0': feature_sf0, 'layer1': feature_sf1}
            cw_features = {'layer0': feature_cw0, 'layer1': feature_cw1}

        elif i_iter % 3 == 1:

            sf_images = sf_image.to(device, non_blocking=True).float()
            rf_images = rf_img.to(device, non_blocking=True).float()
            boxes = box.to(device)

            with torch.amp.autocast(amp_device):
                sf_predictions = model(sf_images)  # forward
                sf_loss, sf_loss_items = compute_loss(sf_predictions[1], boxes)
                sf_box_loss, sf_class_loss, sf_dfl_loss = sf_loss_items
            sf_features_list = extractor.get_feature_maps(sf_images)
            feature_sf0, feature_sf1 = sf_features_list[0], sf_features_list[1]

            rf_predictions = model(rf_images)
            rf_features_list = extractor.get_feature_maps(rf_images)
            feature_rf0, feature_rf1 = rf_features_list[0], rf_features_list[1]

            # loss_con = 0

            rf_features = {'layer0': feature_rf0, 'layer1': feature_rf1}
            sf_features = {'layer0': feature_sf0, 'layer1': feature_sf1}
            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}

        else:  # i_iter % 3 == 2

            cw_images = cw_image.to(device, non_blocking=True).float()
            rf_images = rf_img.to(device, non_blocking=True).float()
            boxes = box.to(device)

            with torch.amp.autocast(amp_device):
                cw_predictions = model(cw_images)
                cw_loss, cw_loss_items = compute_loss(cw_predictions[1], boxes)
                cw_box_loss, cw_class_loss, cw_dfl_loss = cw_loss_items
            cw_features_list = extractor.get_feature_maps(cw_images)
            feature_cw0, feature_cw1 = cw_features_list[0], cw_features_list[1]

            rf_predictions = model(rf_images)
            rf_features_list = extractor.get_feature_maps(rf_images)
            feature_rf0, feature_rf1 = rf_features_list[0], rf_features_list[1]

            # loss_con = 0

            rf_features = {'layer0': feature_rf0, 'layer1': feature_rf1}
            cw_features = {'layer0': feature_cw0, 'layer1': feature_cw1}
            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}

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

            for batch_idx in range(args.batch_size):
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
                half = int(fog_factor_b.shape[0] / 2)

                layer_fsm_loss += fsm_weights[layer] * torch.mean(
                    (fog_factor_b / (hb * wb) - fog_factor_a / (ha * wa)) ** 2) / half / b_feature.size(0)

            loss_fsm += layer_fsm_loss / 4.

        total_loss = (
                args.weight_box * (sf_box_loss + cw_box_loss) +
                args.weight_dfl * (sf_dfl_loss + cw_dfl_loss) +
                args.weight_cls * (sf_class_loss + cw_class_loss) +
                args.weight_fsm * loss_fsm  # FSM Loss
            # + args.weight_con * loss_con  # Consistency Loss
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

        box_losses.append(args.weight_box * (sf_box_loss + cw_box_loss))
        dfl_losses.append(args.weight_dfl * (sf_dfl_loss + cw_dfl_loss))
        cls_losses.append(args.weight_cls * (sf_class_loss + cw_class_loss))
        fsm_losses.append(args.weight_fsm * loss_fsm)
        total_losses.append(total_loss)

        if (sf_box_loss + cw_box_loss) != 0:
            box_loss = args.weight_box * (sf_box_loss + cw_box_loss)
            loss_box_value += box_loss.data.cpu().numpy() / args.iter_size
        if (sf_class_loss + cw_class_loss) != 0:
            class_loss = args.weight_cls * (sf_class_loss + cw_class_loss)
            loss_cls_value += class_loss.data.cpu().numpy() / args.iter_size
        # if (sf_obj_loss + cw_obj_loss) != 0:
        #     loss_obj = args.weight_obj * (sf_obj_loss + cw_obj_loss)
        #     loss_obj_value += loss_obj.data.cpu().numpy() / args.iter_size
        if (sf_dfl_loss + cw_dfl_loss) != 0:
            loss_dfl = args.weight_obj * (sf_dfl_loss + cw_dfl_loss)
            loss_dfl_value += loss_dfl.data.cpu().numpy() / args.iter_size
        if loss_fsm != 0:
            loss_fsm = loss_fsm* args.weight_fsm
            loss_fsm_value += loss_fsm.data.cpu().numpy() / args.iter_size
        # if loss_con != 0:
        #     loss_con_value += loss_con.data.cpu().numpy() / args.iter_size

        # wandb.log({
        #     "box_loss": loss_box_value,
        #     # "class_loss": loss_cls_value,
        #     # "dfl_loss": loss_dfl_value,
        #     "fsm_loss": args.weight_fsm * loss_fsm_value,
        #     "consistency_loss": args.weight_con * loss_con_value,
        #     "total_loss": total_loss,
        # }, step=i_iter)

        print(
            f"Step {i_iter + 1}:total_loss: {total_loss}, box_loss: {loss_box_value},dfl loss: {loss_dfl_value}, cls_loss: {loss_cls_value}, fsm_loss: {loss_fsm_value},"
            # f", consistency_loss: {args.weight_con * loss_con_value}"
            )

        scheduler.step()

        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        final_epoch = (i_iter + 1 == args.num_steps) or stopper.possible_stop
        if final_epoch:  # Calculate mAP
            print("val")
            results, maps, _ = validate.run(batch_size=args.batch_size,
                                            half=amp,
                                            model=ema.ema,
                                            single_cls=False,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            plots=False,
                                            compute_loss=compute_loss)
            print("eval")

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi

    plot_losses(box_losses, cls_losses, dfl_losses, fsm_losses, total_losses)
    print("end")
    # wandb.finish()

    return

if __name__ == '__main__':
    main()
