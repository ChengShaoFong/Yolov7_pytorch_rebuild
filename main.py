import argparse
import torch

from collections import OrderedDict
import sys, os

# from prune.pruning import *
from utils.utils import *
from modelConvertTool.blockBuilder import *
from modelConvertTool.getInfo import *
from modelConvertTool.modelModules import *


############# v7 ##################
import json
import yaml
from pathlib import Path
from threading import Thread
from tqdm import tqdm

from utils.experimental import *
from utils.metrics import ConfusionMatrix

from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr

from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized # TracedModel

from pathlib import Path
from utils.experimental import *


#depend on yolov7 location's suite
# sys.path.append(os.path.abspath("../yolov7/yolov7-main"))  # import v7 module
sys.path.append(os.path.abspath("./models"))   # import own module

print(sys.path)

# test the model convert is success or not (accuracy before and after should be the same)
def testing(data, model):

    batch_size=32
    imgsz=640
    conf_thres=0.001
    iou_thres=0.65 # for NMS
    save_json=True
    single_cls=False
    augment=False
    verbose=False
    weights=None
    dataloader=None
    save_dir=Path('.'),  # for saving images  
    save_txt=False  # for auto-labelling 
    save_hybrid=False  # for hybrid auto-labelling
    save_conf=False  # save auto-label confidences
    plots=True
    wandb_logger=None
    compute_loss=None
    half_precision=False
    trace=False
    is_coco=False
    v5_metric=False

    # load model and set device
    set_logging()
    device = select_device(args.device, batch_size=batch_size)
  
    # Directories
    save_dir = Path(increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load FP32 model
    
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)    # check img_size

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()
    
    # Eval
    model.eval().to(device)

    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    
    nc = 1 if single_cls else int(data['nc'])        # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    confusion_matrix = ConfusionMatrix(nc=nc) 

    model.names = data['names']
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.mdList)}
    coco91class = coco80_to_coco91_class()

    # Logging
    log_imgs = 0
    seen = 0

    # Dataloader
    if device.type != 'cpu':
            model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(model.parameters())))  # run once

    task = args.task if args.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    dataloader = create_dataloader(data[task], imgsz, batch_size, gs, args, pad=0.5, rect=True,
                                prefix=colorstr(f'{task}: '))[0]

    
    # display
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    print(s)
    pbar = tqdm(dataloader, desc=s)
    

    for batch_i, (img, targets, paths, shapes) in enumerate(pbar):
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        
        nb, _, height, width = img.shape   # batch size, channels, height, width     

        with torch.no_grad():

            # Run model
            t = time_synchronized()
            out, train_out = model(img)        # inference and training outputs
            # print(out)
            # input()
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls
            
            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling

            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t
            
        # Statistics per image
        for si, pred in enumerate(out):
        
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
        
            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None
            
            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)

            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            
            # Append statistics (correct, conf, pcls, tcls)                        
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
         
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()

            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()
    
        # Compute statistics
        _stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        
        if len(_stats) and _stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*_stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(_stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        pbar.set_description(str(pf % ('all', seen, nt.sum(), mp, mr, map50, map)))


    # Print speeds  
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})


    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = 'dataset/coco_dataset/annotations_trainval2017/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval._stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    model.float()  # for training
    
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    print(s)
    print(f"Results saved to {save_dir}{s}")

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Converter')
    # parser.add_argument('--type', default="convert", type=str, help='train or convert')
    parser.add_argument('--modeltype', default="yolov7", type=str, help='train or convert')
    parser.add_argument('--weightRoot', default="./weights/yolov7/", type=str, help='weight storage dir')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--data', type=str, default='dataset/coco_dataset/coco.yaml', help='*.data path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')

    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--cfg', type=str, default='./cfg/deploy/yolov7.yaml', help='model.yaml')
    args = parser.parse_args()

    
    weightRoot = args.weightRoot
    args.cfg = check_file(args.cfg)  # check file
    args.data = check_file(args.data) # coco.yaml
    set_logging()

    #load yolov7 model
    device = select_device(args.device)
    model = torch.load(weightRoot + "yolov7.pt", map_location=torch.device('cuda'))['model']
    model.float() 
    print(model)
    input()

################################### process #############################################

    # eval_accuracy= testing(args.data, model)
    
    # rebuild_Model = getModel(model, args.modeltype, cfg = args.cfg)
    rebuild_Model = modelBuilder(model, args.modeltype, args.cfg)
    print(rebuild_Model)
    input()
 
    eval_accuracy, _, _  = testing(args.data, rebuild_Model)

    # out_model = testEval(resModel, args.modeltype, args.data, w_quantBitwidth)

















    # onnx export
    # dummy_input = torch.randn(32, 3, 256, 256, device="cpu") 
    # torch.onnx.export(model, dummy_input, "yolov7_change_1.onnx", opset_version = 11)
    # torch.onnx.export(resModel, dummy_input, "yolov7_origin_1.onnx", opset_version = 11)


    # #check size
    # gs = int(max(model.stride))  # grid size (max stride)
    # args.img_size = [check_img_size(x, gs) for x in args.img_size]

    # # #onnx ouput
    # f = weightRoot.replace('.pt', '.onnx')
    # img = torch.zeros(args.batch_size, 3, *args.img_size).to(device)

    # dummy_input = torch.randn(32, 3, 3, 3, device="cpu")    
    # torch.onnx.export(resModel, img, "yolov7_change_0.onnx", opset_version = 11)
    # torch.onnx.export(model, img, "yolov7_origin_0.onnx", opset_version = 11)
    