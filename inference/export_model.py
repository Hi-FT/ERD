# coding=utf-8
"""
@Data: 2020/08/31
@Author: 算影
@Email: wangmang.wm@alibaba-inc.com
"""
import os
import argparse
import io
import sys

import cv2
import numpy as np

# --------------------------- add mmcv, mmdet, cocoapi to path -----------------------
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', 'cocoapi/pycocotools'))
sys.path.insert(0, os.path.join(this_dir, '..', 'mmcv'))
sys.path.insert(0, os.path.join(this_dir, '..'))

import mmcv
import torch
import onnx
import onnxruntime
from mmcv.runner import load_checkpoint
from onnx import optimizer
from torch.onnx import OperatorExportTypes

from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet pytorch model conversion')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint file')
    parser.add_argument('--shape', type=int, nargs='+', default=[480], help='input image size')
    parser.add_argument('--passes', type=str, nargs='+', help='ONNX optimization passes')
    parser.add_argument('--ts_path', type=str, default='inference/models/ts_models', help='TS model save path')
    parser.add_argument('--ts_save_file', type=str, default='ava_1x.pt', help='TS model save filename')
    parser.add_argument('--onnx_path', type=str, default='inference/models/onnx_models', help='ONNX model save path')
    parser.add_argument('--onnx_save_file', type=str, default='gfl_res2net_101_person.onnx', help='ONNX model save filename')
    args = parser.parse_args()
    return args


def export_onnx_model(model, inputs, passes, dynamic_axes=False):
    """
    Trace and export a model to onnx format.

    Args:
        model (nn.Module):
        inputs (tuple[args]): the model will be called by `model(*inputs)`
        passes (None or list[str]): the optimization passed for ONNX model

    Returns:
        an onnx model
    """
    assert isinstance(model, torch.nn.Module)

    # make sure all modules are in eval mode, onnx may change the training
    # state of the module if the states are not consistent
    def _check_eval(module):
        assert not module.training
    model.apply(_check_eval)

    # export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            if not dynamic_axes:
                torch.onnx.export(
                    model,
                    inputs,
                    f,
                    operator_export_type=OperatorExportTypes.ONNX,
                    opset_version=10,
                    input_names=['INPUT'],
                    export_params=True
                    # verbose=True,  # NOTE: uncomment this for debugging
                )
            else:
                torch.onnx.export(
                    model,
                    inputs,
                    f,
                    operator_export_type=OperatorExportTypes.ONNX,
                    opset_version=10,
                    input_names=['INPUT__0'],
                    output_names=['OUTPUT__0', 'OUTPUT__1', 'OUTPUT__2', 'OUTPUT__3', 'OUTPUT__4',
                                  'OUTPUT__5', 'OUTPUT__6', 'OUTPUT__7', 'OUTPUT__8', 'OUTPUT__9'],
                    dynamic_axes={"INPUT__0": [0],
                                  'OUTPUT__0': [0], 'OUTPUT__1': [0], 'OUTPUT__2': [0], 'OUTPUT__3': [0],
                                  'OUTPUT__4': [0], 'OUTPUT__5': [0], 'OUTPUT__6': [0], 'OUTPUT__7': [0],
                                  'OUTPUT__8': [0], 'OUTPUT__9': [0]},
                    export_params=True
                    # verbose=True,  # NOTE: uncomment this for debugging
                )

            onnx_model = onnx.load_from_string(f.getvalue())

    # # Apply ONNX's Optimization
    # if passes is not None:
    #     all_passes = optimizer.get_available_passes()
    #     assert all(p in all_passes for p in passes), \
    #         'Only {} are supported'.format(all_passes)
    # onnx_model = optimizer.optimize(onnx_model, passes)
    return onnx_model


def export_ts_model(model, imgs):
    """
    Trace and export a model to TorchScript format.

    Args:
        model (nn.Module):
        imgs (tensor): the input to model

    Returns:
        TS model
    """
    assert isinstance(model, torch.nn.Module)

    # make sure all modules are in eval mode
    def _check_eval(module):
        assert not module.training
    model.apply(_check_eval)

    # export the model to TorchScript format
    ts_model = torch.jit.trace(model, imgs)

    return ts_model


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    model = build_detector(cfg.model)
    # print(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # put model on available device
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
    
    model.eval().to(device)
    model.forward = model.forward_dummy  # mmdet/models/detectors/gfl.py::forward_dummy()

    # check evaluation mode
    def _check_eval(module):
        assert not module.training
    model.apply(_check_eval)

    # mock input tensor
    mock_input_tensor = torch.ones(input_shape, dtype=torch.float32, device=device)

    # img_decoded = cv2.imread('demo/1002.jpg', cv2.IMREAD_COLOR)
    # img_decoded = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2RGB)
    # height, width, _ = img_decoded.shape
    # scale = max(height / 800, width / 1000)

    # # keeping aspect ratio resize
    # new_size = int(width / float(scale) + 0.5), int(height / float(scale) + 0.5)
    # img_resized = cv2.resize(img_decoded, dsize=new_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # # pad to square
    # img_padded = np.zeros((800, 1024, 3), dtype=img_resized.dtype)
    # img_padded[:new_size[1], :new_size[0], ...] = img_resized

    # # img_resized = img_resized - np.array([123.675, 116.28, 103.53], dtype=np.float32)
    # # img_resized = img_resized / np.array([58.395, 57.12, 57.375], dtype=np.float32)

    # img_resized = img_padded - np.array([123.675, 116.28, 103.53], dtype=np.float32)
    # img_resized = img_resized / np.array([58.395, 57.12, 57.375], dtype=np.float32)

    # img_resized = img_resized.transpose((2, 0, 1))  # transpose to (C, H, W)
    # mock_input_tensor = torch.from_numpy(img_resized).unsqueeze(0).to(device)

    # img_resized = np.load('img_resized.npy')
    # mock_input_tensor = torch.from_numpy(img_resized).to(device)

    # export to TS model
    ts_model = export_ts_model(model, mock_input_tensor)
    # save TS model to file
    ts_path = args.ts_path
    if not os.path.exists(ts_path):
        os.makedirs(ts_path)
    ts_save_file = os.path.join(ts_path, args.ts_save_file)
    torch.jit.save(ts_model, ts_save_file)

    # # export to ONNX model
    # onnx_model = export_onnx_model(model, (mock_input_tensor, ), args.passes, dynamic_axes=True)
    # # onnx.helper.printable_graph(onnx_model.graph)  # print a human readable representation of the graph
    # # save ONNX model to file
    # onnx_path = args.onnx_path
    # if not os.path.exists(onnx_path):
    #     os.makedirs(onnx_path)
    # onnx_save_file = os.path.join(onnx_path, args.onnx_save_file)
    # onnx.save(onnx_model, onnx_save_file)

    # test consistency
    # pure pth run
    with torch.no_grad():
        outs_pth = model(mock_input_tensor)

    # ts run
    with torch.no_grad():
        outs_ts = ts_model(mock_input_tensor)

    # # onnxruntime run
    # ort_model = onnxruntime.InferenceSession(onnx_save_file)
    # outs_ort = ort_model.run(None, {'INPUT__0': mock_input_tensor.cpu().numpy()})

    print()


if __name__ == '__main__':
    main()
