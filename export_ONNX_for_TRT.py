import argparse
import sys
import time
import yaml

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

from models.experimental import attempt_load
from utils.activations import SiLU
from utils.general import set_logging


def replace_module(module, replaced_module_type, new_module_type, replace_func=None):
    """
    Replace given type in module to a new type. mostly used in deploy.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov7.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[416, 416], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--opset', type=int, default=10, help='opset version')

    opt = parser.parse_args()

    print(opt)
    set_logging()
    t = time.time()

    opt.grid = True
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1

    device = torch.device('cpu')
    model = attempt_load(opt.weights, map_location=device)
    labels = model.names
    model.eval()
    with torch.no_grad():
        model.fuse()

    replace_module(model, nn.SiLU, SiLU)

    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)
    model.model[-1].export = not opt.grid
    model(img)  # dry run

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        file_name = opt.weights[:-len(opt.weights.split(".")[-1])-1]
        f = file_name + f'_{opt.img_size[0]}x{opt.img_size[1]}.onnx'  # filename
        
        model.model[-1].concat = True
        input_names = ["input_0"]
        output_names = ["output_0"]   # , "output_1", "output_2"]
        torch.onnx.export(model,
                          img,
                          f,
                          verbose=False,
                          opset_version=opt.opset,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=None)
        print("export end")

        print("check")
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        try:
            import onnxsim
            print('\nStarting to simplify ONNX...')
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')

        onnx.save(onnx_model, f)
        print('ONNX export success, saved as %s' % f)

        with open(file_name + f"_{opt.img_size[0]}x{opt.img_size[1]}.yaml", "w") as fp:
            yaml.dump({
                "input_name": input_names,
                "output_name": output_names,
                "names": labels,
                "img_size": opt.img_size[0],
                "batch_size": opt.batch_size
            }, fp, yaml.Dumper)
            print(f"params saved to {file_name + 'yaml'}")

        print("")
        print("############# - msg - ##############")
        print(f"input names   : {input_names}")
        print(f"output names  : {output_names}")
        try:
            print(f"output shape  : {model(img).shape}")
        except:
            print(f"output shape  : {[m.shape for m in model(img)]}")
        print(f"img size      : {opt.img_size}")
        print(f"batch size    : {opt.batch_size}")
        print(f"names         : {labels}")

    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
