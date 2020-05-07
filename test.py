"""
Test model.
"""
import argparse

from dataset import *
from utils import *


def test(args):
    # initialize
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # load data
    test_loader = data_loader(args, mode='test')

    # load model
    net = load_model(args, mode='test')
    net.eval()
    torch.set_grad_enabled(False)

    for i, (inputs, paths) in enumerate(test_loader):
        inputs = inputs.to(device)
        preds = net(inputs)

        post_process(args, inputs, preds, paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default='unet', help="Choose the model.")
    parser.add_argument("--batch_size", type=int, default=155, help="The batch size to load the data.")
    parser.add_argument("--data", type=str, default="complete", help="Data prediction type.")
    parser.add_argument("--img_root", type=str, default="test/image_FLAIR", help="The root of training images.")
    parser.add_argument("--output_root", type=str, default="output/prediction", help="The root of result predictions")
    parser.add_argument("--ckpt_root", type=str, default="checkpoint", help="The root of checkpoint files")
    args = parser.parse_args()

    try:
        os.makedirs(args.ckpt_root, exist_ok=True)
        os.makedirs(args.output_root, exist_ok=True)
    except:
        pass

    test(args)