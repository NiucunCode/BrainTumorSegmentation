"""
Train model.
"""
import sys
import argparse

from dataset import *
from utils import *


def train(args):

    # initialize
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # load data
    train_loader = data_loader(args, mode='train')
    valid_loader = data_loader(args, mode='valid')

    # load model
    net, optimizer, best_score, start = load_model(args, mode='train')

    # Train
    for epoch in range(start, start + args.epochs):
        print('\nTraining status: Epoch [', epoch, '/', start + args.epochs, ']')
        loss = 0
        net.train(mode=True)
        torch.set_grad_enabled(True)

        # dynamically adjust learning rate
        lr = args.lr * (0.5 ** (epoch // 4))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for i, (inputs, targets, paths) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = net(inputs)

            optimizer.zero_grad()

            dice_loss = dice_coefficient_loss(preds, targets)
            loss += float(dice_loss)

            dice_loss.backward()
            optimizer.step()

        loss /= (i + 1)
        # print log
        sys.stdout.write(
            "\n[Epoch: %d/%d]: Dice Coefficient Loss= %f" % (epoch, start + args.epochs, loss)
        )

        # validation
        print('\nValidation...')
        loss = 0
        net.eval()
        torch.set_grad_enabled(False)
        for i, (inputs, targets, paths) in enumerate(valid_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = net(inputs)

            dice_loss = dice_coefficient_loss(preds, targets, backprop=False)
            loss += float(dice_loss)

        loss /= (i + 1)
        # print log
        sys.stdout.write(
            "\n[Epoch: %d/%d]: Dice Coefficient Loss= %f" % (epoch, start + args.epochs, loss)
        )

        # save model
        score = 1 - loss
        if score > best_score:
            print("Saving result...")
            best_score = score
            checkpoint = Checkpoint(net, optimizer, epoch, best_score)
            checkpoint.save(os.path.join(args.ckpt_root, args.model + '.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=bool, default=False, help="Model Trianing resume. True for load models.")
    parser.add_argument("--model", type=str, default='unet', help="Choose the model.")
    parser.add_argument("--in_channel", type=int, default=1, help="The number of input image.")
    parser.add_argument("--class_num", type=int, default=2, help="The dim of uNet output.")
    parser.add_argument("--batch_size", type=int, default=80, help="The batch size to load the data.")
    parser.add_argument("--epochs", type=int, default=30, help="The number of epochs of training")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="Drop-out Rate")
    parser.add_argument("--lr", type=float, default=0.001, help="The learning rate.")
    parser.add_argument("--data", type=str, default="complete", help="Data prediction type.")
    parser.add_argument("--space", type=int, default=50, help="Space to distinguish labels.")
    parser.add_argument("--use_gpu", type=bool, default=False, help="True to use gpu, False to use cpu.")
    parser.add_argument("--complete_threshold", type=float, default=0.05, help="Threshold of complete prediction.")
    parser.add_argument("--complete_rate", type=float, default=0.66, help="Data rate of complete prediction.")
    parser.add_argument("--core_threshold", type=float, default=0.05, help="Threshold of core prediction.")
    parser.add_argument("--core_rate", type=float, default=0.66, help="Data rate of core prediction.")
    parser.add_argument("--enhancing_threshold", type=float, default=0.01, help="Threshold of enhancing prediction.")
    parser.add_argument("--enhancing_rate", type=float, default=0.70, help="Data rate of enhancing prediction.")
    parser.add_argument("--img_root", type=str, default="train/image_FLAIR", help="The root of training images.")
    parser.add_argument("--label_root", type=str, default="train/label", help="The root of training labels")
    parser.add_argument("--output_root", type=str, default="output/prediction", help="The root of result predictions")
    parser.add_argument("--ckpt_root", type=str, default="checkpoint", help="The root of checkpoint files")
    args = parser.parse_args()

    try:
        os.makedirs(args.ckpt_root, exist_ok=True)
        os.makedirs(args.output_root, exist_ok=True)
    except:
        pass

    train(args)
