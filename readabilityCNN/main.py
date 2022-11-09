import datetime
import os
import time

import torch
from torch import nn
from torchvision.utils import save_image

from dataloader import get_loader
from model import ReadabilityCNN
from options import get_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(opts):
    # Dirs
    log_dir = os.path.join("readabilityCNN", "experiments", opts.experiment_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    samples_dir = os.path.join(log_dir, "samples")
    logs_dir = os.path.join(log_dir, "logs")

    #Loss
    MSELoss = torch.nn.MSELoss().to(device)

    # Path to data
    image_dir = os.path.join("./",opts.data_root, opts.dataset_name, "image")
    attribute_path = os.path.join("./",opts.data_root, opts.dataset_name, "mdAttributes.txt")
    font_readability_path = os.path.join("./",opts.data_root, "readability.csv")

    # Dataloader
    train_dataloader = get_loader(image_dir, attribute_path, font_readability_path,
                                dataset_name="explor_all",
                                image_size=64,
                                n_style=4,
                                batch_size=64, binary=False,
                                train_num=110, val_num=24)
    test_dataloader = get_loader(image_dir, attribute_path, font_readability_path,
                                dataset_name="explor_all",
                                image_size=64,
                                n_style=4, batch_size=8,
                                mode='test', binary=False,
                                train_num=110, val_num=24)

    #Model
    readabilityCNN = ReadabilityCNN()

    if opts.multi_gpu:
        readabilityCNN = nn.DataParallel(readabilityCNN)
    readabilityCNN = readabilityCNN.to(device)

    #optimizer
    optimizer = torch.optim.Adam(readabilityCNN.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))

    # Resume training
    if opts.init_epoch > 1:
        readabilityFile = os.path.join(checkpoint_dir, f"readabilityCNN_{opts.init_epoch}.pth")

        readabilityCNN.load_state_dict(torch.load(readabilityFile))

    prev_time = time.time()
    logfile = open(os.path.join(log_dir, "loss_log.txt"), 'w')
    val_logfile = open(os.path.join(log_dir, "val_loss_log.txt"), 'w')

    for epoch in range(opts.init_epoch, opts.n_epochs+1):
        for batch_idx, batch in enumerate(train_dataloader):
            img_A = batch['img_A'].to(device)
            realReadabilityScore = (batch['readabilityScore'].to(device)).float()

            realReadabilityScore = realReadabilityScore.reshape((realReadabilityScore.shape[0],1))

            #forward
            predictedReadabilityScore = readabilityCNN(img_A)

            #loss
            readabilityLoss = MSELoss(predictedReadabilityScore, realReadabilityScore)

            optimizer.zero_grad()
            readabilityLoss.backward(retain_graph=True)
            optimizer.step()

            #Console output
            batches_done = (epoch - opts.init_epoch) * len(train_dataloader) + batch_idx
            batches_left = (opts.n_epochs - opts.init_epoch) * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left*(time.time() - prev_time))
            prev_time = time.time()

            message = (
                f"Epoch: {epoch}/{opts.n_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, ETA: {time_left}, "
                f"readabilityCNN loss: {readabilityLoss.item():.6f}"
            )

            print(message)
            logfile.write(message + '\n')
            logfile.flush()

            if batches_done % opts.log_freq == 0:
                img_sample = img_A.data
                save_file = os.path.join(logs_dir, f"epoch_{epoch}_batch_{batches_done}.png")
                save_image(img_sample, save_file, nrow=8, normalize=True)

            #Validation
            if batches_done % opts.sample_freq == 0:
                with torch.no_grad():
                    val_readabilityLoss = torch.zeros(1).to(device)
                    for val_idx, val_batch in enumerate(test_dataloader):
                        if val_idx == 20:  # only validate on first 20 batches, you can change it
                            break

                        val_img_A = val_batch['img_A'].to(device)
                        val_realReadabilityScore = val_batch['readabilityScore'].to(device)

                        val_predictedReadabilityScore = readabilityCNN(val_img_A)

                        val_readabilityLoss += MSELoss(val_predictedReadabilityScore, val_realReadabilityScore)

                        img_sample = val_img_A.data
                        save_file = os.path.join(samples_dir, f"epoch_{epoch}_idx_{val_idx}.png")
                        save_image(img_sample, save_file, nrow=8, normalize=True)

                    val_readabilityLoss =  val_readabilityLoss / 20
                    val_msg = (
                        f"Epoch: {epoch}/{opts.n_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, "
                        f"MESLoss: {val_readabilityLoss.item(): .6f}"
                    )
                    val_logfile.write(val_msg + "\n")
                    val_logfile.flush()

        if opts.check_freq > 0 and epoch % opts.check_freq == 0:
            readabilityFile = os.path.join(checkpoint_dir, f"readabilityCNN_{epoch}.pth")

            torch.save(readabilityCNN.state_dict(), readabilityFile)

def main():
    parser = get_parser()
    opts = parser.parse_args()

    os.makedirs("experiments", exist_ok=True)

    if opts.phase == 'train':
        # Create directories
        log_dir = os.path.join("readabilityCNN", "experiments", opts.experiment_name)
        os.makedirs(log_dir, exist_ok=True)  # False to prevent multiple train run by mistake
        os.makedirs(os.path.join(log_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoint"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "interps"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "logs"), exist_ok=True)

        print(f"Training on experiment {opts.experiment_name}...")
        # Dump options
        with open(os.path.join(log_dir, "opts.txt"), "w") as f:
            for key, value in vars(opts).items():
                f.write(str(key) + ": " + str(value) + "\n")
        train(opts)
    # elif opts.phase == 'test':
    #     print(f"Testing on experiment {opts.experiment_name}...")
    #     test(opts)
    # elif opts.phase == 'test_interp':
    #     print(f"Testing interpolation on experiment {opts.experiment_name}...")
    #     interp(opts)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()