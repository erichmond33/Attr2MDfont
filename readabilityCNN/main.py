import datetime
import os
import time
import random

import torch
from torch import nn
from torchvision.utils import save_image
from sklearn.metrics import explained_variance_score

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
                                batch_size=opts.train_batch_size, binary=False,
                                train_num=110, val_num=24)
    test_dataloader = get_loader(image_dir, attribute_path, font_readability_path,
                                dataset_name="explor_all",
                                image_size=64,
                                n_style=4, batch_size=opts.test_batch_size,
                                mode='test', binary=False,
                                train_num=110, val_num=24)

    #Model
    readability_CNN = ReadabilityCNN(dropout=opts.dropout)

    if opts.multi_gpu:
        readability_CNN = nn.DataParallel(readability_CNN)
    readability_CNN = readability_CNN.to(device)

    #optimizer
    optimizer = torch.optim.Adam(readability_CNN.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))

    # Resume training
    if opts.init_epoch > 1:
        readability_file = os.path.join(checkpoint_dir, f"readability_CNN_{opts.init_epoch}.pth")
        readability_CNN.load_state_dict(torch.load(readability_file))

    prev_time = time.time()
    logfile = open(os.path.join(log_dir, "loss_log.txt"), 'w')
    val_logfile = open(os.path.join(log_dir, "val_loss_log.txt"), 'w')
    real_predictions_logfile = open(os.path.join(log_dir, "real_predictions_log.txt"), 'w')

    for epoch in range(opts.init_epoch, opts.n_epochs+1):
        for batch_idx, batch in enumerate(train_dataloader):
            img_A = batch['img_A'].to(device)
            real_readability_score = (batch['readability_score'].to(device)).float()

            real_readability_score = real_readability_score.reshape((real_readability_score.shape[0],1))

            #forward
            predicted_readability_score = readability_CNN(img_A)

            #loss
            readability_loss = MSELoss(predicted_readability_score, real_readability_score)
            # readability_loss = opts.lambda_readability_CNN * MSELoss(predicted_readability_score, real_readability_score)

            optimizer.zero_grad()
            readability_loss.backward(retain_graph=True)
            optimizer.step()

            #Console output
            batches_done = (epoch - opts.init_epoch) * len(train_dataloader) + batch_idx
            batches_left = (opts.n_epochs - opts.init_epoch) * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left*(time.time() - prev_time))
            prev_time = time.time()

            message = (
                f"Epoch: {epoch}/{opts.n_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, ETA: {time_left}, "
                f"readability_CNN loss: {readability_loss.item():.6f}"
            )

            print(message)
            logfile.write(message + '\n')
            logfile.flush()

            real_predictions_message = (
                f"Epoch: {epoch}/{opts.n_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, ETA: {time_left}, "
                f"real_readability_score: {real_readability_score}, "
                f"predicted_readability_score: {predicted_readability_score}"
            )

            real_predictions_logfile.write(real_predictions_message + '\n')
            real_predictions_logfile.flush()

            if batches_done % opts.log_freq == 0:
                img_sample = img_A.data
                save_file = os.path.join(logs_dir, f"epoch_{epoch}_batch_{batches_done}.png")
                save_image(img_sample, save_file, nrow=8, normalize=True)

            #Validation
            if batches_done % opts.sample_freq == 0:
                with torch.no_grad():
                    readability_CNN.eval()
                    val_readability_loss = torch.zeros(1).to(device)
                    for val_idx, val_batch in enumerate(test_dataloader):
                        if val_idx == 20:  # only validate on first 20 batches, you can change it
                            break

                        val_img_A = val_batch['img_A'].to(device)
                        val_real_readability_score = val_batch['readability_score'].to(device)

                        val_real_readability_score = val_real_readability_score.reshape((val_real_readability_score.shape[0],1))

                        val_predicted_readability_score = readability_CNN(val_img_A)

                        val_readability_loss += MSELoss(val_predicted_readability_score, val_real_readability_score)

                        img_sample = val_img_A.data
                        save_file = os.path.join(samples_dir, f"epoch_{epoch}_idx_{val_idx}.png")
                        save_image(img_sample, save_file, nrow=8, normalize=True)

                    readability_CNN.train()

                    val_readability_loss =  val_readability_loss / 20
                    val_msg = (
                        f"Epoch: {epoch}/{opts.n_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, "
                        f"MESLoss: {val_readability_loss.item(): .6f}, "
                        f"Varience: {explained_variance_score(val_real_readability_score.cpu(), val_predicted_readability_score.cpu()): .2f}"
                    )
                    val_logfile.write(val_msg + "\n")
                    val_logfile.flush()

        if opts.check_freq > 0 and epoch % opts.check_freq == 0:
            readability_file = os.path.join(checkpoint_dir, f"readability_CNN_{epoch}.pth")

            torch.save(readability_CNN.state_dict(), readability_file)

def test_one_epoch(opts, test_logfile, real_predictions_logfile, test_epoch,
                   checkpoint_dir, results_dir,
                   readability_CNN, MSELoss,
                   test_dataloader):
    print(f"Testing epoch: {test_epoch}")

    readability_file = os.path.join(checkpoint_dir, f"readability_CNN_{test_epoch}.pth")
    readability_CNN.load_state_dict(torch.load(readability_file))

    with torch.no_grad():
        for test_idx, test_batch in enumerate(test_dataloader):
            img_A = test_batch['img_A'].to(device)
            real_readability_score = (test_batch['readability_score'].to(device)).float()

            real_readability_score = real_readability_score.reshape((real_readability_score.shape[0],1))

            #forward
            predicted_readability_score = readability_CNN(img_A)

            #loss
            readability_loss = MSELoss(predicted_readability_score, real_readability_score)

            img_sample = img_A.data
            save_file = os.path.join(results_dir, f"test_{test_epoch}_idx_{test_idx}.png")
            save_image(img_sample, save_file, nrow=8, normalize=True)

            real_predictions_message = (
                f"Epoch: {test_epoch}/{opts.n_epochs}, Batch: {test_idx}/{len(test_dataloader)}, "
                f"readability_CNN loss: {readability_loss.item():.6f}, "
                f"real_readability_score: {real_readability_score}, "
                f"predicted_readability_score: {predicted_readability_score}"
            )

            real_predictions_logfile.write(real_predictions_message + '\n')
            real_predictions_logfile.flush()

        test_msg = (
            f"Epoch: {test_epoch}/{opts.n_epochs}, "
            f"readability_CNN Loss: {readability_loss.item(): .6f}, "
            f"Varience: {explained_variance_score(real_readability_score.cpu(), predicted_readability_score.cpu()): .2f}"
        )
        print(test_msg)
        test_logfile.write(test_msg + "\n")
        test_logfile.flush()

def test(opts):
    # Dirs
    log_dir = os.path.join("readabilityCNN", "experiments", opts.experiment_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    results_dir = os.path.join(log_dir, "results")

    #Loss
    MSELoss = torch.nn.MSELoss().to(device)

    # Path to data
    image_dir = os.path.join("./",opts.data_root, opts.dataset_name, "image")
    attribute_path = os.path.join("./",opts.data_root, opts.dataset_name, "mdAttributes.txt")
    font_readability_path = os.path.join("./",opts.data_root, "readability.csv")

    # Dataloader
    test_dataloader = get_loader(image_dir, attribute_path, font_readability_path,
                                dataset_name="explor_all",
                                image_size=64,
                                n_style=4, batch_size=opts.test_batch_size,
                                mode='test', binary=False,
                                train_num=110, val_num=24)

    # Model
    readability_CNN = ReadabilityCNN()

    if opts.multi_gpu:
        readability_CNN = nn.DataParallel(readability_CNN)
    readability_CNN = readability_CNN.to(device)

    # Testing
    test_logfile = open(os.path.join(log_dir, f"test_loss_log_{opts.test_epoch}.txt"), 'w')
    real_predictions_logfile = open(os.path.join(log_dir, "pred_log_test.txt"), 'w')

    readability_CNN.eval()

    if opts.test_epoch == 0:
        for test_epoch in range(opts.check_freq, opts.n_epochs+1, opts.check_freq):
            test_one_epoch(opts, test_logfile, real_predictions_logfile, test_epoch,
                            checkpoint_dir, results_dir,
                            readability_CNN, MSELoss,
                            test_dataloader)
    else:
        test_one_epoch(opts, test_logfile, real_predictions_logfile, test_epoch,
                        checkpoint_dir, results_dir,
                        readability_CNN, MSELoss,
                        test_dataloader)

    readability_CNN.train()


def random_search(opts):
    # Hyperparameters
    lambdas = [0.1, 0.5, 1]
    batchSizes = [16, 32, 64]
    # epochs = [10, 25]
    dropout = [0.5, .75, .9]

    # Random search
    for i in range(100):
        opts.lambda_readability_CNN = random.choice(lambdas)
        opts.batch_size = random.choice(batchSizes)
        opts.n_epochs = random.randint(1, 100)
        opts.dropout = random.choice(dropout)

        opts.experiment_name = f"lambda_{opts.lambda_readability_CNN}_batchSize_{opts.batch_size}_epoch_{opts.n_epochs}_dropout_{opts.dropout}"

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
        test(opts)

        # Saving storage space by only leaving the loss logs
        os.system(f"rm -rf {os.path.join(log_dir, 'checkpoint')}")
        os.system(f"rm -rf {os.path.join(log_dir, 'interps')}")
        os.system(f"rm -rf {os.path.join(log_dir, 'logs')}")
        os.system(f"rm -rf {os.path.join(log_dir, 'results')}")
        os.system(f"rm -rf {os.path.join(log_dir, 'samples')}")
        os.system(f"rm -rf {os.path.join(log_dir, 'opts.txt')}")


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
    elif opts.phase == 'test':
        print(f"Testing on experiment {opts.experiment_name}...")
        test(opts)
    elif opts.phase == "random_search":
        print(f"Random search on experiment {opts.experiment_name}...")
        random_search(opts)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()