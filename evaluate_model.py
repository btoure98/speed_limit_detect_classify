import torch
import config as cfg
import torchvision.transforms as T
import os

from data import SlsDataset
from torch.utils.data import DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test():
    print("================== Loading model ==================")
    model = torch.load(cfg.model_path)
    model.eval()
    print("model loaded successfully")
    print("================== Instantiating data loader ==================")
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor()])
    dataset = SlsDataset(cfg.test_sls, cfg.test_nosls,
                         cfg.test_random, transform)

    print("Test data contains ", len(dataset), "images")
    print("================== Inference ==================")
    good_predictions = 0
    total = 0
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size,
                            shuffle=True, num_workers= os.cpu_count())
    for batch in dataloader:
        images, labels = batch["image"], batch["label"]
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # Forward Pass
        with torch.no_grad():
            outputs = model(images)
            for (output, label) in zip(outputs, labels):
                total += 1
                if output[output.argmax()] > 0.6:
                    if output.argmax() == label.argmax():
                        good_predictions += 1
                else:
                    if label.sum().item() == 0:
                        good_predictions += 1
        # Get metrics

    print("done!")
    print("================== Getting metrics ==================")
    print("Accuracy : ", "%.2f" % (100*good_predictions/total), "%")

if __name__ == "__main__":
    test()