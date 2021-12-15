# Import Dependencies
import wandb
import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
from model import SqueezeNet
from dataset import get_data_loader


# Training Function
def train(epoch):
    model.train()
    running_loss = AverageMeter()
    running_accuracy = AverageMeter()
    t = tqdm(train_loader)

    for idx, (images, labels) in enumerate(t):
        images = images.to(device)
        labels = labels.to(device)

        prediction = model(images)
        loss = criterion(prediction, labels)
        acc = accuracy(labels.data, prediction.data)

        running_loss.update(loss.item(), images.size(0))
        running_accuracy.update(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        t.set_description(f"Epoch: {epoch}, Train Loss: {running_loss.avg:.3f}, Train Acc: {running_accuracy.avg:.3f}")

    print(f"Epoch: {epoch}")
    print(f"Train Loss: {running_loss.avg:.2f}")
    print(f"Train Acc: {running_accuracy.avg:.2f}")

    return running_loss, running_accuracy


def evaluate(epoch):
    # Evaluate the Model
    model.eval()
    running_accuracy = AverageMeter()
    running_loss = AverageMeter()
    t = tqdm(val_loader)

    for idx, (images, labels) in enumerate(t):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            prediction = model(images)
            loss = criterion(prediction, labels)
            acc = accuracy(labels.data, prediction.data)

            running_loss.update(loss.item(), len(images))
            running_accuracy.update(acc)

            t.set_description(f"Epoch: {epoch}, Val Loss: {running_loss.avg:.3f}, Val Acc: {running_accuracy.avg:.3f}\n")

    print(f"Epoch: {epoch}")
    print(f"Val Loss: {running_loss.avg:.2f}")
    print(f"Val Acc: {running_accuracy.avg:.2f}")

    return running_loss, running_accuracy


def main(args):
    # Define device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print("Device: ", device)

    # Set seed for reproducability
    torch.manual_seed(args.seed)

    # Setup wandb
    wandb.init(project="SqueezeNet-PyTorch")
    wandb.config.update(args)

    # Dataset Transforms
    transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.491399689874, 0.482158419622, 0.446530924224],
                              [0.247032237587, 0.243485133253, 0.261587846975]),
         ])

    # Get dataloaders
    train_loader, val_loader = get_data_loader(args.batch_size, transform, num_workers=args.num_workers)

    # Instantiate the Model
    model = SqueezeNet(in_ch=3, num_classes=10)
    model.to(device)

    wandb.watch(model)

    # Model Loss Function
    criterion = nn.CrossEntropyLoss()

    # Model Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=int(len(train_loader)),
        epochs=args.epochs)



    for epoch in range(args.epochs):
        train_loss, train_acc = train(epoch)
        val_loss, val_acc = evaluate(epoch)

        if val_acc.avg > train_acc.avg:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss.avg,
            }, PATH)

if __name__ == '__main__':
