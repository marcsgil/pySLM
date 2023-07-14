import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pathlib
import matplotlib.pyplot as plt

from optics import log


class ImageNet(nn.Module):
    def __init__(self, img_shape, nb_categories):
        """
        Construct a new image categorizing network.

        :param img_shape: The data shape of the image, starting with the channels
        """
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            # nn.Conv2d(1, 1, kernel_size=3, stride=1),
            # nn.Dropout(p=0.25),
            # nn.ReLU(),
            # nn.Flatten(),
            # nn.Linear(((img_shape[-2]-3)//1 + 1)**2, nb_categories),
            nn.Flatten(),
            nn.Linear(img_shape[-2] * img_shape[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(64, nb_categories),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


if __name__ == '__main__':
    batch_size = 64
    nb_epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using PyTorch device {device}...')

    model_path = pathlib.Path('scripted_model.pt')

    log.info('Loading training data...')
    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
    log.info('Loading test data...')
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    try:
        raise RuntimeError()
        model = torch.jit.load(model_path)
        model.eval()  # switch to evaluation mode
        log.info(f'Loaded model from {model_path}, using it right away:\n{model}')
    except RuntimeError as fe:
        log.info(f'Could not load model from {model_path}, creating a new one.')
        initial_batch = next(iter(train_dataloader))
        model = ImageNet(initial_batch[0].shape, 10).to(device=device)
        model = torch.jit.trace(model, initial_batch[0].to(device=device))
        log.info(f'Model:\n{model}')

        # softmax = nn.LogSoftmax(dim=1)
        # def cost_function(logits, target_batch):
        #     label_probabilities_batch = softmax(logits)
        #     target_probabilities_batch = torch.zeros_like(logits)
        #     for _, target_label in enumerate(target_batch):
        #         target_probabilities_batch[_, target_label] = 1
        #     return torch.linalg.norm(label_probabilities_batch - target_probabilities_batch) ** 2
        cost_function = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        def show_progress(epoch):
            # Check loss on test data
            model.eval()
            with torch.inference_mode():
                test_loss, correct_fraction = 0.0, 0.0
                for img_batch, target_batch in test_dataloader:
                    target_batch = target_batch.to(device=device)
                    predictions = model(img_batch.to(device=device))
                    test_loss += cost_function(predictions, target_batch) / len(test_dataloader)
                    correct_fraction += (predictions.argmax(1) == target_batch).type(torch.float).mean().item() / len(test_dataloader)
                log.info(f'Epoch {epoch + 1}/{nb_epochs}: test data loss: {test_loss:0.6f}. Correct fraction {correct_fraction * 100:0.1f}%')


        log.info('Training...')
        show_progress(-1)
        for epoch in range(nb_epochs):
            log.debug(f'Epoch {epoch}/{nb_epochs}...')
            model.train()
            for img_batch, target_batch in train_dataloader:
                loss = cost_function(model(img_batch.to(device=device)), target_batch.to(device=device))
                model.zero_grad()
                loss.backward()
                optimizer.step()
            show_progress(epoch)

        log.info(f'Saving trained model to {model_path}...')
        model_scripted = torch.jit.script(model)  # Export as TorchScript
        model_scripted.save(model_path)

    log.info('Displaying...')

    label_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for _, ax in enumerate(axs.ravel()):
        img, actual_label = training_data[100 + _]
        predicted_label = torch.argmax(model(img.to(device=device))).item()
        eq_sign, eq_color = ('=', 'g') if predicted_label == actual_label else ('≠', 'r')
        ax.set_title(f'{label_names[predicted_label]}{eq_sign}{label_names[actual_label]}', color=eq_color)
        ax.axis("off")
        ax.imshow(img.squeeze(), cmap="gray")

    log.info('Done!')
    plt.show()
