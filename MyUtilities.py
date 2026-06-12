import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from IPPy import utilities 
import torch
import matplotlib.pyplot as plt
from torch import nn

class MayoDataset(Dataset):
    def __init__(self, data_path, data_shape_HR, data_shape_LR, noise_level=0.1):
        super().__init__()
        self.data_path = data_path
        self.data_shape_HR = data_shape_HR
        self.data_shape_LR = data_shape_LR
        self.noise_level = noise_level
        self.fname_list = glob.glob(f'{data_path}/*/*.png')

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        img_path = self.fname_list[idx] 
        hResolution = Image.open(img_path).convert('L') 
        hResolution = transforms.Compose([ 
            transforms.ToTensor(), 
            transforms.Resize(self.data_shape_HR), 
        ])(hResolution)

        lResolution = transforms.Resize(self.data_shape_LR)(hResolution) #Downsampling dell'immagine ad alta risoluzione 
        with torch.no_grad():
            lResolution += utilities.gaussian_noise(lResolution, self.noise_level) #Aggiunta di rumore gaussiano all'immagine a bassa risoluzione ⚠️⚠️⚠️
        lResolution = transforms.Resize(self.data_shape_HR)(lResolution) #Upsampling dell'immagine a bassa risoluzione alla stessa dimensione dell'immagine ad alta risoluzione

        return hResolution, lResolution
    
class MyTrainer():
    def __init__(self, model, train_loader, optimizer, num_epochs, scheduler, loss_fn=torch.nn.MSELoss(), saveModel=False, modelName="model", validation_loader=None, test_loader=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        model.to(self.device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.saveModel = saveModel
        self.modelName = modelName
        self.test_loader = test_loader
        self.validation_loader = validation_loader

    def evaluate(self, samples, grad = False):
        x, y = samples # (x, y): x = grandTruth, y = LowResolution x = [x1,x2,x3....], y = [y1,y2,y3....]
        x = x.to(self.device)
        y = y.to(self.device)
        
        if grad:
            x_pred = self.model(y) 
        else:
            with torch.no_grad():
                x_pred = self.model(y)

        loss = self.loss_fn(x_pred, x) 
        return loss, y, x_pred

    def test(self, evalMetrich = None): 
        print(f'🔁 : Testing=> {evalMetrich.__name__ if evalMetrich else "Loss function"}')
        CumulativeEvaluation = 0.0
        samplesCounter = 0

        for testIndex, testSamples in enumerate(self.test_loader):
            print(f'🔁 : test batch: {testIndex}', end='\r')
            test_Loss, _, modelPredictions = self.evaluate(testSamples, grad=False)

            if evalMetrich:
                for i in range(len(modelPredictions)):

                    x = testSamples[0][i].cpu().numpy()
                    x_pred_sample = modelPredictions[i].cpu().numpy()

                    if evalMetrich.__name__ == 'structural_similarity':
                        x = x.squeeze() 
                        x_pred_sample = x_pred_sample.squeeze() 

                    CumulativeEvaluation += evalMetrich(x, x_pred_sample, data_range=1.0)
                    samplesCounter += 1

            else:
                CumulativeEvaluation += test_Loss.item()

        print(f'⚠️: Average {evalMetrich.__name__ if evalMetrich else "Loss function"} on Test Set: {CumulativeEvaluation/samplesCounter if evalMetrich else CumulativeEvaluation/len(self.test_loader):.4f}')

    def train(self):
        loss_history = []
        validation_Loss_history = []

        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Cumulative training loss, Validation Loss')
        line, = ax.plot([], [], 'b-')
        val_line, = ax.plot([], [], 'r-')

        for epoch in range(self.num_epochs):
            cumulativeLoss = 0.0
            for trainIndex, trainSamples in enumerate(self.train_loader):
                # sample = (hREsolution, lResolution)
                print(f'⚠️: Batch {trainIndex}/{len(self.train_loader)}', end='\r')

                loss, _, _ = self.evaluate(trainSamples, grad=True)
                cumulativeLoss += loss.item()
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.scheduler.step()

            loss_history.append(cumulativeLoss / len(self.train_loader))

            if self.validation_loader is not None:
                print(f'🔁 : Evaluating on validation set...', end='\r')
                cumulativeValidationLoss = 0
                for validationIndex, validationSamples in enumerate(self.validation_loader):
                    print(f'🔁 : validation batch: {validationIndex}', end='\r')
                    validation_Loss, _, _ = self.evaluate(validationSamples, grad=False)
                    cumulativeValidationLoss += validation_Loss
                validation_Loss_history.append(validation_Loss.item()/len(self.validation_loader))
                val_line.set_ydata(validation_Loss_history)
                val_line.set_xdata(range(len(validation_Loss_history)))
                val_line.set_label('Validation Loss')

            line.set_xdata(range(len(loss_history)))
            line.set_ydata(loss_history)
            line.set_label('Cumulative batch Loss')
            ax.legend()
            ax.relim()
            ax.autoscale_view()

            fig.savefig(f"{self.modelName}_Training_Loss_Plot.png")
            print(f'✅: Saved loss plot: {self.modelName}_Training_Loss_Plot.png')

            print()
            print(f'👉 Epoch {epoch}, Loss: {loss.item()}, LR: {self.scheduler.get_last_lr()}', end='\r')
            print()

        plt.close(fig)

        if self.saveModel:
            print('⚠️: Saving model: ', self.modelName)
            torch.save(self.model.state_dict(), f'{self.modelName}.pth')

class FourierLoss(nn.Module):
    def forward(self, x, y):
        fx = torch.fft.fft2(x, norm='ortho')
        fy = torch.fft.fft2(y, norm='ortho')
        return torch.mean(torch.abs(fx - fy))
    
class SSIMLoss(nn.Module): # it compares local patches based on luminance, contrast, and structure. Well correlated with human perception.
    def __init__(self, c1=0.01 ** 2, c2=0.03 ** 2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

    def forward(self, x, y):
        mu_x = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mu_y = torch.nn.functional.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
        sigma_x = torch.nn.functional.avg_pool2d(x * x, kernel_size=3, stride=1, padding=1) - mu_x ** 2
        sigma_y = torch.nn.functional.avg_pool2d(y * y, kernel_size=3, stride=1, padding=1) - mu_y ** 2
        sigma_xy = torch.nn.functional.avg_pool2d(x * y, kernel_size=3, stride=1, padding=1) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)) / ((mu_x ** 2 + mu_y ** 2 + self.c1) * (sigma_x + sigma_y + self.c2) + 1e-8)
        return 1.0 - ssim_map.mean()
    
class Losses():
    def FL_SSIM(self,FL=0.5, SSIM=0.5):
        return lambda x, y: FL * FourierLoss()(x, y) + SSIM * SSIMLoss()(x, y)
    
    def MSE_SSIM_FL(self, MSE=0.3, FL=0.3, SSIM=0.3):
        return lambda x, y: MSE * torch.nn.MSELoss()(x, y) + FL * FourierLoss()(x, y) + SSIM * SSIMLoss()(x, y)