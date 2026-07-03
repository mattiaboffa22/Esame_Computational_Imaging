import glob
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from IPPy import operators, utilities
import torch
import matplotlib.pyplot as plt
from torch import nn
from IPPy.utilities.metrics import PSNR, SSIM

class MayoDataset(Dataset):
    def __init__(self, data_path, data_shape_HR, data_shape_LR, noise_level=0.1, downscale_factor=2):
        super().__init__()
        self.data_path = data_path
        self.data_shape_HR = data_shape_HR if isinstance(data_shape_HR, tuple) else (data_shape_HR, data_shape_HR)
        self.data_shape_LR = data_shape_LR if isinstance(data_shape_LR, tuple) else (data_shape_LR, data_shape_LR)
        self.noise_level = noise_level
        self.fname_list = glob.glob(f'{data_path}/*/*.png')
        self.downscale_factor = downscale_factor

        
        self.downscaler = operators.DownScaling(
            img_shape=self.data_shape_HR,
            downscale_factor=downscale_factor,
        )

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        img_path = self.fname_list[idx] 
        hResolution = Image.open(img_path).convert('L') 
        hResolution = transforms.Compose([ 
            transforms.ToTensor(), 
            transforms.Resize(self.data_shape_HR), 
        ])(hResolution)

        lResolution = self.downscaler(hResolution.unsqueeze(0)).squeeze(0)
        with torch.no_grad():
            lResolution += utilities.gaussian_noise(lResolution, self.noise_level) #Aggiunta di rumore gaussiano all'immagine a bassa risoluzione ⚠️⚠️⚠️
        lResolution = transforms.Resize(self.data_shape_HR)(lResolution) #Upsampling dell'immagine a bassa risoluzione alla stessa dimensione dell'immagine ad alta risoluzione

        return hResolution, lResolution
    
class MyTrainer():
    def __init__(self, model, train_loader, optimizer, num_epochs, scheduler, loss_fn=torch.nn.MSELoss(), saveModel=False, modelName="model", validation_loader=None, test_loader=None, best_model=False, betchTrainingValidationPrint = 5, resultRoot="./Results"):
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
        self.best_model = best_model
        self.betchTrainingValidationPrint = betchTrainingValidationPrint
        self.modelEvaluationOnTest = []
        self.modelEvaluationOnImages = []
        self.resultRoot = resultRoot

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

    def test(self): 
        print(f'🔄️ :Testing')
        if not self.test_loader:
            print(f'🚨 No one test set.')
            return
        
        # Esegui tutte le metriche di valutazione [PSNR e SSIM] sul test set
        metric_cumulative = {PSNR: 0.0, SSIM: 0.0}

        for testIndex, testSamples in enumerate(self.test_loader):
            print(f'🔁 : test batch: {testIndex}', end='\r')
            print()
            _, _, modelPredictions = self.evaluate(testSamples, grad=False)

            for i in range(len(modelPredictions)):
                x = testSamples[0][i].unsqueeze(0).to(self.device)  # GrandTruth
                x_pred_sample = modelPredictions[i].unsqueeze(0).to(self.device)  # Predizione del modello

                psnr = PSNR(x_pred_sample, x)
                ssim = SSIM(x_pred_sample, x)
                print(f"🔄️: PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}", end='\r')

                metric_cumulative[PSNR] += psnr
                metric_cumulative[SSIM] += ssim
                    
        # Calcola la media delle metriche sul test set
        num_samples = len(self.test_loader.dataset)
        avg_psnr = metric_cumulative[PSNR] / num_samples
        avg_ssim = metric_cumulative[SSIM] / num_samples

        self.modelEvaluationOnTest.append({'PSNR': avg_psnr, 'SSIM': avg_ssim})
        print(f'✅: Average PSNR on Test Set: {avg_psnr:.4f}')
        print(f'✅: Average SSIM on Test Set: {avg_ssim:.4f}')

        self.logTrain()

    def train(self):
        epochs = []
        loss_history = []
        validation_Loss_history = []
        currentBestModelValidationLoss = float('inf')

        fig, ax = plt.subplots()
        ax.set_xlabel(f'Epoch')
        ax.set_ylabel('Losses')
        ax.set_title(f'Cumulative training and validation Loss with: {self.loss_fn.__name__}')
        line, = ax.plot([], [], 'b-')
        val_line, = ax.plot([], [], 'r-')

        for epoch in range(self.num_epochs):

            cumulativeLoss = 0.0 # Cumulative training loss over self.betchTrainingValidationPrint betches
            
            for trainIndex, trainSamples in enumerate(self.train_loader):
                # sample = (hREsolution, lResolution)
                print(f'🔄️ : Training on batch {trainIndex}/{len(self.train_loader)}', end='\r')

                loss, _, _ = self.evaluate(trainSamples, grad=True)
                cumulativeLoss += loss.item()
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                if trainIndex > 0 and trainIndex % self.betchTrainingValidationPrint == 0:
                    
                    loss_history.append(cumulativeLoss / self.betchTrainingValidationPrint)
                    epochs.append((epoch * len(self.train_loader) + trainIndex) / len(self.train_loader))
                    # batch attuale / batch per epoca
                    cumulativeLoss = 0.0

                    if self.validation_loader is not None:
                        print(f'🔄️ : Evaluating on validation set...', end='\r')
                        print()

                        cumulativeValidationLoss = 0 # Cumulative validation loss over all validation set

                        for validationIndex, validationBetch in enumerate(self.validation_loader):
                            print(f'🔄️ : validation batch: {validationIndex}/{len(self.validation_loader)}', end='\r')

                            validation_Loss, _, _ = self.evaluate(validationBetch, grad=False)
                            cumulativeValidationLoss += validation_Loss.item() #⚠️⚠️

                        avarageValidationLoss = cumulativeValidationLoss/len(self.validation_loader)

                        # If the current cumulative validation loss is the minimum validation loss them save this model
                        if self.best_model and  avarageValidationLoss < currentBestModelValidationLoss:
                            print()
                            print(f'⏬: Saving best model: {os.path.join(self.resultRoot, self.modelName)}: {avarageValidationLoss}', end='\r')
                            print()
                            currentBestModelValidationLoss = avarageValidationLoss
                            os.makedirs(self.resultRoot, exist_ok=True)
                            torch.save(self.model.state_dict(), os.path.join(self.resultRoot, f'{self.modelName}.pth'))

                        validation_Loss_history.append(avarageValidationLoss)

                        val_line.set_ydata(validation_Loss_history)
                        val_line.set_xdata(epochs)
                        val_line.set_label(f'Cumulative validation loss on: {len(self.validation_loader)} betches')

                    line.set_xdata(epochs)
                    line.set_ydata(loss_history)
                    line.set_label(f'Cumulative training loss on: {self.betchTrainingValidationPrint} betches')
                    ax.legend()
                    ax.relim()
                    ax.autoscale_view()

                    os.makedirs(self.resultRoot, exist_ok=True)
                    fig.savefig(os.path.join(self.resultRoot, f'{self.modelName}_Training_Loss_Plot.png'))
                    print(f'✅: Saved loss plot: {os.path.join(self.resultRoot, f"{self.modelName}_Training_Loss_Plot.png")}', end='\r')
                    print()

            self.scheduler.step()

            print()
            print(f'👉 Epoch {epoch}, Loss: {loss.item()}, LR: {self.scheduler.get_last_lr()}', end='\r')
            print()

        plt.close(fig)

        if self.saveModel and not self.best_model:
            print('✅: Saving model at end training: ', self.modelName)
            torch.save(self.model.state_dict(), f'{self.modelName}.pth')
        
        self.logTrain()

    def testOnImage(self, path, data_shape_HR, data_shape_LR, noise_level):
        hResolution = Image.open(path).convert('L') 
        hResolution = transforms.Compose([ 
            transforms.ToTensor(), 
            transforms.Resize(data_shape_HR), 
        ])(hResolution)

        lResolution = transforms.Resize(data_shape_LR)(hResolution) #Downsampling dell'immagine ad alta risoluzione 
        with torch.no_grad():
            lResolution += utilities.gaussian_noise(lResolution, noise_level) #Aggiunta di rumore gaussiano all'immagine a bassa risoluzione ⚠️⚠️⚠️
        lResolution = transforms.Resize(data_shape_HR)(lResolution) #Upsampling dell'immagine a bassa risoluzione alla stessa dimensione dell'immagine ad alta risoluzione

        # Esegui il modello sull'immagine lResolution e confronta il risultato con hResolution
        self.model.eval()
        with torch.no_grad():
            lResolution = lResolution.unsqueeze(0).to(self.device)
            hResolution = hResolution.unsqueeze(0).to(self.device)
            prediction = self.model(lResolution)

        psnr = PSNR(prediction, hResolution)
        ssim = SSIM(prediction, hResolution)

        self.modelEvaluationOnImages.append({f'{path}_{data_shape_LR}_to_{data_shape_HR}_PSNR': psnr})
        self.modelEvaluationOnImages.append({f'{path}_{data_shape_LR}_to_{data_shape_HR}_SSIM': ssim})

        # Save prediction and target as images
        os.makedirs(self.resultRoot, exist_ok=True)
        pred_img = transforms.ToPILImage()(prediction.squeeze().cpu())
        target_img = transforms.ToPILImage()(hResolution.squeeze().cpu())
        pred_path = os.path.join(self.resultRoot, 'prediction.png')
        target_path = os.path.join(self.resultRoot, 'target.png')
        pred_img.save(pred_path)
        target_img.save(target_path)

        self.logTrain()
        return f'PSNR: {psnr} | SSIM: {ssim}'

    def logTrain(self):

        os.makedirs(self.resultRoot, exist_ok=True)
        # Prepare serializable representations
        loss_fn_name = self.loss_fn.__name__ if self.loss_fn is not None else None
        scheduler_name = type(self.scheduler).__name__ if self.scheduler is not None else None

        loss_params = {
            'FL': getattr(self.loss_fn, 'FL', None),
            'SSIM': getattr(self.loss_fn, 'SSIM', None),
            'MSE': getattr(self.loss_fn, 'MSE', None)
        }

        meta = {
            'num_epochs': self.num_epochs,
            'loss_fn': loss_fn_name,
            'scheduler': scheduler_name,
            'best_model': bool(self.best_model),
            'betchTrainingValidationPrint': int(self.betchTrainingValidationPrint),
            'loss_params': loss_params,
            'modelEvaluationOnTest': self.modelEvaluationOnTest,
            'modelEvaluationOnImages': self.modelEvaluationOnImages
        }

        json_path = os.path.join(self.resultRoot, f'{self.modelName}.json')
        import json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
        return json_path
        
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
        self.kernelSize = 7
        mu_x = torch.nn.functional.avg_pool2d(x, kernel_size=self.kernelSize, stride=1, padding=1)
        mu_y = torch.nn.functional.avg_pool2d(y, kernel_size=self.kernelSize, stride=1, padding=1)
        sigma_x = torch.nn.functional.avg_pool2d(x * x, kernel_size=self.kernelSize, stride=1, padding=1) - mu_x ** 2
        sigma_y = torch.nn.functional.avg_pool2d(y * y, kernel_size=self.kernelSize, stride=1, padding=1) - mu_y ** 2
        sigma_xy = torch.nn.functional.avg_pool2d(x * y, kernel_size=self.kernelSize, stride=1, padding=1) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)) / ((mu_x ** 2 + mu_y ** 2 + self.c1) * (sigma_x + sigma_y + self.c2) + 1e-8)
        return 1.0 - ssim_map.mean()
    
class Losses():
    def __init__(self,FL = 0.5, SSIM = 0.8, MSE = 0.2):
        self.FL = FL
        self.SSIM = SSIM
        self.MSE = MSE

    def FL_SSIM(self,):
        fn = lambda x, y: self.FL * FourierLoss()(x, y) + self.SSIM * SSIMLoss()(x, y)
        fn.__name__ = "FL_SSIM"
        fn.FL = self.FL
        fn.SSIM = self.SSIM
        fn.MSE = None
        return fn

    def MSE_SSIM_FL(self):
        fn = lambda x, y: self.MSE * torch.nn.MSELoss()(x, y) + self.FL * FourierLoss()(x, y) + self.SSIM * SSIMLoss()(x, y)
        fn.__name__ = "MSE_SSIM_FL"
        fn.FL = self.FL
        fn.SSIM = self.SSIM
        fn.MSE = self.MSE
        return fn