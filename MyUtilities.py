import glob
import os
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
    def __init__(self, model, train_loader, optimizer, num_epochs, scheduler, loss_fn=torch.nn.MSELoss(), saveModel=False, modelName="model", validation_loader=None, test_loader=None, best_model=False, betchTrainingValidationPrint = 5, evalMetrich = []):
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
        self.evalMetrich = evalMetrich
        self.modelEvaluationOnTest = []
        self.modelEvaluationOnImages = []

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
        metrics_names = ', '.join([m.__name__ for m in self.evalMetrich]) if self.evalMetrich else "Loss function"
        print(f'🔄️ :Testing=> {metrics_names}')
        if not self.test_loader:
            print(f'🚨 No one test set.')
            return

        if self.evalMetrich:
            # Esegui tutte le metriche di valutazione
            metric_cumulative = {m.__name__: 0.0 for m in self.evalMetrich}
            samplesCounter = 0

            for testIndex, testSamples in enumerate(self.test_loader):
                print(f'🔁 : test batch: {testIndex}', end='\r')
                test_Loss, _, modelPredictions = self.evaluate(testSamples, grad=False)

                for i in range(len(modelPredictions)):
                    x = testSamples[0][i].cpu().numpy()
                    x_pred_sample = modelPredictions[i].cpu().numpy()

                    for metric_fn in self.evalMetrich:
                        x_metric = x.squeeze() if metric_fn.__name__ == 'structural_similarity' else x
                        x_pred_metric = x_pred_sample.squeeze() if metric_fn.__name__ == 'structural_similarity' else x_pred_sample
                        metric_cumulative[metric_fn.__name__] += metric_fn(x_metric, x_pred_metric, data_range=1.0)
                    
                    samplesCounter += 1

            # Calcola e registra le medie per tutte le metriche
            for metric_fn in self.evalMetrich:
                metric_name = metric_fn.__name__
                avg = metric_cumulative[metric_name] / samplesCounter
                self.modelEvaluationOnTest.append({metric_name: avg})
                print(f'⚠️: Average {metric_name} on Test Set: {avg:.4f}')
        else:
            # Usa la funzione di perdita come fallback
            CumulativeEvaluation = 0.0
            for testIndex, testSamples in enumerate(self.test_loader):
                print(f'🔁 : test batch: {testIndex}', end='\r')
                test_Loss, _, _ = self.evaluate(testSamples, grad=False)
                CumulativeEvaluation += test_Loss.item()

            avg = CumulativeEvaluation / len(self.test_loader)
            self.modelEvaluationOnTest.append({"Loss function": avg})
            print(f'⚠️: Average Loss function on Test Set: {avg:.4f}')

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
                            print(f'⏬: Saving best model: ./Results/{self.modelName}: {avarageValidationLoss}', end='\r')
                            print()
                            currentBestModelValidationLoss = avarageValidationLoss
                            torch.save(self.model.state_dict(), f'./Results/{self.modelName}.pth')

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

                    fig.savefig(f"./Results/{self.modelName}_Training_Loss_Plot.png")
                    print(f'✅: Saved loss plot: ./Results/{self.modelName}_Training_Loss_Plot.png', end='\r')
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
        #self.model.eval()
        with torch.no_grad():
            lResolution = lResolution.unsqueeze(0).to(self.device)
            hResolution = hResolution.unsqueeze(0).to(self.device)
            prediction = self.model(lResolution)

        prediction = prediction.squeeze(0)
        target = hResolution.squeeze(0)

        x_pred_np = prediction.cpu().numpy()
        target_np = target.cpu().numpy()

        if self.evalMetrich:
            # Esegui tutte le metriche di valutazione
            for metric_fn in self.evalMetrich:
                x_pred_metric = x_pred_np.squeeze() if metric_fn.__name__ == 'structural_similarity' else x_pred_np
                target_metric = target_np.squeeze() if metric_fn.__name__ == 'structural_similarity' else target_np
                metric_value = metric_fn(target_metric, x_pred_metric, data_range=1.0)
                metric_name = metric_fn.__name__
                self.modelEvaluationOnImages.append({f'{path}_{data_shape_LR}_to_{data_shape_HR}_{metric_name}': metric_value})
        else:
            # Usa la funzione di perdita come fallback
            metric_value = self.loss_fn(prediction.unsqueeze(0), hResolution).item()
            self.modelEvaluationOnImages.append({f'{path}_{data_shape_LR}_to_{data_shape_HR}_Training loss': metric_value})

        # Save prediction and target as images
        os.makedirs('Results', exist_ok=True)
        pred_img = transforms.ToPILImage()(prediction.cpu())
        target_img = transforms.ToPILImage()(target.cpu())
        pred_path = os.path.join('Results', 'prediction.png') #⚠️⚠️⚠️ Fix
        target_path = os.path.join('Results', 'target.png') #⚠️⚠️⚠️ Fix
        pred_img.save(pred_path)
        target_img.save(target_path)

        self.logTrain()
        return metric_value if self.evalMetrich else metric_value

    def logTrain(self):

        os.makedirs('Results', exist_ok=True)
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

        json_path = os.path.join('Results', f'{self.modelName}.json')
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