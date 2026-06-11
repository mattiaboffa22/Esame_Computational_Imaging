import glob
from sched import scheduler
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from IPPy import operators 
from IPPy import utilities 
import torch
import matplotlib.pyplot as plt
import numpy as np

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
    def __init__(self, model, train_loader, optimizer, num_epochs, scheduler, loss_fn=torch.nn.MSELoss(), saveModel=False, validation_set=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        model.to(self.device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.saveModel = saveModel
        self.testSet = next(iter(self.train_loader))
        self.validation_set = validation_set

    def evaluate(self, sample, grad = False): #⚠️⚠️⚠️⚠️ TO DO!!
        # grad=False indica che non vogliamo calcolare i gradienti durante la valutazione. Questo è particolarmente utile durante la fase di test o validazione, dove non è necessario aggiornare i pesi del modello.
        x, y = sample # (x, y): x = grandTruth, y = LowResolution
        x = x.to(self.device)
        y = y.to(self.device)
        
        if grad:
            x_pred = self.model(y) 
        else:
            with torch.no_grad():
                x_pred = self.model(y)

        loss = self.loss_fn(x_pred, x) 
        return loss, y, x_pred

    def test(self, evalMetrich = None): #⚠️⚠️⚠️⚠️ TO DO!!
        print('⚠️: Testing...', end='\r')
        x = self.testSet

        loss, y, x_pred = self.evaluate(x)

        print()

        # Per visualizzare le immagini, prendiamo solo il primo esempio del batch e lo portiamo alla CPU
        x_vis = x[0][0].cpu()
        y_vis = y[0].cpu()
        x_pred_vis = x_pred[0].cpu()

        if evalMetrich is not None:
            print(f'⚠️: Evaluating {evalMetrich.__name__}...', end='\r')
            # La metrica LPIPS utilizza tensori, mentre le altre metriche come PSNR e SSIM utilizzano array numpy. Quindi, se la metrica è LPIPS, manteniamo i tensori, altrimenti li convertiamo in numpy array.
            if not evalMetrich.__name__ == 'lpips':

                x_pred = x_pred.cpu().numpy()
                x = x.cpu().numpy()

            if evalMetrich.__name__ == 'structural_similarity':
                # Squeeze to (H, W) for grayscale images
                x_metric = x.squeeze()
                x_pred_metric = x_pred.squeeze()
                metric_value = evalMetrich(x_metric, x_pred_metric, data_range=1.0)
            else:
                metric_value = evalMetrich(x, x_pred, data_range=1.0) #Assumendo che i valori dei pixel siano normalizzati tra 0 e 1, il data_range è impostato a 1.0. Se i valori dei pixel fossero in un intervallo diverso, ad esempio [0, 255], allora data_range dovrebbe essere impostato a 255. channel_axis=1 indica che la dimensione dei canali è la seconda dimensione del tensore (N, C, H, W).
            print(f'👉 Test Loss: {loss.item()}, {evalMetrich.__name__}: {metric_value}', end='\r')
        else:
            print(f'👉 Test Loss: {loss.item()}', end='\r')
        print()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title('Ground Truth')
        plt.imshow(x_vis.squeeze(), cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title('Noisy Blurred')
        plt.imshow(y_vis.squeeze(), cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title('Reconstructed')
        plt.imshow(x_pred_vis.squeeze(), cmap='gray')
        plt.show()

    def train(self):
        loss_history = []
        validation_Loss_history = []

        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training loss, Validation Loss')
        line, = ax.plot([], [], 'b-')
        val_line, = ax.plot([], [], 'r-')

        for epoch in range(self.num_epochs):
            for index, sample in enumerate(self.train_loader):
                # sample = (hREsolution, lResolution)
                print(f'⚠️: Batch {index}/{len(self.train_loader)}', end='\r')

                loss, _, _ = self.evaluate(sample, grad=True) 
                loss.backward()

                self.optimizer.step() 
                self.optimizer.zero_grad() 

                self.scheduler.step()

                loss_history.append(loss.item())

                line.set_xdata(range(len(loss_history)))
                line.set_ydata(loss_history)
                line.set_label('Training Loss')
                ax.legend()
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

            if self.validation_set is not None: #⚠️⚠️⚠️⚠️ TO DO!!
                validation_Loss, _, _ = self.evaluate(self.validation_set, grad=False)
                validation_Loss_history.append(validation_Loss.item())
                val_line.set_ydata(validation_Loss_history)
                val_line.set_xdata(range(len(validation_Loss_history)))
                val_line.set_label('Validation Loss')
                ax.legend()

            print()
            print(f'👉 Epoch {epoch}, Loss: {loss.item()}, LR: {self.scheduler.get_last_lr()}', end='\r')
            print()

        plt.ioff()
        plt.show()

        if self.saveModel:
            torch.save(self.model.state_dict(), 'model.pth')
            #'model.pth' è il nome del file in cui viene salvato lo stato del modello.
            #self.model.state_dict() è un dizionario che contiene tutti i parametri del modello (pesi e bias) e le loro rispettive chiavi. Questo è il formato standard per salvare i modelli in PyTorch, poiché consente di caricare facilmente i parametri in un modello con la stessa architettura in futuro.