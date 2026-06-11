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
    def __init__(self, model, train_loader, optimizer, num_epochs, scheduler, loss_fn=torch.nn.MSELoss(), saveModel=False, validation_set=None, test_set=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        model.to(self.device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.saveModel = saveModel
        self.testSet = test_set
        self.validation_set = validation_set

    def evaluate(self, samples, grad = False):
        x, y = samples # (x, y): x = grandTruth, y = LowResolution
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
        print(f'⚠️: Testing=> {evalMetrich.__name__ if evalMetrich else "Loss"}')
                        
        evaluation = 0.0

        loss , y_vis, x_pred_vis = self.evaluate(self.testSet, grad=False)
        # Loss, Lr, ModelPredict 
        # Evaluate the metric for the current sample

        #print(f' ⚠️ Loss: {loss.item():.4f}, y: {y_vis.shape}, x_pred: {x_pred_vis.shape}')
        # Loss: 0.0702, y: torch.Size([32, 1, 256, 256]), x_pred: torch.Size([32, 1, 256, 256])

        if evalMetrich is not None:
            #For each sample in the test set, compute the evaluation metric and average it over the entire test set.
            for i in range(len(self.testSet[0])):

                x = self.testSet[0][i].cpu().numpy()
                x_pred_sample = x_pred_vis[i].cpu().numpy()

                if evalMetrich.__name__ == 'structural_similarity':
                    x = x.squeeze() 
                    x_pred_sample = x_pred_sample.squeeze() 

                evaluation += evalMetrich(x, x_pred_sample, data_range=1.0)
        
            average_evaluation = evaluation / len(self.testSet[0])
            print(f'⚠️: Average {evalMetrich.__name__} on Test Set: {average_evaluation:.4f}')
        
        else:
            evaluation = loss.item()
            print(f'⚠️: Loss on Test Set: {evaluation:.4f}')


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
            for index, samples in enumerate(self.train_loader):
                # sample = (hREsolution, lResolution)
                print(f'⚠️: Batch {index}/{len(self.train_loader)}', end='\r')

                loss, _, _ = self.evaluate(samples, grad=True) 
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

            if self.validation_set is not None:
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