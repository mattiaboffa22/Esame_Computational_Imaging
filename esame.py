from torch import nn
import torch
from torch.nn import functional as F
from MyUtilities import MayoDataset, Losses, MyTrainer
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.model_selection import train_test_split

# ⚠️ Receptive Field: 44*44

#class DoubleConv(nn.Module):
#    def __init__(self, in_channels, out_channels):
#        super(DoubleConv, self).__init__()
#        self.inputLayer=nn.Conv2d(in_channels, out_channels, kernel_size=5, padding='same')
#        self.firstConv=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
#        self.residualConnection=nn.Conv2d(out_channels, in_channels, kernel_size=1, padding='same')
#        self.lastLayer=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
#        self.relu = nn.ReLU()
#
#    def forward(self, x):
#        il = self.relu(self.inputLayer(x))
#        fc = self.relu(self.firstConv(il))
#        rc = self.relu(self.residualConnection(fc))
#        return self.lastLayer(rc + x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential( #Esegue in sequenza le operazioni definite al suo interno
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
        )

    def forward(self, x):
        return self.double_conv(x)
    
class downsample(nn.Module):
    """ In questo scenario, prima eseguiamo il downsampling, quindi applichiamo la riduzione della 
    dimensione spaziale dell'immagine utilizzando un'operazione di pooling (in questo caso, MaxPool2d)
    e successivamente applichiamo il DoubleConv per estrarre le caratteristiche dall'immagine ridotta. """
    def __init__(self, in_channels, out_channels):
        super(downsample, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), #Riduce la dimensione spaziale dell'immagine di 2. Il kernel di pooling di dimensione 2x2 viene applicato all'immagine, e il valore massimo all'interno di ogni finestra di pooling viene selezionato per creare l'immagine ridotta.
            DoubleConv(in_channels, out_channels) #Applica il DoubleConv definito in precedenza. 
        )

    def forward(self, x):
        return self.down(x)
    
class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(upsample, self).__init__()
        self.up = nn.PixelShuffle(upscale_factor=2)
        self.conv = DoubleConv((in_channels // 4) + skip_channels, out_channels) 
        #PixcelShuffle ridistribuisce i canali per ingarandire l'immagine, se scala di 2 un pixcel diventa 4 pixcel, quindi 4 canali vengono utilizzati per scalare l'immagine

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False) #Se le dimensioni spaziali dell'immagine upsampled (x) non corrispondono a quelle dell'immagine skip, viene utilizzata la funzione F.interpolate per ridimensionare l'immagine upsampled alle dimensioni dell'immagine skip. L'interpolazione bilineare viene utilizzata per mantenere la qualità dell'immagine durante il ridimensionamento.
            #Anche scalando l'immagine con un fattore di 2, le dimensioni potrebbero non coincidere nel caso di input con dimensioni dispari, quindi è necessario eseguire un ulteriore ridimensionamento per garantire che le dimensioni siano compatibili per la concatenazione.
        x = torch.cat([x, skip], dim=1) #Concatena l'immagine upsampled con l'immagine skip (che proviene dalla fase di downsampling) lungo la dimensione dei canali (dim=1).
        """
        [batch, canali, altezza, larghezza]
            0       1       2        3
        """
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        # --- ENCODER (Contraction Path) ---
        self.input = DoubleConv(in_channels, features[0])               # 1 -> 64
        self.down1 = downsample(features[0], features[1])               # 64 -> 128
        self.down2 = downsample(features[1], features[2])               # 128 -> 256

        # --- BOTTLENECK ---
        self.bottleneck = downsample(features[2], features[3])          # 256 -> 512
        
        # --- DECODER (Expansion Path) ---
        # upsample prende: (in_channels_dalla_base, in_channels_dalla_skip_connection, out_channels)
        self.up1 = upsample(features[3], features[2], features[2])      # 512 + 256 -> 256
        self.up2 = upsample(features[2], features[1], features[1])      # 256 + 128 -> 128
        self.up3 = upsample(features[1], features[0], features[0])      # 128 + 64 -> 64
        
        # --- OUTPUT ---
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) 

    def forward(self, x):
        # --- Encoder ---
        x_input = self.input(x) 
        down1 = self.down1(x_input)
        down2 = self.down2(down1)
        
        # --- Bottleneck ---
        bottleneck = self.bottleneck(down2)
        
        # --- Decoder con Skip Connections ---
        # 1. up1 riceve il bottleneck e la skip connection 'down2'
        up1 = self.up1(bottleneck, down2) 
        
        # 2. up2 riceve up1 e la skip connection 'down1'
        up2 = self.up2(up1, down1)
        
        # 3. up3 riceve up2 e la skip connection iniziale 'x_input'
        up3 = self.up3(up2, x_input)

        # --- Output Finale ---
        return x - self.final_conv(up3) # X = Y + n => X - Y = n
    
dataSetTrain = MayoDataset(data_path='./Mayo/train', data_shape_HR=(256, 256), data_shape_LR=(128, 128), noise_level=0.005)
trainSet, validationSet = train_test_split(dataSetTrain, test_size=0.15, random_state=42)
testSet = MayoDataset(data_path='./Mayo/test', data_shape_HR=(256, 256), data_shape_LR=(128, 128), noise_level=0.005)
betchSize = 40 #Si più non ce la fa!!


print(f'⚠️ Training set size: {len(trainSet)} with betch: {betchSize}')
print(f'⚠️ Test set size: {len(testSet)} with betch: {betchSize}')
print(f'⚠️ Validation set size: {len(validationSet)} with betch: {betchSize}')

trainLoader = DataLoader(trainSet, batch_size=betchSize, shuffle=True)
validationLoader = DataLoader(validationSet, batch_size=betchSize, shuffle=True)
testLoader = DataLoader(testSet, batch_size=betchSize, shuffle=True)

modelResidualUNet = UNet(in_channels=1, out_channels=1) 
#modelUNet.load_state_dict(torch.load('./Results/UNet_MSE_SSIM_FL_0.1.pth'))

optimizerUNet = torch.optim.Adam(modelResidualUNet.parameters(), lr=5e-4)

trainerUNet = MyTrainer(
    model=modelResidualUNet, 
    train_loader=trainLoader, 
    test_loader=testLoader,
    validation_loader=validationLoader,
    optimizer=optimizerUNet,
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizerUNet, T_max=6), 
    loss_fn= Losses(MSE=0.8, FL=0.5, SSIM=1).MSE_SSIM_FL(),
    num_epochs=12,
    modelName="UNet_MSE_SSIM_FL_Residual",
    best_model=True,
    betchTrainingValidationPrint=25,
    evalMetrich=[structural_similarity, peak_signal_noise_ratio]
    # It compares local image patches in terms of luminance, contrast, and structure, and is therefore much more sensitive to structural distortions. SSIM usually takes values between 0 and 1, where values closer to 1 indicate higher similarity.
    # A larger PSNR corresponds to a smaller pixel-wise error. It remains a pixel-wise fidelity measure.
    )


trainerUNet.train()
trainerUNet.test()
print('Test on image: ', trainerUNet.testOnImage(path='./267.png', data_shape_HR=(256, 256), data_shape_LR=(128, 128), noise_level=0.005))


