from torch import nn
import torch
from torch.nn import functional as F
from MyUtilities import MayoDataset, Losses, MyTrainer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

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
    
class AttentionGate(nn.Module):
    def __init__(self, x_channels, skip_channels, inter_channels):
        super().__init__()
        self.x_ch_filters = nn.Conv2d(x_channels // 4, inter_channels, kernel_size=1) #Adatta il numero di canali dell'immagine in ingresso dopo PixelShuffle a un numero intermedio di canali.
        self.skip_ch_filters = nn.Conv2d(skip_channels, inter_channels, kernel_size=1) #Adatta il numero di canali dell'immagine skip (skip_channels) a un numero intermedio di canali (inter_channels) utilizzando una convoluzione 1x1.
        self.up = nn.PixelShuffle(upscale_factor=2) #Aumenta la dimensione spaziale dell'immagine di 2 usando PixelShuffle.
        self.combine_ch = nn.Conv2d(inter_channels, 1, kernel_size=1) #Combina le informazioni provenienti dai canali intermedi (inter_channels) in un singolo canale di output utilizzando una convoluzione 1x1. Questo passaggio è spesso utilizzato per generare una mappa di attenzione che evidenzia le aree importanti dell'immagine.
        self.resize_ch = nn.Conv2d(1, skip_channels, kernel_size=1) #Riporta il numero di canali dell'immagine di attenzione (1) al numero originale di canali dell'immagine skip (skip_channels) utilizzando una convoluzione 1x1. Questo passaggio è necessario per applicare la mappa di attenzione all'immagine skip.

        self.double_conv = DoubleConv(skip_channels, skip_channels)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, skipp):

        original_skipp = skipp

        x = self.up(x)
        if x.shape[-2:] != skipp.shape[-2:]:
            x = F.interpolate(x, size=skipp.shape[-2:], mode='bilinear', align_corners=False)
        
        x = self.x_ch_filters(x)
        skipp_filtered = self.skip_ch_filters(skipp)

        combineFeatchures = self.relu(x + skipp_filtered)
        combineChannel = self.sigmoid(self.combine_ch(combineFeatchures))
        AttentionMask = self.resize_ch(combineChannel)

        return self.double_conv(original_skipp * AttentionMask)
 
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        # --- ENCODER (Contraction Path) ---
        self.input = DoubleConv(in_channels, features[0])               # 1 -> 64
        self.down1 = downsample(features[0], features[1])               # 64 -> 128
        self.down2 = downsample(features[1], features[2])               # 128 -> 256

        # --- BOTTLENECK ---
        self.bottleneck = downsample(features[2], features[3])          # 256 -> 512
        
        # --- DECODER (Expansion Path) using Attention Gates ---
        # Replace previous upsample blocks with AttentionGate + DoubleConv fusion
        # attention: x_channels (from previous layer), skip_channels (from encoder), inter_channels (reduced)
        self.att1 = AttentionGate(x_channels=features[3], skip_channels=features[2], inter_channels=max(1, features[2] // 2))
        self.att2 = AttentionGate(x_channels=features[2], skip_channels=features[1], inter_channels=max(1, features[1] // 2))
        self.att3 = AttentionGate(x_channels=features[1], skip_channels=features[0], inter_channels=max(1, features[0] // 2))
        
        # --- OUTPUT ---
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) 

    def forward(self, x):
        # --- Encoder ---
        x_input = self.input(x) 
        down1 = self.down1(x_input)
        down2 = self.down2(down1)
        
        # --- Bottleneck ---
        bottleneck = self.bottleneck(down2)
        
        # --- Decoder con Skip Connections e Attention Gate ---
        # 1. attention on (bottleneck, down2)
        up1 = self.att1(bottleneck, down2)

        # 2. attention on (up1, down1)
        up2 = self.att2(up1, down1)

        # 3. attention on (up2, x_input)
        up3 = self.att3(up2, x_input)

        # --- Output Finale ---
        return x - self.final_conv(up3) # X = Y + n => X - Y = n


# =====================================================================
# 🔧 CONFIGURAZIONE
# =====================================================================

# --- Variante 1: rumore 0.005, scale 2 (già allenato) ---
# NOISE_LEVEL = 0.005
# SCALE = 2
# MODEL_NAME = "UNet_MSE_SSIM_FL_Residual_Attention_PixelShuffle_noise005_scale2"
# LOAD_WEIGHTS = './ResultsEtE/UNet_MSE_SSIM_FL_Residual_Attention_PixelShuffle_noise005_scale2.pth'
# DO_TRAIN = False   # è già allenato, vuoi solo testarlo

# --- Variante 2: rumore 0.01, scale 2 (già allenato) ---
#NOISE_LEVEL = 0.01
#SCALE = 2
#MODEL_NAME = "UNet_MSE_SSIM_FL_Residual_Attention_PixelShuffle_noise01_scale2"
#LOAD_WEIGHTS = './ResultsEtE/UNet_MSE_SSIM_FL_Residual_Attention_PixelShuffle_noise01_scale2.pth'
#DO_TRAIN = False

# --- Variante 3: rumore 0.005, scale 4 (già allenato) ---
#NOISE_LEVEL = 0.005
#SCALE = 4
#MODEL_NAME = "UNet_MSE_SSIM_FL_Residual_Attention_PixelShuffle_noise005_scale4"
#LOAD_WEIGHTS = None
#DO_TRAIN = True

# --- Variante 4: rumore 0.01, scale 4 (già allenato) ---
NOISE_LEVEL = 0.01
SCALE = 4
MODEL_NAME = "UNet_MSE_SSIM_FL_Residual_Attention_PixelShuffle_noise01_scale4"
LOAD_WEIGHTS = None
DO_TRAIN = True

# =====================================================================


dataSetTrain = MayoDataset(data_path='./Mayo/train', data_shape_HR=(256, 256), data_shape_LR=(128, 128), noise_level=NOISE_LEVEL, downscale_factor=SCALE)
trainSet, validationSet = train_test_split(dataSetTrain, test_size=0.15, random_state=42)
testSet = MayoDataset(data_path='./Mayo/test', data_shape_HR=(256, 256), data_shape_LR=(128, 128), noise_level=NOISE_LEVEL, downscale_factor=SCALE)
betchSize = 8 #PC Mattia si può mettere a 32, PC Pietro và messo a 8


print(f'⚠️ Training set size: {len(trainSet)} with betch: {betchSize}')
print(f'⚠️ Test set size: {len(testSet)} with betch: {betchSize}')
print(f'⚠️ Validation set size: {len(validationSet)} with betch: {betchSize}')

trainLoader = DataLoader(trainSet, batch_size=betchSize, shuffle=True)
validationLoader = DataLoader(validationSet, batch_size=betchSize, shuffle=True)
testLoader = DataLoader(testSet, batch_size=betchSize, shuffle=True)

modelResidualUNet = UNet(in_channels=1, out_channels=1) 
if LOAD_WEIGHTS is not None:
    modelResidualUNet.load_state_dict(torch.load(LOAD_WEIGHTS))

optimizerUNet = torch.optim.Adam(modelResidualUNet.parameters(), lr=5e-4)

trainerUNet = MyTrainer(
    model=modelResidualUNet, 
    train_loader=trainLoader, 
    test_loader=testLoader,
    validation_loader=validationLoader,
    optimizer=optimizerUNet,
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizerUNet, T_max=4), 
    loss_fn= Losses(MSE=0.8, FL=1, SSIM=1).MSE_SSIM_FL(),
    num_epochs=8,
    modelName=MODEL_NAME,
    best_model=True,
    betchTrainingValidationPrint=25,
    resultRoot="./ResultsEtE"
)

if DO_TRAIN:
    trainerUNet.train()

trainerUNet.test()
print('Test on image: ', trainerUNet.testOnImage(path='./267.png', data_shape_HR=(256, 256), data_shape_LR=(128, 128), noise_level=NOISE_LEVEL))


