from torch import nn
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# https://github.com/JoungheeKim/autoencoder-lstm/blob/main/autoencoderLSTM_tutorial(english).ipynb

class Encoder(nn.Module):
    def __init__(self,inputSize=3,hiddenSize=2,layers=2): #Revisit this 
        super(Encoder, self).__init__()
        self.hidden = hiddenSize
        self.layers = layers
        self.lstm = nn.LSTM(inputSize, hiddenSize, layers, batch_first=True, dropout=0.1, bidirectional=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return (hidden, cell)
    

class Decoder(nn.Module):
    def __init__(self,inputSize=3,hiddenSize=2,layers=2,outputSize=3):
        super(Decoder, self).__init__()
        self.hidden = hiddenSize
        self.layers = layers
        self.lstm = nn.LSTM(inputSize, hiddenSize, layers, batch_first=True, dropout=0.1, bidirectional=False)
        self.fc = nn.Linear(hiddenSize, outputSize)
        self.relu = nn.ReLU()

    def forward(self, x, hidden):
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)
    

class Autoencoder(nn.Module):
    def __init__(self,inputSize,hiddenSize,layers):
        super(Autoencoder,self).__init__()
        self.encoder = Encoder(inputSize, hiddenSize, layers)
        self.decoder = Decoder(inputSize, hiddenSize, layers, inputSize) # inputSize = outputSize - as you want to reconstruct the input
        self.criterion = nn.MSELoss()
        self.layers = layers

    def forward(self, x, hidden=None):
        if hidden is None:
            _, (hidden, cell) = self.encoder(x)
            hidden = hidden.repeat(self.layers, 1, 1)
            cell = cell.repeat(self.layers, 1, 1)
        else:
            hidden, cell = hidden

        decode, _ = self.decoder(x, (hidden, cell))
        return decode

def train(learningRate, epochs,runTillConvergence,device, layers, hiddenSize, model, trainLoader, testLoader):
    optim = torch.optim.Adam(model.parameters(), lr=learningRate)

    if runTillConvergence:
        prevLoss = float('inf')
        i = 0
        while True:
            model.train()
            for data in trainLoader:
                inputData = data.to(device)
                batchSize = inputData.size(1)

                # Initialize hidden and cell states
                hidden = torch.zeros((layers, batchSize, hiddenSize)).to(device)
                cell = torch.zeros((layers, batchSize, hiddenSize)).to(device)

                # Forward pass
                output = model(inputData.permute(1, 0, 2), (hidden, cell))
                output = output.permute(1, 0, 2)
                loss = model.criterion(output, inputData)

                optim.zero_grad()
                loss.backward()
                optim.step()
                evalLoss = 0
                with torch.no_grad():
                    for data in testLoader:
                        inputData = data.to(device)
                        batchSize = inputData.size(1)

                        # Initialize hidden and cell states
                        hidden = torch.zeros((layers, batchSize, hiddenSize)).to(device)
                        cell = torch.zeros((layers, batchSize, hiddenSize)).to(device)

                        # Forward pass
                        output = model(inputData.permute(1, 0, 2), (hidden, cell))
                        output = output.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, input_size)
                        loss = model.criterion(output, inputData)
                        evalLoss += loss.item()

                evalLoss /= len(testLoader)
                print(f"Epoch {i+1} - Eval Loss: {evalLoss}")
                if evalLoss > prevLoss:
                    break
                prevLoss = evalLoss
                i += 1
                

    else:
        for epoch in range(epochs):
            model.train()
            for data in trainLoader:
                inputData = data.to(device)
                batchSize = inputData.size(1)

                # Initialize hidden and cell states
                hidden = torch.zeros((layers, batchSize, hiddenSize)).to(device)
                cell = torch.zeros((layers, batchSize, hiddenSize)).to(device)

                # Forward pass
                output = model(inputData.permute(1, 0, 2), (hidden, cell))
                output = output.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, input_size)
                loss = model.criterion(output, inputData)

                optim.zero_grad()
                loss.backward()
                optim.step()

            model.eval()
            evalLoss = 0
            with torch.no_grad():
                for data in testLoader:
                    inputData = data.to(device)
                    batchSize = inputData.size(1)

                    # Initialize hidden and cell states
                    hidden = torch.zeros((layers, batchSize, hiddenSize)).to(device)
                    cell = torch.zeros((layers, batchSize, hiddenSize)).to(device)

                    # Forward pass
                    output = model(inputData.permute(1, 0, 2), (hidden, cell))
                    output = output.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, input_size)
                    loss = model.criterion(output, inputData)
                    evalLoss += loss.item()

            evalLoss /= len(testLoader)
            print(f"Epoch {epoch+1} - Eval Loss: {evalLoss}")
    return model

    


#Clean
clean = pd.read_csv('KDDTrainAE.csv')

# Assuming your data is in a single column and each row represents a timestep - change this to match your data
#We only care about the columns that arent the first and last ones as they are the class and index
#get the df without the first and last columns
timeseriesClean= torch.tensor(clean.iloc[:,1:-1].values, dtype=torch.float32)


#Anomalous data to get reconstruction error
anom = pd.read_csv('KDDTrainNumerical.csv')
# Assuming your data is in a single column and each row represents a timestep - change this to match your data
timeseriesAnom = torch.tensor(anom.iloc[:,1:-1].values, dtype=torch.float32)



# Step 2: Create a Custom Dataset Class
class TimeSeriesDatasetSlidingWindow(Dataset):
    def __init__(self, data,windowSize):
        self.data = data
        self.windowSize = windowSize
        self.windows = self.windowCreation()

    def windowCreation(self):
        windows = []
        for i in range(0,len(self.data) - self.windowSize + 1):
            windows.append(self.data[i:i + self.windowSize])
        return torch.stack(windows)
    
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]

# Step 3: Use DataLoader to Load the Data


windowSize = 10
#Given we want 1000 values for the output and have a sliding window of N, we take the last N-1 values and copy them to the front of the data
#This is to ensure that the first N-1 values have a full window of data
timeseriesAnom = torch.cat((timeseriesAnom[-(windowSize-1):],timeseriesAnom),0)
#We only care about this for the anomalous data


datasetC = TimeSeriesDatasetSlidingWindow(timeseriesClean,windowSize=windowSize)
datasetA = TimeSeriesDatasetSlidingWindow(timeseriesAnom,windowSize=windowSize)
batch_size = 64  # Set your desired batch size
shuffle = False  # Set to True to have the data reshuffled at every epoch
dataLoaderC = DataLoader(datasetC, batch_size=batch_size, shuffle=shuffle)
dataLoaderA = DataLoader(datasetA, batch_size=1, shuffle=shuffle)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
inputSize = 41
hiddenSize = 32
outputSize = 16
layers = 2
learningRate = 0.0005
#Max number of iterations is the number of episodes in the training loop
epochs = 200
runTillConvergence = False


model = Autoencoder(inputSize,hiddenSize,layers)
model.to(device=device)
model = train(learningRate,epochs,runTillConvergence,device,layers,hiddenSize,model,dataLoaderC,dataLoaderC)


# Now you can use the trained model to make predictions on dataLoaderA (anomalous data) - Loss is the reconstruction error

model.eval()
with open('output.csv', 'w') as f:
    f.write("timestamp")
    for i in range(len(anom.keys())-1):
        f.write(f",reconstructionError-{i}")
    f.write("\n")
    with torch.no_grad():
        for i, data in enumerate(dataLoaderA):
            inputData = data.to(device)
            batch_size = inputData.size(1)

            # Initialize hidden and cell states
            hidden = torch.zeros((layers, batch_size, hiddenSize)).to(device)
            cell = torch.zeros((layers, batch_size, hiddenSize)).to(device)

            # # Forward pass
            output = model(inputData.permute(1, 0, 2), (hidden, cell))
            output = output.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, input_size)

            losses = []
            f.write(f"{i}")
            for channel in range(inputSize):
                channelOut = output[:, :, channel]  # Extract the specific channel
                channelIn = inputData[:, :, channel]
                loss = model.criterion(channelOut, channelIn).item()
                f.write(f",{loss}")
            f.write("\n")
            # Write the losses to the file
         