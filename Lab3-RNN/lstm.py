import torch
import torch.nn as nn


class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_gate = nn.Linear(in_features=input_dim+hidden_dim, out_features=hidden_dim)
        self.input_gate = nn.Linear(in_features=input_dim+hidden_dim, out_features=hidden_dim)
        self.cell_update = nn.Linear(in_features=input_dim+hidden_dim, out_features=hidden_dim)
        self.output_gate = nn.Linear(in_features=input_dim+hidden_dim, out_features=hidden_dim)
        
    def forward(self, x, hidden, cell):
        state = torch.concat((x, hidden), dim=-1)
        f = torch.sigmoid(self.forget_gate(state))
        i = torch.sigmoid(self.input_gate(state))
        c = torch.tanh(self.cell_update(state))
        cell = f * cell + i * c
        output = torch.sigmoid(self.output_gate(state))
        hidden = output * torch.tanh(cell)
        return output, hidden, cell
    
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, block_num, output_class):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.block_num = block_num
        self.blocks = nn.ModuleList([LSTMBlock(input_dim=input_dim, hidden_dim=hidden_dim) for i in range(block_num)])
        self.classify = nn.Linear(in_features=hidden_dim, out_features=output_class)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.device = device
    
    def forward(self, input):
        input_length = input.size()[0]
        hiddens = [torch.zeros(1, self.hidden_dim).to(input.device) for i in range(self.block_num)]
        cells = [torch.zeros(1, self.hidden_dim).to(input.device) for i in range(self.block_num)]
        output = None
        for i in range(input_length):
            x = input[i]
            for b in range(self.block_num):
                hidden = hiddens[b]
                cell = cells[b]
                x, hidden, cell = self.blocks[b](x, hidden, cell)
                hiddens[b] = hidden
                cells[b] = cell
            output = x
        output = self.softmax(self.classify(output))
        return output      
        
        
if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm = LSTM(52, 128, 1, 10)
    input = torch.rand(5,1,52)
    output = lstm(input)
    print(output.shape)