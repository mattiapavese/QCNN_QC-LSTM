import torch 
from torch.nn import Conv2d, BatchNorm2d, Module, ReLU, Tanh, Sigmoid


class HiddenBlock(Module):
    
    def __init__(self, channels_in, hidden_dim, kernel_size):
        
        super(HiddenBlock, self).__init__()
        
        self.conv1=Conv2d(channels_in+hidden_dim, hidden_dim,  kernel_size, padding='same')
        self.batchnorm1=BatchNorm2d(hidden_dim)
        self.conv2=Conv2d(hidden_dim, hidden_dim,  kernel_size, padding='same')
        self.batchnorm2=BatchNorm2d(hidden_dim)
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = ReLU()(self.batchnorm1(x))
        x = self.conv2(x)
        x = ReLU()(self.batchnorm2(x))
        
        return x
    
class OutputBlock(Module):
    pass


        
class ConvLSTM(Module):
    
    def __init__(self, channels_in, hidden_dim, kernel_size, channels_out=None, pass_states = False, return_sequence = False):
        
        super(ConvLSTM, self).__init__()
        
        if return_sequence:
            if channels_out is None: raise TypeError("Please specify a number of channels for output sequence")
            
            #here we could use "OutputBlock"
            self.out = Conv2d(hidden_dim, channels_out, kernel_size, padding='same')
        
        self.hidden_dim = hidden_dim
        
        #here we could use "HiddenBlock"
        self.forget = Conv2d(channels_in+hidden_dim, hidden_dim,  kernel_size, padding='same')
        self.input = Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same')
        self.candidate = Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same')
        self.output = Conv2d(channels_in+hidden_dim, hidden_dim, kernel_size, padding='same')
        
        self.return_sequence = return_sequence
        self.pass_states = pass_states
        
        
    def forward(self, x):
        
        
        if  not self.pass_states :
            hidden_state = torch.zeros((x.shape[0],)+(self.hidden_dim,)+x.shape[-2:])
            cell_state = torch.zeros((x.shape[0],)+(self.hidden_dim,)+x.shape[-2:])
        else: 
            x, hidden_state, cell_state = x

        outputs=[]
        for i in range(x.shape[1]):
            
            x_temp = x[:, i, :, :, :]
            
            x_conc = torch.cat([x_temp, hidden_state], dim=1)
            
            forg = Sigmoid()(self.forget(x_conc))
            inp= Sigmoid()(self.input(x_conc))
            cand = Tanh()(self.candidate(x_conc))
            out = Sigmoid()(self.output(x_conc))
            
            cell_state *= forg
            cell_state += (inp*cand)
            
            hidden_state = (Tanh()(cell_state))*out
            if self.return_sequence:
                #here we can change activation function for output seq!! (if we use "OutputBlock" we add ReLU there)
                outputs.append(ReLU()(self.out(hidden_state))) 
        if self.return_sequence: 
            return  hidden_state, cell_state, torch.stack(outputs, dim=1)
        else:
            return hidden_state, cell_state 


class EncDecConvLSTM(Module):
    
    def __init__(self, n_features, hidden_dim, n_outputs, kernel_size ):
        
        super(EncDecConvLSTM, self).__init__()
        self.encoder = ConvLSTM(n_features, hidden_dim, kernel_size, pass_states = False, return_sequence = False)
        self.decoder = ConvLSTM(n_outputs, hidden_dim, kernel_size, n_outputs, pass_states = True, return_sequence = True)
        
        
    def forward(self, x):
        enc_in, dec_in = x
        h_enc,c_enc = self.encoder(enc_in)
        _,_, output_seq = self.decoder([dec_in,h_enc,c_enc])

        return output_seq
    
    def forecast(self, x, n_steps, dec_in_min, dec_in_max):
        
        enc_in, dec_in = x #here dec_in is a 1-timestep token of 0s
        assert dec_in.shape[1]==1, "in forecast, decoder input must be a one-timestep token"

        h_to_dec,c_to_dec = self.encoder(enc_in)
        forecasted_seq=[]
        
        for i in range(n_steps):
            
            h_to_dec, c_to_dec, dec_out = self.decoder([dec_in,h_to_dec,c_to_dec])
            forecasted_seq.append(dec_out)
            dec_in = (dec_out-dec_in_min) / (dec_in_max-dec_in_min)
        
        return torch.cat(forecasted_seq, dim=1)
               



            
            
            
            
            
            
            
        