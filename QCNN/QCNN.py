import pennylane as qml
import torch
from torch.nn.functional import unfold, pad
from qnode import spec_qnode

class QConv2d(torch.nn.Module):
    
    def __init__(self, in_channels, kernel_size, wires=6, stride=1, padding='same'):
        
        super(QConv2d, self).__init__()
        
        assert wires%2==0, "please build QConv2d vqc with an even number of wires"
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding 
        self.wires = wires 
        
        self.device = qml.device("qulacs.simulator", wires=self.wires)
        
        self.define_spec() #define 
        self.define_circuit()
        
        self.qlayer = qml.qnn.TorchLayer( self.qnode, self.weight_shapes )
        
        
    def define_spec(self):
        
        if isinstance(self.kernel_size, tuple) or isinstance(self.kernel_size, list):
            self.k_height, self.k_width = self.kernel_size
            kernel_entries = self.k_height * self.k_width * self.in_channels
        elif isinstance(self.kernel_size, int):
            self.k_height = self.k_width = self.kernel_size
            kernel_entries = self.k_height * self.k_width * self.in_channels
            
        else:
            raise ValueError('kernel_size must be either an int or a tuple containing kernel dimensions (kernel_height, kernel_width)')
        
        if isinstance(self.stride, tuple) or isinstance(self.stride, list):
            self.s_height, self.s_width = self.stride
        elif isinstance(self.stride, int):
            self.s_height = self.s_width = self.stride
        else:
            raise ValueError('stride must be either an int or a tuple containing stride values over two axis (stride_height, stride_width)')
        
        if self.padding == 'same': 
            
            p_height = self.k_height - 1
            p_width = self.k_width - 1
            if p_height % 2 == 0: 
                self.p_top = self.p_bottom = int(p_height/2)
            else:
                self.p_top, self.p_bottom = int(p_height/2) + 1, int(p_height/2)   
            if p_width % 2 == 0: 
                self.p_left = self.p_right = int(p_width/2)
            else:
                self.p_left, self.p_right = int(p_width/2) + 1, int(p_width/2)
            
            #@NotImplemented
            #to compute padding that always maintain same dimension: (need height and width of input!)
            #p_height = self.k_heigth + self.height *(self.s_heigth - 1) - self.s_heigth
            #p_width = self.k_width + self.width *(self.s_width - 1) - self.s_width
        
        elif isinstance(self.padding, tuple) or isinstance(self.padding, list):
            self.p_left, self.p_right, self.p_top, self.p_bottom  = self.padding
        
        elif self.padding == 'valid':
            pass
            
        else:
            raise ValueError("padding must be either 'same' or 'valid' or a four element tuple indicating (left pad, right pad, top pad, bottom pad)")
        
        
    def define_circuit(self): #FIXME: time to implement a better circuit for QCNN
        
        self.qnode, self.weight_shapes = spec_qnode(self.device, self.wires)
   
    def forward(self, x):
    
        height, width = x.shape[-2:]
        
        if self.padding != 'valid':
            x = pad( x, (self.p_left, self.p_right, self.p_top, self.p_bottom ) )
        
        out_shape = ( int( ( height + self.p_top + self.p_bottom - self.k_height ) / self.s_height ) + 1 , 
                      int( ( width + self.p_left + self.p_right - self.k_width ) / self.s_width ) + 1 )
        
        x = torch.transpose( unfold( x, kernel_size=self.kernel_size, stride=self.stride) , -1, -2 ) 
        x = torch.transpose(self.qlayer(x) , -1, -2 )

        return torch.reshape(x, x.shape[:2] + out_shape)







