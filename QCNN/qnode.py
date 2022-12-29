import pennylane as qml

def spec_qnode(device, wires):
    
    @qml.qnode(device)
    def qnode(inputs, weights):
        
        qml.AmplitudeEmbedding( inputs, wires = range(wires), normalize=True, pad_with=0 )
        
        for i in range(wires):
            qml.Rot(*weights[ i*3 : (i*3) + 3 ], wires = i) 
        
        for i in range(wires):
            qml.CNOT(wires=[i, (i+1)%wires])
        
        return qml.probs( wires = list(range(wires)) ) #[qml.expval(qml.PauliZ(i)) for i in range(self.wires)]
        
    weight_shapes={"weights": 3*wires} #number of weights must be specified a priori based on the circuit structure
    
    return qnode, weight_shapes

