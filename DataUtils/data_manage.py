import pandas as pd
import glob
import numpy as np
from itertools import islice
import os 
import csv


def data_handler(path, coordinates, d_x, d_y, 
                window_enc=24, horizon=1, window_dec=8, jump=5,
                keep_columns_inputs = ['temperature_2m (°C)','relativehumidity_2m (%)','dewpoint_2m (°C)'],
                keep_columns_labels=['temperature_2m (°C)']):
    
    info={}
    
    df_collection_enc_inputs, df_collection_dec_inputs, df_collection_labels = to_pandas_with_check_len_and_columns(path, coordinates,
                                                                             keep_columns_inputs, 
                                                                             keep_columns_labels)
    
    df_unrolled_enc_inputs, df_unrolled_dec_inputs, df_unrolled_labels = map(lambda x: pd.concat(x, axis=1), 
                                                 [df_collection_enc_inputs, df_collection_dec_inputs, df_collection_labels])
    
    # ************** TRAIN / TEST SPLIT ****************************
    
    split_value=int(0.8*len(df_unrolled_enc_inputs))
    
    df_unrolled_enc_inputs_train = df_unrolled_enc_inputs[:split_value] 
    df_unrolled_enc_inputs_test = df_unrolled_enc_inputs[split_value:]
    
    df_unrolled_dec_inputs_train = df_unrolled_dec_inputs[:split_value]
    df_unrolled_dec_inputs_test = df_unrolled_dec_inputs[split_value:]
    
    df_unrolled_labels_train = df_unrolled_labels[:split_value]
    df_unrolled_labels_test = df_unrolled_labels[split_value:]
    
    num_features_enc = len(keep_columns_inputs) #eventually add hour and month column -> increase number of features 
    num_features_dec = len(keep_columns_labels)
    
    info['n_inputs']=num_features_enc
    info['n_labels']=num_features_dec
    
    
    #****************** NORMALIZATION *********************
    #normalize train inputs and test inputs (based on train min/max) (RECOVER SOMEWAY MIN AND MAX!!!)
    #need to normalize both encoder inputs and decoder inputs
    #labels doo not have to be normalized
    
    enc_inputs_train_min, enc_inputs_train_max = df_unrolled_enc_inputs_train.min(),  df_unrolled_enc_inputs_train.max()
    dec_inputs_train_min, dec_inputs_train_max = df_unrolled_dec_inputs_train.min(),  df_unrolled_dec_inputs_train.max()
    
    df_unrolled_enc_inputs_train = (df_unrolled_enc_inputs_train - enc_inputs_train_min)/( enc_inputs_train_max - enc_inputs_train_min)
    df_unrolled_enc_inputs_test = (df_unrolled_enc_inputs_test - enc_inputs_train_min)/( enc_inputs_train_max - enc_inputs_train_min)
    
    df_unrolled_dec_inputs_train = (df_unrolled_dec_inputs_train - dec_inputs_train_min)/( dec_inputs_train_max - dec_inputs_train_min)
    df_unrolled_dec_inputs_test = (df_unrolled_dec_inputs_test - dec_inputs_train_min)/( dec_inputs_train_max - dec_inputs_train_min)
    
    
    #put min and max tensors of train set to normalize decoder iputs in inference 
    info['enc_in_min'] = np.reshape( enc_inputs_train_min.to_numpy() , (num_features_enc, d_x, d_y), order='F').transpose(0,2,1)
    info['enc_in_max'] = np.reshape( enc_inputs_train_max.to_numpy() , (num_features_enc, d_x, d_y), order='F').transpose(0,2,1)
    info['dec_in_min'] = np.reshape( dec_inputs_train_min.to_numpy() , (num_features_dec, d_x, d_y), order='F').transpose(0,2,1)
    info['dec_in_max'] = np.reshape( dec_inputs_train_max.to_numpy() , (num_features_dec, d_x, d_y), order='F').transpose(0,2,1)
                    
    
    #************** MAKE TENSORS ****************
    #build tensors (encoder_inputs, decoder_inputs, decoder_outputs)
    encoder_inputs_train, decoder_inputs_train, decoder_labels_train = make_tensors(df_unrolled_enc_inputs_train, 
                                                                                    df_unrolled_dec_inputs_train,
                                                                                    df_unrolled_labels_train,
                                                                                    window_enc, horizon, window_dec, jump)
    
    encoder_inputs_test, decoder_inputs_test , decoder_labels_test = make_tensors(df_unrolled_enc_inputs_test, 
                                                                df_unrolled_dec_inputs_test,
                                                                df_unrolled_labels_test,
                                                                window_enc, horizon, window_dec, jump)
    
    #*********** RESHAPE TENSORS TO OBTAIN SPACETIME DATA **********************
    
    encoder_inputs_train, encoder_inputs_test = map(lambda x: reshape_tensor_to_space_time_config(x,num_features_enc,d_x,d_y) , 
                                                         [encoder_inputs_train, encoder_inputs_test]) 
    
    decoder_inputs_train, decoder_inputs_test = map(lambda x: reshape_tensor_to_space_time_config(x,num_features_dec,d_x,d_y), 
                                                    [decoder_inputs_train, decoder_inputs_test])
    
    decoder_labels_train, decoder_labels_test = map(lambda x: reshape_tensor_to_space_time_config(x,num_features_dec,d_x,d_y) , 
                                                         [decoder_labels_train, decoder_labels_test]) 
    
    
    #put zeros at the beginning of the tensor decoder_inputs_train (FOR NOW WE DO NOT DO IT)
    #decoder_inputs_train[:,0,:,:,:] = np.zeros( shape = decoder_inputs_train[:,0,:,:,:].shape )
    
    #create token of zeros of one single timtestep for decoder_inputs_test (FOR NOW WE DO NOT DO IT...)
    #decoder_inputs_test = np.zeros( shape = (decoder_labels_test.shape[0], 1) + decoder_labels_test.shape[2:] )
    
    #BUT WE DO THIS:
    dec_in_shape = decoder_inputs_test.shape
    decoder_inputs_test = decoder_inputs_test[:,0,:,:,:].reshape((dec_in_shape[0],)+(1,)+dec_in_shape[2:])
    
    train_data = (encoder_inputs_train, decoder_inputs_train, decoder_labels_train)
    test_data = (encoder_inputs_test, decoder_inputs_test, decoder_labels_test)
    
    return train_data, test_data, info


def to_pandas_with_check_len_and_columns(path, coordinates, keep_columns_inputs, keep_columns_labels, enter_dir_specifier='/'):
    
    df_collection_enc_inputs=[]
    df_collection_dec_inputs=[]
    df_collection_labels=[]
    
    globbed_csv=[]
    for coord in coordinates:
        lat, long = coord
        globbed_csv.append(glob.glob(f"{path}{enter_dir_specifier}{lat}_{long}.csv")[0])
    
    len_df = None
    columns_df = None
    for i,csv_path in enumerate(globbed_csv):
        df, len_df, columns_df = basic_controls_and_manip(csv_path, len_df, columns_df)
        df_enc_inputs = df[keep_columns_inputs]
        df_dec_inputs = df[keep_columns_labels]
        df_labels = df[keep_columns_labels]
        df_collection_enc_inputs.append(df_enc_inputs)
        df_collection_dec_inputs.append(df_dec_inputs)
        df_collection_labels.append(df_labels)
    
    return df_collection_enc_inputs, df_collection_dec_inputs, df_collection_labels


def basic_controls_and_manip(csv_path, len_df, columns_df):
    
    df = pd.read_csv(csv_path)
    df.time = df.time.astype(np.datetime64) #set 'time' to np.datetime type
    df = df.set_index('time') # set 'time' column as index
    df = df.resample('H').first() #Fill all the missing timesteps with Nan 
    
    if df.isna().sum().sum() != 0: #interpolate if Nan are found
        df = df.interpolate()
    
    #not yet implemented    
    add_hour_column() #eventually add hour column
    add_month_column() #evenutally add month column
    
    if len_df is None:
        len_df = len(df)
        columns_df = len(df.columns)
    else:
        assert len_df==len(df), "found non matching len of dataframes"
        assert columns_df==len(df.columns), "found non matching number of columns in dataframes"
    
    return df, len_df, columns_df

def add_hour_column():
    pass

def add_month_column():
    pass


def make_tensors(df_unrolled_enc_inputs, df_unrolled_dec_inputs, df_unrolled_labels, window_enc, horizon, window_dec, jump):

    offs_window_enc=pd.DateOffset(hours=window_enc) 
    offs_horizon = pd.DateOffset(hours=horizon) 
    offs_window_dec=pd.DateOffset(hours=window_dec) 
    offs_one_h = pd.DateOffset(hours=1) 
    
    time_indexes=df_unrolled_enc_inputs.index
    
    encoder_inputs = []
    decoder_inputs = []
    decoder_labels = []
    
    index = 0
    
    while True:

        enc_in_start = time_indexes[index]
        enc_in_end = enc_in_start + offs_window_enc - offs_one_h 
        dec_out_start = enc_in_end + offs_horizon
        dec_out_end = dec_out_start + offs_window_dec - offs_one_h 
        dec_in_start = dec_out_start - offs_one_h
        dec_in_end = dec_out_end - offs_one_h

        enc_in_chunck = df_unrolled_enc_inputs.loc[enc_in_start: enc_in_end]
        dec_in_chunck = df_unrolled_dec_inputs.loc[dec_in_start : dec_in_end]
        lab_chunck = df_unrolled_labels.loc[dec_out_start : dec_out_end]

        if len(lab_chunck) < window_dec:
            break
        index+=jump
        
        encoder_inputs.append(enc_in_chunck.to_numpy())
        decoder_inputs.append(dec_in_chunck.to_numpy())
        decoder_labels.append(lab_chunck.to_numpy())

        
    return np.stack(encoder_inputs), np.stack(decoder_inputs), np.stack(decoder_labels)


def reshape_tensor_to_space_time_config(tens,channels,d_x,d_y):
    timesteps = tens.shape[1]
    return np.reshape(tens,(-1,timesteps,channels,d_x,d_y), order='F').transpose(0,1,2,4,3)



def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


#to remove headers

def remove_header(data_path,header_start=2, enter_dir_specifier='/'):
    files = os.listdir(data_path)
    for filename in files:
        newfilename = 'n_' + filename
        with open(data_path+enter_dir_specifier +filename, 'r') as inp, open(data_path+enter_dir_specifier +newfilename, 'w') as out:
            writer = csv.writer(out)
            i=0
            for row in csv.reader(inp):
                if i > header_start:
                    writer.writerow(row)
                i+=1
        os.remove(data_path+enter_dir_specifier +filename)
        os.rename(data_path+enter_dir_specifier +newfilename,data_path+enter_dir_specifier + filename)
        
        
    
#to manage failed downloads

def nested_from_iter(shape, iter):
    nest = np.zeros(shape).tolist()
    for i, el in enumerate(iter):
        nest[int(i/shape[0])][int(i%shape[1])] = el
    return nest

def nested_of_zeros(shape):
    return np.zeros(shape).tolist()

def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.time = df.time.astype(np.datetime64) #set 'time' to np.datetime type
    df = df.set_index('time') # set 'time' column as index
    df = df.resample('H').first() #Fill all the missing timesteps with Nan 

    if df.isna().sum().sum() != 0: #interpolate if Nan are found
        df = df.interpolate()
        
    return df

def find_neighbour_indexes(shape, indexes):
    
    i, j = indexes
    d0,d1 = shape
    assert i<d0 and j<d1
    assert i>=0 and j>=0
    neigh_indexes=[]
    
    if j+1<d1:
        neigh_indexes.append((i, j+1))
    if j-1>=0:
        neigh_indexes.append((i, j-1))
        
    if i+1< d0:
        neigh_indexes.append((i+1, j))
        
        if j+1<d1:
            neigh_indexes.append((i+1, j+1))
        if j-1>=0:
            neigh_indexes.append((i+1, j-1))
    
    if i-1 >= 0:
        neigh_indexes.append((i-1, j))
        
        if j+1<d1:
            neigh_indexes.append((i-1, j+1))
        if j-1>=0:
            neigh_indexes.append((i-1, j-1))
    
    return neigh_indexes


def replace_failed_download(shape, coordinates, failed_download, enter_dir_specifier='/' ):
    
    coordinates_matrix = nested_from_iter(shape, iter=coordinates)
    dfs_matrix= nested_of_zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if coordinates_matrix[i][j] not in failed_download:
                lat, long = coordinates_matrix[i][j]
                df = read_csv(f'data{enter_dir_specifier}{lat}_{long}.csv')
                
                if df.isna().sum().sum() < 1000:
                    dfs_matrix[i][j] = df
                else:
                    dfs_matrix = None
            else:
                dfs_matrix[i][j] = None    

    for i in range(shape[0]):
        for j in range(shape[1]):
            if dfs_matrix[i][j] is not None:
                pass
            else:
                neighb_indexes = find_neighbour_indexes(shape, (i,j))
                
                dfs_neighb=[]
                for neigh_index in neighb_indexes:
                    ii, jj = neigh_index
                    dfs_neighb.append(dfs_matrix[ii][jj])
                
                dfs_neighb = list(filter(lambda item: item is not None, dfs_neighb))
                assert len(dfs_neighb) != 0
                
                average_df = dfs_neighb[0]
                for df in dfs_neighb[1:]:
                    
                    average_df = average_df +  df
                
                average_df = average_df / len(dfs_neighb)
                
                lat, long = coordinates_matrix[i][j]
                average_df.to_csv(f'data{enter_dir_specifier}{lat}_{long}.csv')