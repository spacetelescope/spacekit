import numpy as np
from scipy.ndimage.filters import uniform_filter1d

class Transformer:
    
    @staticmethod
    def hypersonic_pliers(path_to_train, path_to_test):
        
        """
        Using Numpy to extract data into 1-dimensional arrays
        separate target classes (y) for training and test data
        assumes y (target) is first column in dataframe
        
        #TODO: option to pass in column index loc for `y` if not default (0) 
        #TODO: option for `y` to already be 0 or 1 (don't subtract 1)
        #TODO: allow option for `y` to be categorical (convert via binarizer)
        """
        
        Train = np.loadtxt(path_to_train, skiprows=1, delimiter=',')
        X_train = Train[:, 1:]
        y_train = Train[:, 0, np.newaxis] - 1.
        
        Test = np.loadtxt(path_to_test, skiprows=1, delimiter=',')
        X_test = Test[:, 1:]
        y_test = Test[:, 0, np.newaxis] - 1.
        
        del Train,Test
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        
        return X_train, X_test, y_train, y_test


    @staticmethod
    def thermo_fusion_chisel(matrix1, matrix2=None):
            """
            Scales each array of a matrix to zero mean and unit variance.
            returns matrix/matrices of same shape as input but scaled
            matrix2 is optional - useful if data was already train-test split
            example: matrix1=X_train, matrix2=X_test
            
            """
 
            matrix1 = ((matrix1 - np.mean(matrix1, axis=1).reshape(-1,1)) / 
                np.std(matrix1, axis=1).reshape(-1,1))
            
            print("Mean: ",matrix1[0].mean())
            print("Variance: ",matrix1[0].std())
            
            if matrix2 is not None:
                matrix2 = ((matrix2 - np.mean(matrix2, axis=1).reshape(-1,1)) / 
                    np.std(matrix2, axis=1).reshape(-1,1))
            
                print("Mean: ",matrix2[0].mean())
                print("Variance: ",matrix2[0].std())
                return matrix1,matrix2
            else:
                return matrix1


    @staticmethod
    def babel_fish_dispenser(matrix1, matrix2=None, step_size=None, axis=2):
            """        
            Adds an input corresponding to the running average over a set number
            of time steps. This helps the neural network to ignore high frequency 
            noise by passing in a uniform 1-D filter and stacking the arrays. 
            
            **ARGS
            step_size: integer, # timesteps for 1D filter. defaults to 200
            axis: which axis to stack the arrays
            
            ex:
            babel_fish_dispenser(matrix1=X_train, matrix2=X_test, step_size=200)
            """
            if step_size is None:
                step_size=200
                
            # calc input for flux signal rolling avgs 
            filter1 = uniform_filter1d(matrix1, axis=1, size=step_size)
            # store in array and stack on 2nd axis for each obs of X data
            matrix1 = np.stack([matrix1, filter1], axis=2)
            
            if matrix2 is not None:
                filter2 = uniform_filter1d(matrix2, axis=1, size=step_size)
                matrix2 = np.stack([matrix2, filter2], axis=2)
                print(matrix1.shape,matrix2.shape)
                return matrix1,matrix2
            else:
                print(matrix1.shape)
                return matrix1


    @staticmethod
    def fast_fourier(matrix, bins):
        """
        takes in array and rotates #bins to the left as a fourier transform
        returns vector of length equal to input array
        """

        shape = matrix.shape
        fourier_matrix = np.zeros(shape, dtype=float)
        
        for row in matrix:
            signal = np.asarray(row)
            frequency = np.arange(signal.size/2+1, dtype=np.float)
            phase = np.exp(complex(0.0, (2.0*np.pi)) * frequency * bins / float(signal.size))
            ft = np.fft.irfft(phase * np.fft.rfft(signal))
            fourier_matrix += ft
        
        return fourier_matrix
        
        

