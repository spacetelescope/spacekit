
# ********* starskøpe.spacekit.computer ********* #
"""Helper functions for generating predictions, 
calculating scores, and evaluating a machine learning model.

# -----------------
# STATIC CLASS METHODS 
# -----------------
# * predictions 
#   get_preds()
#   fnfp()
#
# * Plots
#   keras_history()
#   plot_confusion_matrix()
#   roc_plots()
#
# * All-in-One-Shot
#   compute() 
#
#
TODO       
- save metriks to textfile/pickle objs and/or dictionaries
#
# ********* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ ********* #
"""
import pandas as pd
import numpy as np
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, jaccard_score, fowlkes_mallows_score
from sklearn.metrics import confusion_matrix, roc_curve

class Computer:

    def __init__(self):
        self.get_preds = get_preds()
        self.fnfp = fnfp()
        self.keras_history = keras_history()
        self.fusion_matrix = fusion_matrix()
        self.roc_plots = roc_plots()
        self.compute = Compute()


    @staticmethod
    def get_preds(X,y,model=None,verbose=False):
        if model is None:
            model=model
        # class predictions 
        y_true = y.flatten()
        
        y_pred = model.predict_classes(X).flatten() 
        preds = pd.Series(y_pred).value_counts(normalize=False)
        
        if verbose:
            print(f"y_pred:\n {preds}")
            print("\n")

        return y_true, y_pred

    @staticmethod
    def fnfp(X,y,model, training=False):

        import numpy as np

        y_pred = np.round( model.predict(X) )

        pos_idx = y==1
        neg_idx = y==0

        #tp = np.sum(y_pred[pos_idx]==1)/y_pred.shape[0]
        fn = np.sum(y_pred[pos_idx]==0)/y_pred.shape[0]

        #tn = np.sum(y_pred[neg_idx]==0)/y_pred.shape[0]
        fp = np.sum(y_pred[neg_idx]==1)/y_pred.shape[0]

        if training:
            print(f"FN Rate (Training): {round(fn*100,4)}%")
            print(f"FP Rate (Training): {round(fp*100,4)}%")
        else:
            print(f"FN Rate (Test): {round(fn*100,4)}%")
            print(f"FP Rate (Test): {round(fp*100,4)}%")

    @staticmethod
    def keras_history(history, figsize=(10,4)):
        """
        side by side sublots of training val accuracy and loss (left and right respectively)
        """
        
        import matplotlib.pyplot as plt
        
        fig,axes=plt.subplots(ncols=2,figsize=(15,6))
        axes = axes.flatten()

        ax = axes[0]
        ax.plot(history.history['accuracy'])
        ax.plot(history.history['val_accuracy'])
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Test'], loc='upper left')

        ax = axes[1]
        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        ax.set_title('Model Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    # FUSION_MATRIX()
    @staticmethod 
    def fusion_matrix(matrix, classes=None, normalize=True, title='Fusion Matrix', cmap='Blues',
        print_raw=False): 
        """
        FUSION MATRIX!
        -------------
        It's like a confusion matrix...without the confusion.
        
        matrix: can pass in matrix or a tuple (ytrue,ypred) to create on the fly 
        classes: class names for target variables
        """
        from sklearn import metrics                       
        from sklearn.metrics import confusion_matrix #ugh
        import itertools
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        
        # make matrix if tuple passed to matrix:
        if isinstance(matrix, tuple):
            y_true = matrix[0].copy()
            y_pred = matrix[1].copy()
            
            if y_true.ndim>1:
                y_true = y_true.argmax(axis=1)
            if y_pred.ndim>1:
                y_pred = y_pred.argmax(axis=1)
            fusion = metrics.confusion_matrix(y_true, y_pred)
        else:
            fusion = matrix
        
        # INTEGER LABELS
        if classes is None:
            classes=list(range(len(matrix)))

        #NORMALIZING
        # Check if normalize is set to True
        # If so, normalize the raw fusion matrix before visualizing
        if normalize:
            fusion = fusion.astype('float') / fusion.sum(axis=1)[:, np.newaxis]
            thresh = 0.5
            fmt='.2f'
        else:
            fmt='d'
            thresh = fusion.max() / 2.
        
        # PLOT
        fig, ax = plt.subplots(figsize=(10,10))
        plt.imshow(fusion, cmap=cmap, aspect='equal')
        
        # Add title and axis labels 
        plt.title(title) 
        plt.ylabel('TRUE') 
        plt.xlabel('PRED')
        
        # Add appropriate axis scales
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        #ax.set_ylim(len(fusion), -.5,.5) ## <-- This was messing up the plots!
        
        # Text formatting
        fmt = '.2f' if normalize else 'd'
        # Add labels to each cell
        #thresh = fusion.max() / 2.
        # iterate thru matrix and append labels  
        for i, j in itertools.product(range(fusion.shape[0]), range(fusion.shape[1])):
            plt.text(j, i, format(fusion[i, j], fmt),
                    horizontalalignment='center',
                    color='white' if fusion[i, j] > thresh else 'black',
                    size=14, weight='bold')
        
        # Add a legend
        plt.colorbar()
        plt.show() 
        return fusion, fig

    @staticmethod
    def roc_plots(X,y,model):
        """Calculates ROC_AUC score and plots Receiver Operator Characteristics (ROC)

        Arguments:
            X {feature set} -- typically X_test
            y {labels} -- typically y_test
            model {classifier} -- the model name for which you are calculting roc score

        Returns:
            roc -- roc_auc_score (via sklearn)
        """
        from sklearn import metrics
        from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

        y_true = y.flatten()
        y_hat = model.predict(X)

        fpr, tpr, thresholds = roc_curve(y_true, y_hat) 

        # Threshold Cutoff for predictions
        crossover_index = np.min(np.where(1.-fpr <= tpr))
        crossover_cutoff = thresholds[crossover_index]
        crossover_specificity = 1.-fpr[crossover_index]

        fig,axes=plt.subplots(ncols=2, figsize=(15,6))
        axes = axes.flatten()

        ax=axes[0]
        ax.plot(thresholds, 1.-fpr)
        ax.plot(thresholds, tpr)
        ax.set_title("Crossover at {0:.2f}, Specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))

        ax=axes[1]
        ax.plot(fpr, tpr)
        ax.set_title("ROC area under curve: {0:.2f}".format(roc_auc_score(y_true, y_hat)))
        plt.show()
        
        roc = roc_auc_score(y_true,y_hat)

        return roc


        ### COMPUTE SCORES ###

    @staticmethod
    def compute(X, y, model, hist=None, preds=True, summary=True, fusion=True, 
                classes=None, report=True, roc=True):
        """
        evaluates model predictions and stores the output in a dict
        returns `results`
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn import metrics
        from sklearn.metrics import jaccard_score,accuracy_score, recall_score, fowlkes_mallows_score

        # initialize a spare improbability drive
        res = {}
        res['model'] = model.name
        
        # class predictions 
        if preds:
            y_true = y.flatten()
            y_pred = model.predict_classes(X).flatten()
            res['preds'] = [y_pred]

        if summary:
            summary = model.summary()
            res['summary'] = model.summary


        # FUSION MATRIX
        if fusion:
            if classes is None:
                classes=set(y)
                #classes=['0','1']
            else:
                classes=classes
            # Plot fusion matrix
            FM = fusion_matrix(matrix=(y_true,y_pred), 
                                        classes=classes)
            res['FM'] = FM

        # ROC Area Under Curve
        if roc:
            ROC = roc_plots(X, y, model)
            res['ROC'] = ROC

        # CLASSIFICATION REPORT
        if report:
            num_dashes=20
            print('\n')
            print('---'*num_dashes)
            print('\tCLASSIFICATION REPORT:')
            print('---'*num_dashes)
            # generate report
            report = metrics.classification_report(y_true,y_pred)
            res['report'] = report
            print(report)


        # save to dict:
        res['jaccard'] = jaccard_score(y_true, y_pred)
        res['fowlkes'] = fowlkes_mallows_score(y_true,y_pred)
        res['accuracy'] = accuracy_score(y_true, y_pred)
        res['recall'] = recall_score(y_true, y_pred)
        
        #Plot Model Training Results (PLOT KERAS HISTORY)
        if hist is not None:
            HIST = keras_history(hist)
            res['HIST'] = HIST

        return res

