'''
defines functions used for plotting training, validation and test data from machine 
learning models in tensorflow

'''

from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from pandas import DataFrame
from numpy import where
from numpy import sum
from numpy import argmax
from IPython.display import display, HTML

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_metrics(history, c=None):  
    '''
    plot the metrics defined in the model
    '''
    
    metrics = ['loss', 'prc', 'precision', 'recall', 'auc', 'tp']
    for _, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        
        if c is not None:
            c=c
        else:
            c=0

        name = metric.replace("_"," ").capitalize()
        if metric=='loss':
            plt.subplot(3,3,_+1)
            plt.semilogy(history[metric], color=colors[c], label='Train')
            plt.semilogy(history['val_' + metric], color=colors[c+2], linestyle="-", label='Val')
            
        elif metric == 'tp':
            plt.subplot(3,3,_+1)
            plt.plot(history['val_' + metric], color=colors[c], linestyle="-", label='Val TP')
            plt.plot(history['val_fp'], color=colors[c+1], linestyle="--", label='Val FP')
            
        else:
            plt.subplot(3,3,_+1)
            plt.plot(history[metric], color=colors[c+1], label='Train')
            plt.plot(history['val_' + metric], color=colors[c+3], linestyle="-", label='Val')
    
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.title(name)
        plt.grid(visible=True, axis='both')
        plt.legend()

def plot_cm(labels, predictions, cmap):
    '''
    plot a confusion matrix
    '''
    cm = confusion_matrix(labels, predictions, normalize='true')
    heatmap(cm.round(4)*100, annot=True, fmt="f", cmap=cmap, annot_kws={'fontsize':'xx-large'})
    plt.title("Confusion matrix (%)")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    ax = plt.gca()
    ax.set_aspect('equal')
    print(f'NEOs Found (True Positives): {cm[1][1]*100:.2f}%')
    print(f'NEOs Missed (False Positives): {cm[1][0]*100:.2f}%')
    print(f'Non-NEOs Identified (True Negatives): {cm[0][0]*100:.2f}%')
    print(f'Non-NEOs Incorrectly Identified (False Negatives): {cm[0][1]*100:.2f}%')
    # print(f'Total NEOs: {sum(cm[1][1],cm[0][1])}\n')

def plot_roc(name, labels, predictions, **kwargs):
    '''
    plot an ROC curve
    '''
    fp, tp, thresholds = roc_curve(labels, predictions)
    # optimal_idx = argmax(tp - fp)
    # optimal_threshold = thresholds[optimal_idx]
    # print(optimal_idx, optimal_threshold)
    
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    # plt.plot(100*optimal_idx, 100*optimal_threshold, label=optimal_threshold)
    
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    # plt.xlim([-0.5,20])
    # plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    return fp, tp, thresholds

def plot_prc(name, labels, predictions, **kwargs):
    '''
    plt a precision-recall curve
    '''
    precision, recall, _ = precision_recall_curve(labels, predictions)
    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    
def quantify_prediction(prediction, labels, threshold):
    '''
    determine if predictions were correct
    '''
    p = DataFrame(prediction, columns=['Prediction'])
    p['Binary Pred'] = where(p['Prediction'] > threshold, 1, 0)
    p['Label'] = labels[:len(p)]
    p['Corr Pred'] = where(p['Binary Pred'] == p['Label'], 1, 0)
    counts = p['Corr Pred'].value_counts()
    print("\nPrediction Accuracy = ",(counts[1]/len(p['Corr Pred']))*100, "%\n")
    return p


def results(results):
    
    plt.subplot(1,3,1)
    plt.title('Receiver Operating Characteristic Curve')
    # plot_roc("Train", results['train_labels'], results['train_predictions'], color=colors[0])
    plot_roc("Train", results['train_labels'], results['train_predictions_logits'], color=colors[0])
    plot_roc("Test", results['test_labels'], results['test_predictions_logits'], color=colors[1], linestyle='--')
    plt.legend(loc='lower right')


    plt.subplot(1,3,2)
    plt.title('Precision-Recall Curve')
    # plot_prc("Train", results['train_labels'], results['train_predictions'], color=colors[0])
    plot_prc("Test", results['test_labels'], results['test_predictions_logits'], color=colors[1], linestyle='--')
    plt.legend(loc='lower left')

    plt.subplot(1,3,3)
    plot_cm(results['test_labels'], results['test_predictions_bool'], 'plasma')

def display_side_by_side(dfs:list, captions:list, tablespacing=5):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    for (caption, df) in zip(captions, dfs):
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += tablespacing * "\xa0"
    display(HTML(output))