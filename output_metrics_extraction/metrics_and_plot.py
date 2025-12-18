import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from pathlib import Path
import pathlib
import re


def compute_metrics(PATH_FOLDER, LOG_NAME, reference, cutoff, k_value, aggregate_metrics=True):
    """
    PATH_FOLDER : Path to the folder containing REPLx/Roundy/df_sorted_all.csv files
    LOG_NAME : Base name for the log files
    aggregate_metrics : Boolean, whether to aggregate metrics across rounds
    reference : DataFrame with reference scores, must contain 'hgvs_pro' and 'score' columns
    cutoff : float, threshold to binarize scores
    k_value : int, value of k for nDCG@k calculation
    """
    
    #Prepare reference dataframe adding shifted scores and binary labels
    rows = []
    reference['score_shifted'] = reference['score'] - reference['score'].min()
    reference['binary_true'] = [1 if x > cutoff else 0 for x in reference['score']]
    print(f"minimum value {reference['score'].min()}")
    print(f"maximum value {reference['score'].max()}")
    print(f"minimum_shifted {reference['score_shifted'].min()}")
    print(f"maximum_shifted {reference['score_shifted'].max()}")
    
    reference.sort_values(by='score' , ascending=False)
    
    for z in range(1,4):
        for i in range(1, 11):
            if not PATH_FOLDER.exists():
                print('Check path')
                    
                break
                
            else:
                df_sorted_all = pd.read_csv( PATH_FOLDER / f'REPL{z}' / f'Round{i}' / 'df_sorted_all.csv')

                #Prepare suggested dataframe, adding true scores and binary labels

                suggested = df_sorted_all[['variant' , 'y_pred']].copy()
                
                #Map true shifted scores and binary labels
                suggested['y_score'] = suggested['variant'].map(reference.set_index('hgvs_pro')['score_shifted']).dropna()
                suggested['binary_true'] = suggested['variant'].map(reference.set_index('hgvs_pro')['binary_true'])
                suggested['binary_predicted'] = [1 if x > cutoff else 0 for x in suggested['y_pred']]

                suggested.dropna(subset='y_score' , inplace=True)
                
                #Sort by predicted scores
                suggested.sort_values(by='y_pred', ascending=False, inplace=True)
                #Reset index, just in case
                suggested.reset_index(drop=True, inplace=True)

                #Reshape for nDCG
                y_true = suggested['y_score'].values.reshape(1, -1)
                y_score = suggested['y_pred'].values.reshape(1, -1)

                #nDCG@k
                ndcg = ndcg_score(y_true , y_score , k=k_value)


                #Spearman/p-value
                rho, pval = spearmanr(suggested['y_score'], suggested['y_pred'])

                #mAP
                AP = average_precision_score(suggested['binary_true'] , suggested['y_pred'])
                
                row_log = {'': f'Round{i}', f'nDCG@k={k_value}' : ndcg ,'Spearman_rho': rho, 'Spearman_pval': pval, 'AP' : AP }

                rows.append(row_log)
    metrics_df = pd.DataFrame(rows)

    if aggregate_metrics is False:
    #raw metrics, not for plotting
        metrics_df.to_csv(LOG_NAME + '.csv', index=False)
    #metrics aggregation and saving
    else:
        metrics_agg = metrics_df.groupby('').agg(ndcg_mean=(f'nDCG@k={k_value}' , 'mean'), ndcg_std=(f'nDCG@k={k_value}' , 'std') , spearman_mean=('Spearman_rho' , 'mean') , spearman_std=('Spearman_rho' , 'std'), m_average_precision_mean=('AP' , 'mean'), m_average_precision_std=('AP' , 'std')).reset_index()
        metrics_agg['round_n'] = metrics_agg[''].apply(lambda x : int(re.search(r'\d+' , x).group()))
        metrics_agg.sort_values(by='round_n', inplace=True)
        metrics_agg.to_csv(LOG_NAME + '_aggregate.csv', index=False)

        return print(metrics_agg)
    

def plot_metrics(legend_label , mean1 , mean2, mean1_name=str, mean2_name=str, save=False):
    """
    metrics plotter script
    
    graph_title : str, title of the graph
    legend_label: str, the metric you want to plot (i.e., 'Spearman', 'nDCG', 'm_average_precision') 
    :param mean1: mean metrics dataframe for dataset 1
    :param mean2: mean metrics dataframe for dataset 2
    :param mean1_name: mean1 name
    :param mean2_name: mean2 name
    :param save: save or not the figure as PNG
    """
    
    color=sns.color_palette("Set2")

    legend_label=legend_label
    plt.figure(figsize=(12,6))
    
    #dataset1
    plt.plot(range(1,11) , mean1[f'{legend_label.lower()}_mean'] , label=f'{legend_label} {mean1_name}', color=color[3] , marker='^')
    plt.fill_between(range(1,11), mean1[f'{legend_label.lower()}_mean'] - mean1[f'{legend_label.lower()}_std'] , mean1[f'{legend_label.lower()}_mean'] + mean1[f'{legend_label.lower()}_std'] , color=color[3], alpha=0.2)

    #dataset2
    plt.plot(range(1,11) , mean2[f'{legend_label.lower()}_mean'] , label=f'{legend_label} {mean2_name}', color=color[1] , marker='o')
    plt.fill_between(range(1,11), mean2[f'{legend_label.lower()}_mean'] - mean2[f'{legend_label.lower()}_std'] , mean2[f'{legend_label.lower()}_mean'] + mean2[f'{legend_label.lower()}_std'] , color=color[1], alpha=0.2)

    plt.xticks(range(1,11))
    plt.xlabel('Round')
    plt.legend()
    plt.title(f'{graph_title} {legend_label}' , fontsize=10)
    
    if legend_label.lower() == 'spearman':
        plt.ylim(-1,1)
        plt.xlim(0.8 , 10.2)
        plt.yticks([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
    if legend_label.lower() == 'ndcg':
        plt.ylim(0.5,1)
        plt.xlim(0.8 , 10.2)
        plt.yticks([0.4,0.6,0.8,1])
    if legend_label.lower() == 'm_average_precision' :
        plt.ylim(0,1)
        plt.xlim(0.8 , 10.2)
        plt.yticks([0,0.2,0.4,0.6,0.8,1])
        
    if save is True:
        plt.savefig(graph_title + f'_{legend_label}.png' , dpi=400)
        
    fig = plt.gcf()
    return fig