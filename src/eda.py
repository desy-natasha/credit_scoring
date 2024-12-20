from scipy import stats
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
    
def plot_custom(df, x_val, y_val = None, types = "hist", palette_col = "mako", fig_size = (15,4),hue_order = None, orient = "h", hue_val = None):
    fig = plt.figure(figsize=fig_size)
    palette = sns.color_palette(palette_col)

    if types == "hist":
        sns.histplot(df, x = x_val)
    elif types == "bar":
        sns.barplot(x = x_val, y = y_val, data = df, palette = palette, hue = hue_val, orient = orient, errorbar=None)
    elif types == "box":
        sns.boxplot(x = x_val, y = y_val,data = df, palette = palette)
    elif types == "count":
        sns.countplot(x = x_val ,data = df,hue = y_val, hue_order = hue_order, palette = palette)

def barplot_compare(data, cols):
    grouped = data.groupby(['Label',cols]).count().reset_index()
    all_data_grouped=data.groupby([cols]).count().reset_index()

    notdefault = grouped.loc[grouped.Label==0]
    default = grouped.loc[grouped.Label==1]

    # create figure and three axes side-by-side
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 5))

    # create bar plot for not default
    sns.barplot(x=cols, y='Credit amount', palette='mako', data=notdefault, ax=ax1)

    # add value labels to not default plot
    total=notdefault['Credit amount'].sum()
    for i, v in enumerate(notdefault['Credit amount']):
        ax1.text(i, v, f"{v} ({round(v/total * 100,2)}%)", color='red', fontweight='bold')

    ax1.set_xticklabels(data[cols].unique(), rotation=45)
    ax1.set_xlabel(cols)
    ax1.set_ylabel('Percentages of Not Default')
    ax1.set_title('Not Default')
    
    # create bar plot for default
    sns.barplot(x=cols, y='Credit amount', palette='mako', data=default, ax=ax2)

    # add value labels to default plot
    total=default['Credit amount'].sum()
    for i, v in enumerate(default['Credit amount']):
        ax2.text(i, v, f"{v} ({round(v/total * 100,2)}%)", color='red', fontweight='bold')
    
    ax2.set_xticklabels(data[cols].unique(), rotation=45)
    ax2.set_xlabel(cols)
    ax2.set_ylabel('Percentages of Default')
    ax2.set_title('Default')
    
    # create bar plot for all data
    sns.barplot(x=cols, y='Credit amount', palette='mako', data=all_data_grouped, ax=ax3)

    # add value labels to not fraud cluster plot
    total=all_data_grouped['Credit amount'].sum()
    for i, v in enumerate(all_data_grouped['Credit amount']):
        ax3.text(i, v, f"{v} ({round(v/total * 100,2)}%)", color='red', fontweight='bold')
        
    ax3.set_xticklabels(data[cols].unique(), rotation=45)
    ax3.set_xlabel(cols)
    ax3.set_ylabel('Percentages')
    ax3.set_title('All Data')
    
    plt.show()

def mannwhitney_test(df, label, variable,title,fig_size = (15,4)):
    print('Column: ', variable)

    label_value = df[label].unique()
    df_Label1 = df.loc[df[label] == label_value[0], variable]
    df_Label2 = df.loc[df[label] == label_value[1], variable]

    print('Non-Parametric t-test (independent)')
    stat, p = stats.mannwhitneyu(df_Label1, df_Label2)
    print("(stat = %.3f, p = %.3f)" % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('\nSame distribution (fail to reject H0)')
    else:
        print('\nDifferent distribution (reject H0)')
    N1 = df_Label1.nunique()
    N2 = df_Label2.nunique()

    degreeFreedom = (N1 + N2 - 2)
    std1 = df_Label1.std()
    std2 = df_Label2.std()
    
    std_N1N2 = sqrt( ((N1 - 1)*(std1)**2 + (N2 - 1)*(std2)**2) / degreeFreedom) 
    diff_mean = df_Label1.mean() - df_Label2.mean()
    MoE = stats.t.ppf(0.975, degreeFreedom) * std_N1N2 * sqrt(1/N1 + 1/N2)
    print ('The difference between groups is {:3.1f} [{:3.1f} to {:3.1f}] (mean [95% CI])'.format(diff_mean, diff_mean - MoE, diff_mean + MoE))
    df_plot = {'default':df.loc[df.Label==1][variable],'non default':df.loc[df.Label==0][variable]}
    fig, ax = plt.subplots(figsize=fig_size)
    ax.boxplot(df_plot.values(),vert=False)
    ax.set_yticklabels(df_plot.keys())
    ax.set_title(title)
    plt.show()

def chi2_test(df, cols):    
    # create the contingency table
    df_cont = pd.crosstab(index = df['Label'], columns = df[cols])
    
    # calculate degree of freedom
    degree_f = (df_cont.shape[0]-1) * (df_cont.shape[1]-1)
    # sum up the totals for row and columns
    df_cont['Total']= df_cont.sum(axis=1)
    df_cont.loc['Total']= df_cont.sum()
    
    # display the observed value table
    # display(df_cont)
    
    # create the expected value table
    df_exp = df_cont.copy()    
    df_exp.iloc[:,:] = np.multiply.outer(df_cont.sum(1).values,df_cont.sum().values) / df_cont.sum().sum()            

    # display the expected value table
    # display(df_exp)
        
    # calculate chi-square values
    df_chi2 = ((df_cont - df_exp)**2) / df_exp    
    df_chi2['Total']= df_chi2.sum(axis=1)
    df_chi2.loc['Total']= df_chi2.sum()
    
    # get chi-square score
    chi_square_score = df_chi2.loc['Total']['Total']
    
    # calculate the p-value
    from scipy import stats
    p = stats.distributions.chi2.sf(chi_square_score, degree_f)
    alpha = 0.05
    print('Column: ', cols)
    print('Chi-Square test')
    if p > alpha:
        print('Independent of each other (fail to reject H0)')
    else:
        print('Dependent on each other (reject H0)')
    print("(stat = %.3f, p = %.3f)" % (chi_square_score, p))
    print('---------------------------------')
