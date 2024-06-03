# import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,roc_auc_score,classification_report,\
    confusion_matrix,roc_curve,precision_score,recall_score, matthews_corrcoef
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from scipy.stats import chi2


# write a function to conduct the chi-square test
def chi_square_test(threshold, df1, df2):
        # get the positive count in each group
    n1 = sum(df1["true"])
    n2 = sum(df2["true"])
    # print(n1, n2)
    p_values = []
    for i in threshold:
        # print("------------------",i,"------------------")
        df1['pred'] = np.where(df1['prob'] > i, 1, 0)
        df2['pred'] = np.where(df2['prob'] > i, 1, 0)
        rec1 = recall_score(df1['true'], df1['pred'])
        rec2 = recall_score(df2['true'], df2['pred'])

        # get the negative count in each group
        tp1 = n1 * rec1
        fp1 = n1 - tp1
        tp2 = n2 * rec2
        fp2 = n2 - tp2
        # make the confusion matrix
        confusion_matrix = [[tp1, fp1], [tp2, fp2]]
        # print(confusion_matrix)
        # run the chi-square test
        if tp1*fp1 == 0 or tp2*fp2 == 0:
            p_value = 1
        else:
            chi2, p_value, dof, expected = chi2_contingency(confusion_matrix)
        
        p_values.append(p_value)

    return p_values

# write a function to compare the two proportions, that is the recall
# This is the z-test
def compare_prop(threshold, df1, df2):
    '''
    This function is used to compare the recall between two groups
    :param threshold: the threshold of the probability
    :param df1: the data frame of group1
    :param df2: the data frame of group2
    '''
    n1 = sum(df1["true"])
    n2 = sum(df2["true"])
    p_values = []
    for i in threshold:
        df1['pred'] = np.where(df1['prob'] > i, 1, 0)
        df2['pred'] = np.where(df2['prob'] > i, 1, 0)
        rec1 = recall_score(df1['true'], df1['pred'])
        rec2 = recall_score(df2['true'], df2['pred'])
        # get the pooled proportion
        pooled_prop = (rec1*n1 + rec2*n2)/(n1 + n2)
        # get the standard error
        se = np.sqrt(pooled_prop*(1-pooled_prop)*(1/n1 + 1/n2))
        # get the z-score
        z_score = abs(rec1 - rec2)/se
        # get the p-value two-sided
        p_value =2*(1 - norm.cdf(z_score))
        if np.isnan(p_value) == True:
            p_value = 1
        p_values.append(p_value)
    return p_values

def compare_prop_acc(threshold, df1, df2):
    '''
    This function is used to compare the accuracy between two groups
    :param threshold: the threshold of the probability
    :param df1: the data frame of group1
    :param df2: the data frame of group2
    '''
    n1 = df1.shape[0]
    n2 = df2.shape[0]
    p_values = []
    for i in threshold:
        df1['pred'] = np.where(df1['prob'] > i, 1, 0)
        df2['pred'] = np.where(df2['prob'] > i, 1, 0)
        rec1 = accuracy_score(df1['true'], df1['pred'])
        rec2 = accuracy_score(df2['true'], df2['pred'])
        # get the pooled proportion
        pooled_prop = (rec1*n1 + rec2*n2)/(n1 + n2)
        # get the standard error
        se = np.sqrt(pooled_prop*(1-pooled_prop)*(1/n1 + 1/n2))
        # get the z-score
        z_score = abs(rec1 - rec2)/se
        # get the p-value two-sided
        p_value =2*(1 - norm.cdf(z_score))
        if np.isnan(p_value) == True:
            p_value = 1
        p_values.append(p_value)
    return p_values

# write a function to plot the accuracy, precision, recall, f1, specificity
def plot_metrics(threshold, acc, pre, rec, spec, f1):
    # plot the threshold vs. accuracy, precision, recall, f1, specificity
    plt.figure(figsize=(6, 6))
    plt.plot(threshold, acc, label='accuracy')
    plt.plot(threshold, pre, label='precision')
    plt.plot(threshold, rec, label='recall')
    plt.plot(threshold, f1, label='f1')
    plt.plot(threshold, spec, label='specificity')
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy, Precision, Recall, F1, Specificity')
    plt.title('Threshold vs. Accuracy, Precision, Recall, F1, Specificity')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    plt.show()

# define a function to calculate the accuracy, precision, recall, f1, specificity
def cal_acc_pre_rec_f1(threshold, df):
    acc = []
    pre = []
    rec = []
    spec = []
    f1 = []
    for i in threshold:
        df['pred'] = np.where(df['prob'] > i, 1, 0)
        acc.append(accuracy_score(df['true'], df['pred']))
        pre.append(precision_score(df['true'], df['pred']))
        rec.append(recall_score(df['true'], df['pred']))
        spec.append(1 - (confusion_matrix(df['true'], df['pred'])[0][1] / (confusion_matrix(df['true'], df['pred'])[0][0] + confusion_matrix(df['true'], df['pred'])[0][1])))
        f1.append(f1_score(df['true'], df['pred']))
    return acc, pre, rec, spec, f1

# write a function to plot the p-values
def plot_pvalues(threshold, p_values):
    # plot the threshold vs. p-values
    plt.figure(figsize=(6, 6))
    plt.plot(threshold, p_values)
    plt.xlabel('Threshold')
    plt.ylabel('P-values')
    plt.title('Threshold vs. P-values')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.axhline(y=0.05, color='r', linestyle='--')
    plt.grid()
    plt.show()

# write a function to plot the ROC curve
def plot_roc_curve(data):
    y_true = data['true']
    y_pred_prob = data['prob']
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], color='red', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    plt.show()
    print("The AUC score is: ", roc_auc_score(y_true, y_pred_prob))

# write a function to get the mertics
def get_outputs(data):
    cut_score = 0.5
    y_true = data['true']
    y_pred_prob = data['prob']
    y_predicted = np.where(y_pred_prob > cut_score, 1, 0)
    print(classification_report(y_true, y_predicted))
    print("The confusion matrix is: \n", confusion_matrix(y_true, y_predicted))
    print("The recall score is: ", recall_score(y_true, y_predicted))
    # print the accuracy score
    print("The accuracy score is: ", accuracy_score(y_true, y_predicted))
    # print the specificity score
    print("The specificity score is: ", recall_score(y_true, y_predicted, pos_label=0))
    print("The precision score is: ", precision_score(y_true, y_predicted))
    print("The f1 score is: ", f1_score(y_true, y_predicted), "\n")
    # print the matthews correlation coefficient
    print("The matthews correlation coefficient is: ", matthews_corrcoef(y_true, y_predicted))

# define a function to get the Separation Fairness metric
def separation(threshold, group1, group2, data):
    '''
    This function is used to calculate the separation fairness metric.
    :param threshold: the threshold for the fairness metric
    :param group1: the first group
    :param group2: the second group
    :param data: the dataset
    :return: the separation fairness metric
    :group1 and group2 are male, female, white, black, hispanic, asian, native, 
                           high, low, low-mid, mid-high only.
    '''
    import numpy as np
    if group1 in ["male","female"]:
        male = 1
        female = 0
        group = "X1SEX"
        print("Reminder: Group1 is male by default...")
        group1_data = data.loc[data[group] == male, ['prob', 'true']]
        group2_data = data.loc[data[group] == female, ['prob', 'true']]
    elif group1 in ["white", "black", "hispanic", "asian", "native"]:
        group = "X1RACE"
        group1_data = data.loc[data[group] == group1, ['prob', 'true']]
        group2_data = data.loc[data[group] == group2, ['prob', 'true']]
    else:
        group = "SES_level"
        group1_data = data.loc[data[group] == group1, ['prob', 'true']]
        group2_data = data.loc[data[group] == group2, ['prob', 'true']]

    
    # get the metrics for binary classification of group1 and group2
    acc1, pre1, rec1, spec1, f11 = cal_acc_pre_rec_f1(threshold=threshold, df=group1_data)
    acc2, pre2, rec2, spec2, f12 = cal_acc_pre_rec_f1(threshold=threshold, df=group2_data)

    
    print("--------------------", group1, "--------------------")
    # print the metrics
    get_outputs(data=group1_data)
    # plot the metrics
    plot_metrics(threshold, acc1, pre1, rec1, spec1, f11)
    # plot the ROC curve
    plot_roc_curve(data=group1_data)
    print("--------------------", group2, "--------------------")
    # print the metrics
    get_outputs(data=group2_data)
    plot_metrics(threshold, acc2, pre2, rec2, spec2, f12)
    plot_roc_curve(data=group2_data)

    # conduct the chi-square test on the recall
    p_values = compare_prop(threshold=threshold, df1=group1_data, df2=group2_data)
    print("--------------------", group1, "vs.", group2, "--------------------")
    # plot the p-values
    plot_pvalues(threshold, p_values)

# write a function to test statistical disparity
# this function will return p-values
def stats_disparity(threshold, group1, group2, data):
    """
    This function is used to test statistical disparity between two groups
    :param threshold: a list of threshold values
    :param group1: the first group
    :param group2: the second group
    :param data: the data frame
    :return: a list of p-values
    """
    if group1 in ["male","female"]:
        male = 1
        female = 0
        group = "X1SEX"
        print("Reminder: Group1 is male by default...")
        group1_data = data.loc[data[group] == male, ['prob', 'true']]
        group2_data = data.loc[data[group] == female, ['prob', 'true']]
    elif group1 in ["white", "black", "hispanic", "asian", "native"]:
        group = "X1RACE"
        group1_data = data.loc[data[group] == group1, ['prob', 'true']]
        group2_data = data.loc[data[group] == group2, ['prob', 'true']]
    else:
        group = "SES_level"
        group1_data = data.loc[data[group] == group1, ['prob', 'true']]
        group2_data = data.loc[data[group] == group2, ['prob', 'true']]
    
    # get the sizes of each group
    n1 = group1_data.shape[0]
    n2 = group2_data.shape[0]
    print("The size of group1 is: ", n1)
    print("The size of group2 is: ", n2)
    p_values = []
    outputs = []
    for i in threshold:
        group1_data['pred'] = np.where(group1_data['prob'] > i, 1, 0)
        group2_data['pred'] = np.where(group2_data['prob'] > i, 1, 0)
        # get the proportion of positive outcome
        group1_pos = group1_data['pred'].mean()
        group2_pos = group2_data['pred'].mean()
        paired_output = (group1_pos, group2_pos)
        outputs.append(paired_output)
        # compare the proportion of positive outcome using z-test
        # get the pooled proportion
        pooled_prop = (group1_pos*n1 + group2_pos*n2)/(n1 + n2)
        # get the standard error
        se = np.sqrt(pooled_prop*(1-pooled_prop)*(1/n1 + 1/n2))
        # get the z-score
        z_score = abs(group1_pos - group2_pos)/se
        # get the p-value two-sided
        p_value =2*(1 - norm.cdf(z_score))
        if np.isnan(p_value) == True:
            p_value = 1
        p_values.append(p_value)
    
    return p_values

def DAF_LR(threshold, fair, group1, group2, data):
    '''
    This function is used to calculate the DAF on given group and fair variables
    :param threshold: the threshold of the probability
    :param fair: the fair variable
    :param group: the group variable, it should be the levels of X1SEX, X1RACE only
    '''
    if group1 in ["white", "black", "hispanic", "asian", "native"]:
        # subset the data based on the group1 and group2
        data_subgroup = data.loc[(data['X1RACE'].isin([group1,group2])), :]
        group = 'X1RACE'
    else:
        data_subgroup = data.loc[(data['X1SEX'].isin([group1,group2])), :]
        group = 'X1SEX'

    # make a list to load the p values
    a2a3 = []
    a2 = []
    a3 = []
    for i in threshold:
        # print(i)
        data_subgroup['pred'] = np.where(data_subgroup['prob'] > i, 1, 0)
        # note when running the logistic regression, it is possible that there are
        # so few observations in one of the groups that the model cannot be fit
        # so the threshold should be set to avoid this problem
        # print(data_subgroup['pred'].value_counts())
        # add a condition to avoid the error of singular matrix
        if len(data_subgroup['pred'].unique()) == 1:
            a2a3.append(1)
            a2.append(1)
            a3.append(1)
            continue
        # FIRST STAGE: test whether a2=a3=0 to determine whether there is DAF
        logit_model_full = sm.Logit.from_formula('pred ~ {} * {}'.format(fair, group), data=data_subgroup).fit()
        logit_model_constrained = sm.Logit.from_formula('pred ~ {}'.format(fair), data=data_subgroup).fit()
        # Calculate the log-likelihood for the full and constrained models
        ll_full = logit_model_full.llf
        ll_constrained = logit_model_constrained.llf
        # Compute the test statistic
        lr_statistic = 2 * (ll_full - ll_constrained)
        # Compute the degrees of freedom as the difference in number of parameters
        df = logit_model_full.df_model - logit_model_constrained.df_model
        # Compute the p-value for the test statistic using chi-square distribution
        p_value = chi2.sf(lr_statistic, df)
        a2a3.append(p_value)

        # SECOND STAGE: test whether a2=0 to determine whether there is uniform DAF
        logit_model_full = sm.Logit.from_formula('pred ~ {} + {}'.format(fair, group), data=data_subgroup).fit()
        logit_model_constrained = sm.Logit.from_formula('pred ~ {}'.format(fair), data=data_subgroup).fit()
        # Calculate the log-likelihood for the full and constrained models
        ll_full = logit_model_full.llf
        ll_constrained = logit_model_constrained.llf
        # Compute the test statistic
        lr_statistic = 2 * (ll_full - ll_constrained)
        # Compute the degrees of freedom as the difference in number of parameters
        df = logit_model_full.df_model - logit_model_constrained.df_model
        # Compute the p-value for the test statistic using chi-square distribution
        p_value = chi2.sf(lr_statistic, df)
        a2.append(p_value)

        # THIRD STAGE: test whether a3=0 to determine whether there is non-uniform DAF
        logit_model_full = sm.Logit.from_formula('pred ~ {} * {}'.format(fair, group), data=data_subgroup).fit()
        logit_model_constrained = sm.Logit.from_formula('pred ~ {} + {}'.format(fair, group), data=data_subgroup).fit()
        # Calculate the log-likelihood for the full and constrained models
        ll_full = logit_model_full.llf
        ll_constrained = logit_model_constrained.llf
        # Compute the test statistic
        lr_statistic = 2 * (ll_full - ll_constrained)
        # Compute the degrees of freedom as the difference in number of parameters
        df = logit_model_full.df_model - logit_model_constrained.df_model
        # Compute the p-value for the test statistic using chi-square distribution
        p_value = chi2.sf(lr_statistic, df)
        a3.append(p_value)
    
    return a2a3, a2, a3


# write a function to conduct the MH test
def DAF_MH(threshold, fair, group1, group2, data, K=10):
    """
    This function is to conduct the Mantel-Haenszel test for DAF
    :param threshold: the threshold of probability
    :param fair: the fairness variable
    :param K: the number of stata of the fairness variable
    :param group1: the name of group1
    :param group2: the name of group2
    :param data: the dataset
    """
    if group1 in ["white", "black", "hispanic", "asian", "native"]:
        # subset the data based on the group1 and group2
        data_subgroup = data.loc[(data['X1RACE'].isin([group1,group2])), :]
        group = 'X1RACE'
    else:
        data_subgroup = data.loc[(data['X1SEX'].isin([group1,group2])), :]
        group = 'X1SEX'

    # separate the fair variable into K bins.
    # NOTE: for some of faire variables, the number of unique values is less than K
    # so we need to check if the number of unique values is less than K
    if data_subgroup[fair].nunique() < K:
        print(f"Warning: The number of unique values in '{fair}' is less than {K}, which may result in an error. Here use the duplicates='drop' to drop the duplicates=drop")
    # create a new column to indicate the level of the fair variable
    data_subgroup['fair_level'] = pd.qcut(data_subgroup[fair], K, labels=False, duplicates='drop')
    # get the levels of the fair variable
    levels = data_subgroup['fair_level'].unique()

    # make a list to load the p values
    p_values = []
    for i in threshold:
        # print(i)
        data_subgroup['pred'] = np.where(data_subgroup['prob'] > i, 1, 0)
        count_2_pos_list = []
        count_2_pos_exp_list = []
        count_2_pos_var_list = []
        for j in levels:
            # NOTE: here we need to subset the data based on the fair variable's level
            # rather than the K levels since pd.qcut may drop some levels
            data_subgroup_ = data_subgroup.loc[(data_subgroup['fair_level'] == j), :]
            # get the group1's subset
            data_subgroup_1 = data_subgroup_.loc[(data_subgroup_[group] == group1), :]
            # get the group2's subset
            data_subgroup_2 = data_subgroup_.loc[(data_subgroup_[group] == group2), :]
            # get the group size
            n1 = data_subgroup_1.shape[0]
            n2 = data_subgroup_2.shape[0]
            nk = data_subgroup_.shape[0]
            # count the positive in each group
            count_1_pos = data_subgroup_1['pred'].sum()
            count_2_pos = data_subgroup_2['pred'].sum()
            # count the negative in each group
            count_1_neg = n1 - count_1_pos
            count_2_neg = n2 - count_2_pos
            # get the the expected number of positive case in the reference group
            # note here the expected number's method is slightly different from the one in the paper
            # but the result is the same
            expected_2 = n2 * (count_1_pos + count_2_pos) / (n1 + n2)
            # get the the variance of the positive case in the reference group
            var_2 = (count_2_pos+count_2_neg)*(count_2_pos+count_1_pos)*(count_1_pos+count_1_neg)*(count_1_neg+count_2_neg)/(nk**2*(nk-1))
            count_2_pos_list.append(count_2_pos)
            count_2_pos_exp_list.append(expected_2)
            count_2_pos_var_list.append(var_2)
        # calculate the chi-square statistics
        part_1 = abs(np.sum(np.array(count_2_pos_list))- np.sum(np.array(count_2_pos_exp_list)))
        chi_square = (part_1-0.5)**2/np.sum(np.array(count_2_pos_var_list))
        # get the p-value
        # TODO: the degree of freedom is correct or not??? It should be 1.
        p_value = 1-chi2.cdf(chi_square, 1)
        # print(p_value)
        p_values.append(p_value)
    return p_values


def DAF_MH_MultiFV(threshold, fair, group1, group2, data, K=5):
    """
    This function is used to get the MH test p-values for multiple fair variables.
    :param threshold: the threshold to get the predicted labels
    :param fair: the list of fair variables, the elements should be strings,i.e., the variable's name
    :param group1: the name of group1, i.e., the majority group
    :param group2: the name of group2, i.e., the minority group
    :param data: the dataset
    """
    if group1 in ["white", "black", "hispanic", "asian", "native","non-white","non-hispanic"]:
        # subset the data based on the group1 and group2
        data_subgroup = data.loc[(data['X1RACE'].isin([group1,group2])), :]
        group = 'X1RACE'
    else:
        data_subgroup = data.loc[(data['X1SEX'].isin([group1,group2])), :]
        group = 'X1SEX'

    # separate the fair variable into K bins.
    # NOTE: for some of faire variables, the number of unique values is less than K
    # so we need to check if the number of unique values is less than K
    # using a list to load number of levels of each fair variable
    levels_list = []
    name_tempt_list = []
    for index, fair_ in enumerate(fair):
        if data_subgroup[fair_].nunique() < K:
            print(f"Warning: The number of unique values in '{fair_}' is less than {K}, which may result in an error. Here use the duplicates='drop' to drop the duplicates=drop")
        # create a new column to indicate the level of the fair variable
        name_tempt = 'fair_{}_levels'.format(index)
        data_subgroup[name_tempt] = pd.qcut(data_subgroup[fair_], K, labels=False, duplicates='drop')
        # get the levels of the fair variable
        levels_num= data_subgroup[name_tempt].nunique()
        levels_list.append(levels_num)
        name_tempt_list.append(name_tempt)
    # get the number of fair variables
    num_fair = len(fair)

    # make a list to load the p values
    p_values = []
    for cut in threshold:
        # print(i)
        data_subgroup['pred'] = np.where(data_subgroup['prob'] > cut, 1, 0)
        count_2_pos_list = []
        count_2_pos_exp_list = []
        count_2_pos_var_list = []
        for i in range(levels_list[0]): # extract the number of fair variables
            for j in range(levels_list[1]): # extract the number of levels of each fair variable
                for k in range(levels_list[2]):
                    # NOTE: here we need to subset the data based on the fair variable's level
                    # rather than the K levels since pd.qcut may drop some levels
                    # get the subgroup
                    data_subgroup_ = data_subgroup.loc[(data_subgroup[name_tempt_list[0]] == i) & (data_subgroup[name_tempt_list[1]] == j) & (data_subgroup[name_tempt_list[2]] == k), :]
                    print(data_subgroup_.shape)
                    # get the group1's subset
                    data_subgroup_1 = data_subgroup_.loc[(data_subgroup_[group] == group1), :]
                    # get the group2's subset
                    data_subgroup_2 = data_subgroup_.loc[(data_subgroup_[group] == group2), :]
                    # get the group size
                    n1 = data_subgroup_1.shape[0]
                    n2 = data_subgroup_2.shape[0]
                    nk = data_subgroup_.shape[0]
                    # count the positive in each group
                    count_1_pos = data_subgroup_1['pred'].sum()
                    count_2_pos = data_subgroup_2['pred'].sum()
                    # count the negative in each group
                    count_1_neg = n1 - count_1_pos
                    count_2_neg = n2 - count_2_pos
                    # get the the expected number of positive case in the reference group
                    # note here the expected number's method is slightly different from the one in the paper
                    # but the result is the same
                    expected_2 = n2 * (count_1_pos + count_2_pos) / (n1 + n2)
                    # get the the variance of the positive case in the reference group
                    var_2 = (count_2_pos+count_2_neg)*(count_2_pos+count_1_pos)*(count_1_pos+count_1_neg)*(count_1_neg+count_2_neg)/(nk**2*(nk-1))
                    count_2_pos_list.append(count_2_pos)
                    count_2_pos_exp_list.append(expected_2)
                    count_2_pos_var_list.append(var_2)
            # calculate the chi-square statistics
        part_1 = abs(np.sum(np.array(count_2_pos_list))- np.sum(np.array(count_2_pos_exp_list)))
        chi_square = (part_1-0.5)**2/np.sum(np.array(count_2_pos_var_list))
        # get the p-value
        # TODO: the degree of freedom is correct or not??? It should be 1.
        p_value = 1-chi2.cdf(chi_square, 1)
        # print(p_value)
        p_values.append(p_value)
    return p_values


def DAF_LR_MultiFV(threshold, fair, group1, group2, data):
    '''
    This function is used to calculate the DAF on given group and fair variables
    :param threshold: the threshold of the probability
    :param fair: the fair variable
    :param group: the group variable, it should be the levels of X1SEX, X1RACE only
    :NOTE: the using the multi-level categorical vairable with some of the levels are empty will result in an error
    : of singular matrix. So we need to check the number of observations in each level of the fair variable
    : and the beginining of the threshold should be set to avoid this problem.
    '''
    if group1 in ["white", "black", "hispanic", "asian", "native","other","non-white","non-hispanic"]:
        # subset the data based on the group1 and group2
        data_subgroup = data.loc[(data['X1RACE'].isin([group1,group2])), :]
        group = 'X1RACE'
        # based on the note above, need to change X1RACE to binary categorical variable
        data_subgroup['X1RACE'] = data_subgroup['X1RACE'].cat.remove_unused_categories()
        # check the distribution of X1RACE
        print(data_subgroup['X1RACE'].value_counts())
    else:
        data_subgroup = data.loc[(data['X1SEX'].isin([group1,group2])), :]
        group = 'X1SEX'

    # make a list to load the p values
    non_daf = []
    uni_daf = []
    non_uni_daf = []
    for i in threshold:
        # print(i)
        data_subgroup['pred'] = np.where(data_subgroup['prob'] > i, 1, 0)
        # note when running the logistic regression, it is possible that there are
        # so few observations in one of the groups that the model cannot be fit
        # so the threshold should be set to avoid this problem
        # print(data_subgroup['pred'].value_counts())
        # add a condition to avoid the error of singular matrix
        # insert a check point
        print("------------------ Threshold: ",i, " ------------------")
        # check the distribution of the predicted labels
        print(data_subgroup['pred'].value_counts())
        # write a condition to avoid the singular matrix for logistic regression
        if len(data_subgroup['pred'].unique()) == 1:
            non_daf.append(1)
            uni_daf.append(1)
            non_uni_daf.append(1)
            continue
        # FIRST STAGE: test whether a4=a5=a6=a7=0 to determine whether there is DAF
        # the fair variable is a list of 3 strings, re-fit the logistic regression
        print("------------------ check point 2 ------------------")
        logit_model_full = sm.Logit.from_formula('pred ~ {}*{} + {}*{} + {}*{}'.format(fair[0], group, fair[1], group,fair[2], group), data=data_subgroup).fit()
        #print(logit_model_full.summary())
        # Calculate the log-likelihood for the full and constrained models
        ll_full = logit_model_full.llf
        # fit the logistic regression with only the fair variable
        print("------------------ check point 3 ------------------")
        logit_model_constrained = sm.Logit.from_formula('pred ~ {}+{}+{}'.format(*fair), data=data_subgroup).fit()
        # print(logit_model_constrained.summary())
        ll_constrained = logit_model_constrained.llf
        # Compute the test statistic
        lr_statistic = 2 * (ll_full - ll_constrained)
        # Compute the degrees of freedom as the difference in number of parameters
        df = logit_model_full.df_model - logit_model_constrained.df_model
        # Compute the p-value for the test statistic using chi-square distribution
        p_value = chi2.sf(lr_statistic, df)
        non_daf.append(p_value)

        print("------------------ check point 4 ------------------")
        # SECOND STAGE: test whether coefficient of group variable sig to determine whether there is uniform DAF
        logit_model_full = sm.Logit.from_formula('pred ~ {}+{}+{}+{}'.format(*fair, group), data=data_subgroup).fit()
        ll_full = logit_model_full.llf
        # since the constraint model is actually same to the constrained mode above, so we can use the same model
        logit_model_constrained = sm.Logit.from_formula('pred ~ {}+{}+{}'.format(*fair), data=data_subgroup).fit()
        # Calculate the log-likelihood for the full and constrained models
        ll_constrained = logit_model_constrained.llf
        # Compute the test statistic
        lr_statistic = 2 * (ll_full - ll_constrained)
        # Compute the degrees of freedom as the difference in number of parameters
        df = logit_model_full.df_model - logit_model_constrained.df_model
        # Compute the p-value for the test statistic using chi-square distribution
        p_value = chi2.sf(lr_statistic, df)
        uni_daf.append(p_value)

        print("------------------ check point 5 ------------------")
        # THIRD STAGE: test whether a3=0 to determine whether there is non-uniform DAF
        logit_model_full = sm.Logit.from_formula('pred ~ {}*{} + {}*{} + {}*{}'.format(fair[0], group,\
                            fair[1], group,fair[2], group), data=data_subgroup).fit()
        ll_full = logit_model_full.llf
        logit_model_constrained = sm.Logit.from_formula('pred ~ {}+{}+{}+{}'.format(*fair, group), data=data_subgroup).fit()
        # Calculate the log-likelihood for the full and constrained models
        ll_constrained = logit_model_constrained.llf
        # Compute the test statistic
        lr_statistic = 2 * (ll_full - ll_constrained)
        # Compute the degrees of freedom as the difference in number of parameters
        df = logit_model_full.df_model - logit_model_constrained.df_model
        # Compute the p-value for the test statistic using chi-square distribution
        p_value = chi2.sf(lr_statistic, df)
        non_uni_daf.append(p_value)
    
    return non_daf, uni_daf, non_uni_daf