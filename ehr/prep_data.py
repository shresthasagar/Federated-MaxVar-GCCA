import math
from tqdm import tqdm, trange
import pandas as pd
from collections import OrderedDict
import torch
import os


# sample 1 filenames
bene_2008_sample1 = 'ehr/data/sample1/Beneficiary_2008_Summary_File_Sample_1.csv'
bene_2009_sample1 = 'ehr/data/sample1/Beneficiary_2009_Summary_File_Sample_1.csv'
bene_2010_sample1 = 'ehr/data/sample1/Beneficiary_2010_Summary_File_Sample_1.csv'
carrier_claims_A_sample1 = 'ehr/data/sample1/Carrier_Claims_Sample_1A.csv'
carrier_claims_B_sample1 = 'ehr/data/sample1/Carrier_Claims_Sample_1B.csv'
inpatient_claims_sample1 = 'ehr/data/sample1/Inpatient_Claims_Sample_1.csv'
outpatient_claims_sample1 = 'ehr/data/sample1/Outpatient_Claims_Sample_1.csv'
pde_claims_sample1 = 'ehr/data/sample1/Prescription_Drug_Events_Sample_1.csv'

# sample 2 filenames
bene_2008_sample2 = 'ehr/data/sample2/DE1_0_2008_Beneficiary_Summary_File_Sample_2.csv'
bene_2009_sample2 = 'ehr/data/sample2/DE1_0_2009_Beneficiary_Summary_File_Sample_2.csv'
bene_2010_sample2 = 'ehr/data/sample2/DE1_0_2010_Beneficiary_Summary_File_Sample_2.csv'
carrier_claims_A_sample2 = 'ehr/data/sample2/DE1_0_2008_to_2010_Carrier_Claims_Sample_2A.csv'
carrier_claims_B_sample2 = 'ehr/data/sample2/DE1_0_2008_to_2010_Carrier_Claims_Sample_2B.csv'
inpatient_claims_sample2 = 'ehr/data/sample2/DE1_0_2008_to_2010_Inpatient_Claims_Sample_2.csv'
outpatient_claims_sample2 = 'ehr/data/sample2/DE1_0_2008_to_2010_Outpatient_Claims_Sample_2.csv'
pde_claims_sample2 = 'ehr/data/sample2/DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_2.csv'

# sample 2 filenames
bene_2008_sample3 = 'ehr/data/sample3/DE1_0_2008_Beneficiary_Summary_File_Sample_3.csv'
bene_2009_sample3 = 'ehr/data/sample3/DE1_0_2009_Beneficiary_Summary_File_Sample_3.csv'
bene_2010_sample3 = 'ehr/data/sample3/DE1_0_2010_Beneficiary_Summary_File_Sample_3.csv'
carrier_claims_A_sample3 = 'ehr/data/sample3/DE1_0_2008_to_2010_Carrier_Claims_Sample_3A.csv'
carrier_claims_B_sample3 = 'ehr/data/sample3/DE1_0_2008_to_2010_Carrier_Claims_Sample_3B.csv'
inpatient_claims_sample3 = 'ehr/data/sample3/DE1_0_2008_to_2010_Inpatient_Claims_Sample_3.csv'
outpatient_claims_sample3 = 'ehr/data/sample3/DE1_0_2008_to_2010_Outpatient_Claims_Sample_3.csv'
pde_claims_sample3 = 'ehr/data/sample3/DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_3.csv'


dgn_cols_str = ['ICD9_DGNS_CD_'+str(i+1) for i in range(10)]
pr_cols_str = ['ICD9_PRCDR_CD_'+str(i+1) for i in range(6)]
med_cols_str = ['HCPCS_CD_'+str(i+1) for i in range(45)]

def get_top_keys(d, num_keys):
    '''
    Returns a list of num_keys (key, value) that has the highest values
    '''
    top_list = sorted(d.items(), key=lambda x: x[1])
    return top_list[-num_keys:]

def get_count_dict_from_dataframe(df, df_cols):
    """
    @params:
        df : dataframe from the dataset with columns whose elements need to be counted
        df_cols: columns list to be search for
    @return:
        (dict_result, nan_count, list_slice) : -> tuple
        dict_result: dictionary containing key as the entry and the value as the number of occurences of that entry
        nan_count: number of nan entries
        list_slice: slice of the dataframe containing the final dict_result indices
    """
    dict_result = {}
    nan_cnt = 0
    df_view = df[df_col]
    for df_col in df_cols:
        for i in tqdm(range(len(df))):
            if isinstance(df_view[i], str):
                if df_view[i] not in dict_result:
                    dict_result[df_view[i]] = 1
                else:
                    dict_result[df_view[i]] += 1
            else:
                nan_cnt += 1
                
    return dict_result, nan_cnt

def co_occurence_mat(df, list1, list2, list1_cols, list2_cols):
    """
    Build a co-occurence list with co-occurence count of the elements from list1 with elements from list2
    @params:
        df: dataframe consisting of the data
        list1: list of ordered (key, value) tuples, ordered according to value
        list2: list of ordered (key, value) tuples, ordered according to value
        list1_cols: columns in df to search for list1 keys
        list2_cols: columns in df to search for list2 keys
    """
    def form_cosets(l1, l2):
        co_set = []
        for i in l1:
            for j in l2:
                co_set.append((i,j))
        return co_set

    list1_view = df[list1_cols]
    list2_view = df[list2_cols]
    
    list1_keys = set(x[0] for x in list1)
    list2_keys = set(x[0] for x in list2)
    
    co_occur_dict = OrderedDict()
    
    for i in range(len(list1)):
        for j in range(len(list2)):
            co_occur_dict[(list1[i][0], list2[j][0])] = [i,j,0]
    
    import time
    # this is too slow do not do it in a loop
    l1_view = list1_view[list1_cols]
    l2_view = list2_view[list2_cols]

    for i in trange(len(df)):
        # t1=time.time()
        l1 = set(l1_view.values[i])
        l2 = set(l2_view.values[i])
        # print('and values()', time.time() - t1); t1=time.time()

        l1_inter_keys = l1 & list1_keys
        l2_inter_keys = l2 & list2_keys
        # print('and oper', time.time() - t1); t1=time.time()
        co_set = form_cosets(l1_inter_keys, l2_inter_keys)
        
        for tup in co_set:
            co_occur_dict[tup][2] += 1
        # print('tup oper', time.time() - t1); t1=time.time()

    return co_occur_dict

def get_matrix(ordered_dict):
    """
    Contruct a matrix out of the ordered dictionary: 
    Entries in the dictionary should be of the following format
    (key1, key2): (row_idx, col_idx, count)
    """
    ordered_list = list(ordered_dict.items())
    
    num_rows, num_cols = ordered_list[-1][1][0], ordered_list[-1][1][1]
    
    mat = torch.zeros((num_rows+1, num_cols+1))
    for item in ordered_list:
        mat[item[1][0], item[1][1]] = item[1][2]
    return mat
    


def get_view_from_csv(filename, col_list1, col_list2, n_rows, n_cols, top_keys_file=None):
    """
    Pre-process CMS data and return a view of co-occurence using col_list1 as the row and col_list2 as the columns
    
    @params:
        filename: csv file containing ehr data (one of the claims data)
        col_list1: list of column names that constitute the examples 
        col_list2: list of column names that constitute the features
        n_rows: number of most frequent item from col_list1 to consider. This translates to number of examples of the view
        n_cols: number of most frequent items from the col_list2 to consider. This translates to number of features of the view
    @return:
        the view matrix and the ordered dictionary containing the element id of the view matrix
    """
    print("Computing count of entries from csv files...")
    df = pd.read_csv(filename)
    list1_dict, nan_cnt_list1 = get_count_dict_from_dataframe(df, col_list1)    
    list2_dict, nan_cnt_list2 = get_count_dict_from_dataframe(df, col_list2)
    print("Counts computed.")

    print("Getting the top n_rows and n_cols from the retrieved dictionary of counts...")
    if top_keys_file==None:
        list1 = get_top_keys(list1_dict, n_rows)
        list2 = get_top_keys(list2_dict, n_cols)
    else:
        df_keys = pd.read_csv(top_keys_file)
        list1_top_keys, nan_cnt_list1 = get_count_dict_from_dataframe(df_keys, col_list1)    
        list2_top_keys, nan_cnt_list2 = get_count_dict_from_dataframe(df_keys, col_list2)
        list1 = get_top_keys(list1_top_keys, n_rows)
        list2 = get_top_keys(list2_top_keys, n_cols)
        
    print("Fetching most frequent features completed.")

    print("Building co-occurence matrix...")
    co_occur_data = co_occurence_mat(df, list1, list2, col_list1, col_list2)
    return get_matrix(co_occur_data), co_occur_data

def get_top_list_from_all_views(filenames: list, col_list1, col_list2, n_rows_per_file=500, n_cols_per_file=300, shuffle=False):

    list1_dict = {}
    list2_dict = {}

    for filename in filenames:
        df_keys = pd.read_csv(filename)
        list1_top_keys, nan_cnt_list1 = get_count_dict_from_dataframe(df_keys, col_list1)    
        list2_top_keys, nan_cnt_list2 = get_count_dict_from_dataframe(df_keys, col_list2)
        list1 = dict(get_top_keys(list1_top_keys, n_rows_per_file))
        list2 = dict(get_top_keys(list2_top_keys, n_cols_per_file))
        
        list1_dict = {**list1_dict, **list1}
        list2_dict = {**list2_dict, **list2}


    if shuffle:
        import random
        l = list(list1_dict.items())
        random.shuffle(l)
        list1_dict = dict(l)

        l = list(list2_dict.items())
        random.shuffle(l)
        list2_dict = dict(l)

    return list(list1_dict.items()), list(list2_dict.items())

def get_all_views_combined(filenames, col_list1, col_list2, n_rows, n_cols):
    print('fetching frequency dictionaries')
    list1_dict, list2_dict = get_top_list_from_all_views(filenames, col_list1, col_list2, n_rows_per_file=n_rows, n_cols_per_file=n_cols)

    print("Building co-occurence matrix...")
    co_occur_data = []
    co_occur_mat = []
    for filename in filenames:
        df = pd.read_csv(filename)
        co_occur_data.append(co_occurence_mat(df, list1_dict, list2_dict, col_list1, col_list2))
        co_occur_mat.append(get_matrix(co_occur_data[-1]))
    return co_occur_mat, co_occur_data

def get_diagnosis_classes_list_from_dict(filename=None, data=None):
    """
    Get CMS classes from the ordered dictionary with keys as tuple of (diagnosis, category2)
    returns a list of classes index
    """
    if filename is not None:
        _, data = torch.load(filename)

    if isinstance(data, list):
        data = data[0]
    
    classes = []
    cl_list = ((100, 139), 
            (140, 239), 
            (240, 289),
            (290, 319),
            (320, 389),
            (390, 459), 
            (460, 519),
            (520, 579),
            (580, 629),
            (630, 679),
            (680, 709),
            (710, 739),
            (740, 759),
            (760, 779),
            (780, 799),
            (800, 999))

    keys = list(data.keys())
    num_rows = data[keys[-1]][0]+1
    num_cols = data[keys[-1]][1]+1
    classes = [0]*num_rows
    for i in trange(num_rows):
        # print(keys[i*num_cols][0])
        if keys[i*num_cols][0].startswith('0'):
            classes[i] = 1
        elif keys[i*num_cols][0].startswith('E'):
            classes[i] = 17
        elif keys[i*num_cols][0].startswith('V'):
            classes[i] = 18
        elif keys[i*num_cols][0].startswith('O'):
            classes[i] = 19
        else:
            num = int(keys[i*num_cols][0][:3])
            for j in range(len(cl_list)):
                if num >= cl_list[j][0] and num <= cl_list[j][1]:
                    classes[i] = j+1
    return classes

def test_data(data_dict, filename, key1_col_list, key2_col_list):
    df = pd.read_csv(filename)
    # get a random sample
    data_dict = list(data_dict[0].items())
    rand_id = torch.randint(len(data_dict), (1,1)).item()
    key1, key2 = data_dict[rand_id][0]
    true_count = data_dict[rand_id][1][-1]
    # key1 = '53019'
    # key2 = 'V1051'
    # true_count = 0

    test_count = 0
    # search in the file for co-occurence count of the given key pair
    
    for id1 in key1_col_list:
        for i in range(len(df)):
            if df[id1][i] == key1:
                for id2 in key2_col_list:
                    if df[id2][i] == key2:
                        test_count += 1
    if true_count==test_count:
        print('success. True count: {} and test count: {}'.format(true_count, test_count))
    else:
        print('the data is incorrect. True count: {} and test count: {}'.format(true_count, test_count))
    

if __name__ == '__main__':
    # inpatient = pd.read_csv(inpatient_claims_sample1)
    # pr_dict, nan_cnt_pr = get_count_dict_from_dataframe(inpatient, pr_cols_str)    
    # dgn_dict, nan_cnt_dgn = get_count_dict_from_dataframe(inpatient, dgn_cols_str)
    # _, labeled_data = torch.load('/scratch/sagar/Projects/federated_max_var_gcca/ehr/data/dictionary_data/combined_views')
    # test_data(labeled_data, os.path.join('/scratch/sagar/Projects/federated_max_var_gcca', inpatient_claims_sample1), dgn_cols_str, pr_cols_str)

    filenames = [outpatient_claims_sample1,
            outpatient_claims_sample2,
            outpatient_claims_sample3]

    mat, data = get_all_views_combined(filenames,dgn_cols_str, med_cols_str[:10], 5000, 500)
    torch.save((mat, data), 'data/dictionary_data/diag_med_1000_500')