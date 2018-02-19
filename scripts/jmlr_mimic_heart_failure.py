import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (12, 8)
font = {'family' : 'sans-serif',
        'sans-serif' : 'Helvetica',
        'weight' : 'normal'}
matplotlib.rc('font', **font)
global_fig_format = 'pdf'

import os
import numpy as np
import pandas as pd
import seaborn as sns
import time
import pickle
from array import array

def do_qc(time, value, lb=None, ub=None, min_period=None, name=None):
    # remove null values
    new_time = time[~np.isnan(value)]
    new_val = value[~np.isnan(value)]
    
    new_val = new_val[np.where(new_time > 0.)]
    new_time = new_time[np.where(new_time > 0.)]

    # QC by lower bound
    if(lb is not None):
        valid_idx = np.where(new_val > lb)[0]
        new_time = new_time[valid_idx]
        new_val = new_val[valid_idx]
    
    # QC by upper bound
    if(ub is not None):
        valid_idx = np.where(new_val <= ub)[0]
        new_time = new_time[valid_idx]
        new_val = new_val[valid_idx]
    return new_time, new_val


"""
select cohort
"""
cohort = 'heart_failure'
output_dir = '/data/lifangc/mimic/cohort2/{}'.format(cohort)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""
setup path to raw data
"""
mimic_dir = '/data/lifangc/mimic/data/mimic3v1.4/'
icddef_file_name = 'D_ICD_DIAGNOSES.csv.gz'
adm_file_name = 'ADMISSIONS.csv.gz'
diag_file_name = 'DIAGNOSES_ICD.csv.gz'

lab_item_file_name = 'D_LABITEMS.csv.gz'
lab_data_file_name = 'LABEVENTS.csv.gz'

chart_item_file_name = 'D_ITEMS.csv.gz'
chart_data_file_name = 'CHARTEVENTS.csv.gz'

"""
first-pass filtering through checking outcomes and data flag
"""
icddef_df = pd.read_csv(os.path.join(mimic_dir, icddef_file_name), compression='gzip')
diag_df = pd.read_csv(os.path.join(mimic_dir, diag_file_name), compression='gzip')
adm_df = pd.read_csv(os.path.join(mimic_dir, adm_file_name), compression='gzip')

# be aware if memory usage when loading chart data! (need ~100GB and 15 mins. on fat server)
t0 = time.time()
chart_df = pd.read_csv(os.path.join(mimic_dir, chart_data_file_name), compression='gzip')
lab_df = pd.read_csv(os.path.join(mimic_dir, lab_data_file_name), compression='gzip')
print('elapsed time for loading the dataframes: {} seconds'.format(time.time()-t0))

"""
define target ICD-9 code for filtering
"""
if(cohort == 'heart_failure'):
    target_icd9_list = icddef_df[icddef_df['ICD9_CODE'].str.startswith('428')]['ICD9_CODE'].values
else:
    raise NotImplementedError
print(target_icd9_list)

target_hadm = np.unique(diag_df[diag_df['ICD9_CODE'].isin(target_icd9_list)]['HADM_ID'].values)
print('after ICD-9 filtering: ', len(target_hadm))

"""
filter by outcomes and data availability
"""
match_adm = adm_df[adm_df['HADM_ID'].isin(target_hadm)]
match_adm = match_adm[(match_adm['DISCHARGE_LOCATION'] != 'DEAD/EXPIRED') & (match_adm['HAS_CHARTEVENTS_DATA'] == 1)]
target_hadm = np.unique(match_adm['HADM_ID'].values)
print('after outcome filtering: ', len(target_hadm))

"""
second-pass filtering through data quality control
"""
cohort_chart_df = chart_df[chart_df['HADM_ID'].isin(target_hadm)]
cohort_lab_df = lab_df[lab_df['HADM_ID'].isin(target_hadm)]

vital_item = [(0, 'RR', 220210),
              (1, 'HR', 220045),
              (3, 'SBP', 220179),
              (4, 'Temp', 223761)
              ]
vital_bound_list = [(0., 70.), (0., 300.), (0., 260), (90., 110.)]
lab_item = [(6, 'BUN', 51006),
            (7, 'CO2', 50804),
            (8, 'Calcium', 50893),
            (9, 'Chloride', 50902),
            (10, 'Creatinine', 50912),
            (12, 'Glucose', 50931),
            (13, 'Hct', 51221),
            (14, 'Hgb', 51222),
            (15, 'MCH', 51248),
            (16, 'MCHC', 51249),
            (17, 'MCV', 51250),
            (18, 'INR', 51237),
            (19, 'PT', 51274),
            (20, 'PTT', 51275),
            (21, 'Platelet', 51265),
            (22, 'Potassium', 50971),
            (23, 'RBC', 51279),
            (24, 'RDW', 51277),
            (25, 'Sodium', 50983),
            (26, 'WBC', 51301)
            ]
chart_item_id_array = [x[2] for x in vital_item]
cohort_chart_df = cohort_chart_df[cohort_chart_df['ITEMID'].isin(chart_item_id_array)]
lab_item_id_array = [x[2] for x in lab_item]
cohort_lab_df = cohort_lab_df[cohort_lab_df['ITEMID'].isin(lab_item_id_array)]

"""
filter through number of measurements
"""
sample_thr = 5
final_target_hadm = []
for i, one_hadm_id in enumerate(target_hadm):
    # if(i % 1000 == 0):
    #     print(i)
    one_chart_df = cohort_chart_df[cohort_chart_df['HADM_ID'] == one_hadm_id]
    one_lab_df = cohort_lab_df[cohort_lab_df['HADM_ID'] == one_hadm_id]
    vital_flag = True
    lab_flag = True
    for j, (item_index, item_name, item_id) in enumerate(vital_item):
        item_df = one_chart_df[one_chart_df['ITEMID'] == item_id]
        item_val = item_df['VALUENUM'].values
        item_val = item_val[~np.isnan(item_val)]
        item_val = item_val[np.where((item_val > vital_bound_list[j][0]) & (item_val <= vital_bound_list[j][1]))]
        if(len(item_val) < sample_thr):
            vital_flag = False
            break
            
    for j, (item_index, item_name, item_id) in enumerate(lab_item):
        item_df = one_lab_df[one_lab_df['ITEMID'] == item_id]
        item_val = item_df['VALUENUM'].values
        item_val = item_val[~np.isnan(item_val)]
        item_val = item_val[np.where(item_val > 0.)]
        if(len(item_val) < sample_thr):
            lab_flag = False
            break
    if(vital_flag & lab_flag):
        final_target_hadm.append(one_hadm_id)
print(len(final_target_hadm))

cohort_chart_df = chart_df[chart_df['HADM_ID'].isin(final_target_hadm)]
cohort_chart_df = cohort_chart_df[cohort_chart_df['ITEMID'].isin(chart_item_id_array)]

cohort_lab_df = lab_df[lab_df['HADM_ID'].isin(final_target_hadm)]
cohort_lab_df = cohort_lab_df[cohort_lab_df['ITEMID'].isin(lab_item_id_array)]

np.save(os.path.join(output_dir, 'cohort_hadm'), final_target_hadm)
print(len(final_target_hadm), min(final_target_hadm), max(final_target_hadm))

"""
plot the distribution of each picked covariates
"""
pop_vital_stat = []
for i, (item_index, item_name, item_id) in enumerate(vital_item):
    print(item_name)
    item_df = cohort_chart_df[cohort_chart_df['ITEMID'] == item_id]
    item_val = item_df['VALUENUM'].values
    item_val = item_val[~np.isnan(item_val)]
    
    plt.figure()
    sns.distplot(item_val, kde=False, color='b', label='Before QC')

    item_val = item_val[np.where((item_val > vital_bound_list[i][0]) & (item_val <= vital_bound_list[i][1]))]
    print(max(item_val), min(item_val), np.nanmean(item_val), np.nanstd(item_val))
    pop_vital_stat.append((np.nanmean(item_val), np.nanstd(item_val)))
    
    plt.figure()
    sns.distplot(item_val, kde=False, color='g', label='After QC')
    
    plt.title(item_name)
    plt.savefig(os.path.join(output_dir, '{}_dist.{}'.format(item_name, global_fig_format)), format=global_fig_format)
    plt.close()
pickle.dump(pop_vital_stat, open(os.path.join(output_dir, 'pop_vital_stat.p'), 'wb'))

# output population mean and standard deviation
for i, (item_index, item_name, item_id) in enumerate(vital_item):
    f = open(os.path.join(output_dir, 'feature{}_stat.bin'.format(item_index)), 'wb')
    float_array = array('d', [pop_vital_stat[i][0], pop_vital_stat[i][1]])
    float_array.tofile(f)
    f.close()

lab_bound_list = []
pop_lab_stat = []
for i, (item_index, item_name, item_id) in enumerate(lab_item):
    print(item_name)
    item_df = cohort_lab_df[cohort_lab_df['ITEMID'] == item_id]
    item_val = item_df['VALUENUM'].values
    item_val = item_val[~np.isnan(item_val)]
    
    plt.figure()
    sns.distplot(item_val, kde=False, color='b', label='Before QC')

    item_val = item_val[np.where(item_val > 0)]
    pop_lab_stat.append((np.nanmean(item_val), np.nanstd(item_val)))
    sns.distplot(item_val, kde=False, color='g', label='After QC')
    
    plt.title(item_name)
    plt.savefig(os.path.join(output_dir, '{}_dist.{}'.format(item_name, global_fig_format)), format=global_fig_format)
    plt.close()

pickle.dump(pop_lab_stat, open(os.path.join(output_dir, 'pop_lab_stat.p'), 'wb'))

# output population mean and standard deviation
for i, (item_index, item_name, item_id) in enumerate(lab_item):
    f = open(os.path.join(output_dir, 'feature{}_stat.bin'.format(item_index)), 'wb')
    float_array = array('d', [pop_lab_stat[i][0], pop_lab_stat[i][1]])
    float_array.tofile(f)
    f.close()
# print(pop_vital_stat)
# print(pop_lab_stat)


"""
do quality control of each covariate output data for C++ processing
"""
cohort_chart_df.to_hdf(os.path.join(output_dir, 'chart_df_before_qc.h5'.format(cohort)), 'data')
cohort_lab_df.to_hdf(os.path.join(output_dir, 'lab_df_before_qc.h5'.format(cohort)), 'data')

t0 = time.time()
qc_remove_hadm = []
sample_per_hadm = np.zeros(len(final_target_hadm))
for hidx, hadm in enumerate(final_target_hadm):
    hadm_dir = os.path.join(output_dir, 'hadm_{}/'.format(hadm))
    # print(hadm_dir)
    if not os.path.exists(hadm_dir):
        os.makedirs(hadm_dir)
        
    ref_time = pd.to_datetime(adm_df[adm_df['HADM_ID'] == hadm]['ADMITTIME']).values[0]
    hadm_sample_num = 0
    for i, (item_index, item_name, item_id) in enumerate(vital_item):
        item_df = cohort_chart_df[(cohort_chart_df['HADM_ID'] == hadm) & (cohort_chart_df['ITEMID'] == item_id)].copy()
        item_df['CHARTTIME'] = pd.to_datetime(item_df['CHARTTIME'])
        item_df = item_df.sort_values(by='CHARTTIME')
        
        item_time = (item_df['CHARTTIME'].values - ref_time)/(10**9)
        item_time = item_time.astype(np.float32)/3600.
        item_value = item_df['VALUENUM'].values.astype(np.float32)
        
        qc_time, qc_value = do_qc(item_time, item_value, 
                                  lb=vital_bound_list[i][0], 
                                  ub=vital_bound_list[i][1], 
                                  min_period=None,
                                  name=item_name
                                  )
        assert len(qc_time) == len(qc_value)
        hadm_sample_num += len(qc_time)

        if((len(qc_time) < sample_thr)):
            print('hadm {}: {} too few # of measurements: {}'.format(hadm, item_name, len(qc_time)))
            if(hadm not in qc_remove_hadm):
                qc_remove_hadm.append(hadm)
        if(len(np.where(qc_value == 0.)[0]) > 0):
            print('hadm {}: {} found zero value #: {}'.format(hadm, item_name, len(np.where(qc_value == 0.)[0])))
        if(len(np.where(qc_time < 0.)[0]) > 0):
            print('hadm {}: {} found negative timestamp #: {}'.format(hadm, item_name, len(np.where(qc_time < 0.)[0])))
            print(qc_time[np.where(qc_time < 0.)[0]])
        data_vec = np.vstack((qc_time, qc_value)).T.reshape(-1)
        data_vec = np.hstack(([len(qc_time)], data_vec))
        np.savetxt(os.path.join(hadm_dir, 'feature{}.txt'.format(item_index)), data_vec, delimiter='\n', fmt='%6.6f')
        
    for i, (item_index, item_name, item_id) in enumerate(lab_item):
        item_df = cohort_lab_df[(cohort_lab_df['HADM_ID'] == hadm) & (cohort_lab_df['ITEMID'] == item_id)].copy()
        item_df['CHARTTIME'] = pd.to_datetime(item_df['CHARTTIME'])
        item_df = item_df.sort_values(by='CHARTTIME')
        
        item_time = (item_df['CHARTTIME'].values - ref_time)/(10**9)
        item_time = item_time.astype(np.float32)/3600.
        item_value = item_df['VALUENUM'].values.astype(np.float32)

        # only clean up zero and negative values
        qc_time, qc_value = do_qc(item_time, item_value, 
                                  lb=0., 
                                  ub=None, 
                                  min_period=None,
                                  name=item_name
                                  )
        assert len(qc_time) == len(qc_value)
        hadm_sample_num += len(qc_time)

        if((len(qc_time) < sample_thr)):
            print('hadm {}: {} too few # of measurements: {}'.format(hadm, item_name, len(qc_time)))
            if(hadm not in qc_remove_hadm):
                qc_remove_hadm.append(hadm)
        if(len(np.where(qc_value == 0.)[0]) > 0):
            print('hadm {}: {} found zero value #: {}'.format(hadm, item_name, len(np.where(qc_value == 0.)[0])))
        if(len(np.where(qc_time < 0.)[0]) > 0):
            print('hadm {}: {} found negative timestamp #: {}'.format(hadm, item_name, len(np.where(qc_time < 0.)[0])))
            print(qc_time[np.where(qc_time < 0.)[0]])
        data_vec = np.vstack((qc_time, qc_value)).T.reshape(-1)
        data_vec = np.hstack(([len(qc_time)], data_vec))
        np.savetxt(os.path.join(hadm_dir, 'feature{}.txt'.format(item_index)), data_vec, delimiter='\n', fmt='%6.6f')
    sample_per_hadm[hidx] = hadm_sample_num
et = time.time()
print('Finish processing all patients; elapsed time: {} seconds'.format(et-t0))

"""
filter the patients match criteria after qc
"""
final_exp_hadm = []
max_sample_num = None
min_sample_num = None
for hidx, hadm in enumerate(final_target_hadm):
    if(hadm not in qc_remove_hadm):
        final_exp_hadm.append('hadm_{}'.format(hadm))
        max_sample_num = sample_per_hadm[hidx] if(max_sample_num is None) else(max(max_sample_num, sample_per_hadm[hidx]))
        min_sample_num = sample_per_hadm[hidx] if(min_sample_num is None) else(min(min_sample_num, sample_per_hadm[hidx]))
print('max sample # {}'.format(max_sample_num))
print('min sample # {}'.format(min_sample_num))

final_exp_hadm = np.asarray(final_exp_hadm)
np.save(os.path.join(output_dir, 'cohort_hadm_match'), final_exp_hadm)
np.savetxt(os.path.join(output_dir, 'cohort_hadm_match.txt'), final_exp_hadm, delimiter='\n', fmt='%s')
print(len(final_exp_hadm), min(final_exp_hadm), max(final_exp_hadm))

