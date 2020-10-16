# Prediction of massive transfusion
import os
import numpy as np
import pandas as pd
os.chdir('D:\\USB_2020_09_20\\DATATHON\\Transfusion\\data\\Final data')
df=pd.read_excel('cases_20200929.xlsx')  # In cases file, creat new column with fileid2 'A1_171211_075736' from fileid'A1_171211_075736.vital'
df.head()


variablelist=['fileid', 'fileid2', 'opid', 'caseid', 'opname', 'sex', 'age', 'em_yn', 'weight',
              'height', 'dept', 'orin', 'orout', 'opstart', 'opend', 'admission_date', 
              'discharge_date', 'death_time', 'aneinfo_position', 'aneinfo_anetype',
              'aneinfo_em', 'preop_hb', 'preop_hct', 'preop_plt', 'preop_bun', 'preop_cr', 
              'preop_pt', 'preop_ptt', 'preop_alb', 'preop_got', 'preop_gpt', 'preop_na', 
              'preop_k', 'preop_tnt', 'preop_tni', 'preop_sat', 'preop_glu', 'preop_be', 
              'preop_pao2', 'preop_paco2', 'preop_hco3', 'preop_ph', 'postop_cr_peak1', 
              'postop_cr_peak2', 'postop_cr_peak3', 'postop_cr_peak7', 'postop_tnt_peak30', 
              'postop_tni_peak30', 'premedi_htn', 'premedi_dm', 'premedi_tb', 
              'premedi_liver_dz', 'premedi_copd', 'premedi_asthma', 'premedi_heart_dz', 
              'premedi_thyroid_dz', 'premedi_renal_dz', 'premedi_hematologic_dz', 
              'premedi_vascular_dz', 'premedi_neurologic_dz', 'premedi_other_dz',
              'premedi_pregnancy', 'premedi_obesity', 'premedi_smoking', 'premedi_smoking_dose',
              'premed_em', 'premedi_asa', 'nurse_htn', 'nurse_dm', 'nurse_tb', 'nurse_liver_dz',
              'nurse_other_dz', 'sum_uo', 'sum_ebl', 'sum_crystalloid', 'sum_colloid', 'sum_fluid',
              'sum_rbc', 'sum_ffp', 'sum_plt', 'sum_cryo', 'preop_egfr']
variablelist2=['fileid', 'fileid2', 'opid', 'caseid', 'opname', 'sex', 'age', 'em_yn', 'weight',
              'height', 'dept', 'orin', 'opstart', 'admission_date', 
              'aneinfo_position', 'aneinfo_anetype',
              'aneinfo_em', 'preop_hb', 'preop_hct', 'preop_plt', 'preop_bun', 'preop_cr', 
              'preop_pt', 'preop_ptt', 'preop_alb', 'preop_got', 'preop_gpt', 'preop_na', 
              'preop_k', 'preop_tnt', 'preop_tni', 'preop_sat', 'preop_glu', 'preop_be', 
              'preop_pao2', 'preop_paco2', 'preop_hco3', 'preop_ph',
              'premedi_htn', 'premedi_dm', 'premedi_tb', 
              'premedi_liver_dz', 'premedi_copd', 'premedi_asthma', 'premedi_heart_dz', 
              'premedi_thyroid_dz', 'premedi_renal_dz', 'premedi_hematologic_dz', 
              'premedi_vascular_dz', 'premedi_neurologic_dz', 'premedi_other_dz',
              'premedi_pregnancy', 'premedi_obesity', 'premedi_smoking', 'premedi_smoking_dose',
              'premed_em', 'premedi_asa', 'preop_egfr']
 # include only preoperative factors
df=df[variablelist2]

# Only cases who have ART/ecg track (intersection of ‘cases.xlsx’ and ‘vataldb_trks_art_ecg.xlsx’)
track=pd.read_csv('vitaldb_trks_art_ecg_20200929.csv')
print(track.shape)
track2=track.drop_duplicates(['fileid'], keep='first') # drop dulicate cases
print(track2.shape)

track2.to_excel('vitaldb_trks_art_ecg_dropduplicate.xlsx')

df2=pd.merge(df, track2, how='right') 
print(df2.shape) # N=25,926


# massive transfusion
massive=pd.read_csv('mass-trans-phenotype-20200929.csv') 
    # change the column name from 'fileid' to 'fileid2' 
    # create new column with 'masstf=1/ 0.5' (1: massive transf, 0.5: transf but not massive [grey zone])
massive.head()
print(massive.shape) # N=2559 (who underwent any transfusion (either massive or not massive))
massive.masstf.value_counts()

population=pd.merge(df2, massive, how='left')
population['masstf']=population['masstf'].fillna(0)
# 0: no transfusion, 1: massive transfusion, 0.5: transfusion but not massive (grey zone)
print(population.shape)

# unknown drop
unknown=pd.read_csv('unknown-patients-20200929.csv') # change the column name from 'fileid' to 'fileid2'
unknown.head()
print(unknown.shape) # N=47
isin_filter=population['fileid2'].isin(unknown['fileid2'])
isin_filter
d=population[isin_filter].index
d
population2=population.drop (d, axis=0)
print(population2.shape)  # N=25,880
population2.masstf.value_counts()  
population2.head()
# 0: no transfusion (N=23,334), 1: massive transfusion (N=582), 0.5: transfusion but not massive (grey zone, N=1,964)

duplication=population2.duplicated(['fileid'], keep=False) # Is there duplicated case?
len(population2[duplication]) # N=2 

population3=population2.drop_duplicates(['fileid'], keep=False) # drop dulicate cases
print(population3.shape) # N=25,878
population3.to_excel('finalpopulation.xlsx')
