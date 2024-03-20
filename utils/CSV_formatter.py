import pandas as pd
import numpy as np


def insert_bno_description(gen_bno, bno_table):
    formatted_gen_bno = ""
    if len(str(gen_bno).split('-')) > 1:
        pass
    else:
        for element in str(gen_bno).split('.'):
            formatted_gen_bno+=element
        for element in range(5-len(formatted_gen_bno)):
            formatted_gen_bno+='0'
    result_row = bno_table[(bno_table['KOD10'] == formatted_gen_bno) | (bno_table['KOD10'] == formatted_gen_bno[:-2] + 'H0')]
    if not result_row.empty:
        return result_row['NEV'].values[0]
    return ''

def format_diagnosis_csv(gen_csv: pd.DataFrame):
    bno_table = pd.read_excel('BNOTORZS.xlsx')
    gen_csv['BNO-10']= gen_csv['BNO-10'].apply(lambda x: "" if pd.isna(x) else x)
    gen_csv['BNO leírás'] = gen_csv['BNO-10'].apply(lambda x: insert_bno_description(x,bno_table))
    gen_csv['Bejegyzés dátuma'] = gen_csv.apply(lambda x: x['Forrás(ok)'].split('_')[1],axis= 1)
    gen_csv = gen_csv.reindex(columns=[gen_csv.columns[0],gen_csv.columns[1],gen_csv.columns[5],gen_csv.columns[2],gen_csv.columns[4],gen_csv.columns[3]])
    gen_csv = gen_csv.drop_duplicates(subset='Diagnózis',keep='first').sort_values("Bejegyzés dátuma").reset_index(drop=True)
    return gen_csv

def create_extended_log(gen_csv: pd.DataFrame, feedback_datas: pd.DataFrame):
    if feedback_datas['sor'].values[0] == "-" or feedback_datas['sor'].values[0] == "":
        extended_row = feedback_datas
    else:
        subject_row = pd.DataFrame(gen_csv.iloc[int(feedback_datas['sor']) - 1]).transpose()
        extended_row = pd.concat([feedback_datas, subject_row], axis=1)

    return extended_row