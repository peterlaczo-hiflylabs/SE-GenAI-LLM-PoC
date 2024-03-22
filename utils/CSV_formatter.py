import pandas as pd
import numpy as np
from collections import Counter

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

def check_header(gen_csv: pd.DataFrame, doc_type):
    match doc_type:
        case "anam":
            doc_header = ["Diagnózis", "Kezdete", "BNO-10", "Forrás(ok)"]
        case "gyogyszer":
            doc_header = ["Gyógyszerallergia", "Kezdete", "Forrás(ok)"]
    
    if Counter(gen_csv.columns) != Counter(doc_header):
        new_row = gen_csv.columns
        gen_csv = pd.concat([pd.DataFrame(new_row, index=gen_csv.columns).transpose(),gen_csv], ignore_index=True)
        gen_csv = gen_csv.rename({gen_csv.columns[i]: doc_header[i] for i in range(len(gen_csv.columns))}, axis=1)
    return gen_csv

def format_date(x):
    if isinstance(x,float):
        return '{:.0f}'.format(x) if x.is_integer() else '{:.2f}'.format(x)
    return x

def format_anamnezis_csv(gen_csv: pd.DataFrame):
    bno_table = pd.read_excel('BNOTORZS.xlsx')
    gen_csv['BNO-10']= gen_csv['BNO-10'].apply(lambda x: "" if pd.isna(x) else x)
    gen_csv = check_header(gen_csv,"anam")
    gen_csv['BNO leírás'] = gen_csv['BNO-10'].apply(lambda x: insert_bno_description(x,bno_table))
    breakpoint()
    gen_csv['Kezdete'] = gen_csv['Kezdete'].apply(lambda x: format_date(x))
    gen_csv['Bejegyzés dátuma'] = gen_csv['Forrás(ok)'].apply(lambda x: x.split('_')[1] if isinstance(x, float) == False and len(x.split('_')) > 1 else '')
    gen_csv['Rendezési dátum'] = gen_csv.apply(lambda x: x['Bejegyzés dátuma'] if str(x['Kezdete']) == 'nan' else x['Kezdete'],axis= 1)
    gen_csv = gen_csv.reindex(columns=[gen_csv.columns[0],gen_csv.columns[1],gen_csv.columns[5],gen_csv.columns[2],gen_csv.columns[4],gen_csv.columns[3],gen_csv.columns[6]])
    gen_csv = gen_csv.drop_duplicates(subset='Diagnózis',keep='first').sort_values("Rendezési dátum").reset_index(drop=True)
    return gen_csv

def format_gyogyszer_csv(gen_csv: pd.DataFrame):
    gen_csv = check_header(gen_csv,"gyogyszer")
    gen_csv = gen_csv.drop_duplicates(subset='Gyógyszerallergia',keep='first').sort_values("Kezdete").reset_index(drop=True)
    gen_csv = gen_csv[(gen_csv['Gyógyszerallergia'] != "Gyógyszerallergia") & (gen_csv['Gyógyszerallergia'] != "Gyógyszerérzékenység")]
    gen_csv['Bejegyzés dátuma'] = gen_csv['Forrás(ok)'].apply(lambda x: x.split('_')[1] if isinstance(x, float) == False and len(x.split('_')) > 1 else '')
    gen_csv = gen_csv.reindex(columns=[gen_csv.columns[0],gen_csv.columns[1],gen_csv.columns[3],gen_csv.columns[2]])
    return gen_csv

def create_extended_log(gen_csv: pd.DataFrame, feedback_datas: pd.DataFrame):
    if feedback_datas['sor'].values[0] == "-" or feedback_datas['sor'].values[0] == "":
        extended_row = feedback_datas
    else:
        subject_row = pd.DataFrame(gen_csv.iloc[int(feedback_datas['sor']) - 1]).transpose()
        extended_row = pd.concat([feedback_datas, subject_row], axis=1)

    return extended_row