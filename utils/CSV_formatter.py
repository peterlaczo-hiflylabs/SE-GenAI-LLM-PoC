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

#Temporarily disposed of
# def split_listed_bnos(table: pd.DataFrame):
#     final_table = table.copy()
#     shift_counter = 0
#     for idx, row in table.iterrows():
#         bno_ids = row['BNO-10'].split('-')
#         if len(bno_ids) > 1:
#             for i in range(len(bno_ids)):
#                 final_table.iloc[idx + shift_counter]['BNO-10'] = f"{bno_ids[0]} ({row['BNO-10']})"
#                 new_row = row.copy()
#                 new_row['BNO-10'] = f"{bno_ids[i]} ({row['BNO-10']})"
#                 previous_part = final_table.iloc[:idx + shift_counter + 1]
#                 later_part = final_table.iloc[idx + shift_counter +1:]
#                 final_table = pd.concat([previous_part,new_row.to_frame(),later_part], ignore_index=True)
#                 shift_counter += 1
#             # row['BNO-10'] = f"{bno_ids[0]} ({row['BNO-10']})"
#     return final_table

def format_diagnosis_csv(gen_csv: pd.DataFrame):
    bno_table = pd.read_excel('BNOTORZS.xlsx')
    # gen_csv = split_listed_bnos(gen_csv)
    # gen_csv['Kezdete'] = gen_csv.apply(lambda x: x['Forrás(ok)'].split('_')[1] if str(x['Kezdete']) == 'nan' else x['Kezdete'],axis= 1)
    gen_csv['BNO-10']= gen_csv['BNO-10'].apply(lambda x: "" if pd.isna(x) else x)
    gen_csv['BNO leírás'] = gen_csv['BNO-10'].apply(lambda x: insert_bno_description(x,bno_table))
    return gen_csv