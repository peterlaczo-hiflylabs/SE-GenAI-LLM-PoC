import pandas as pd
import numpy as np
import os
from pathlib import Path
from utils.blob_storage_handlers import upload_to_blob_storage

def format_table(df_raw, blob_storage, container_name):

    df_raw['ADMIT_DATE'] = df_raw['ADMIT_DATE'].dt.date

    df_raw = df_raw.rename(columns={
        'Ananmézis': 'Anamnézis',
        'Therápia': 'Terápia'
    })
    df_melt = df_raw.melt(
        id_vars=['NPI', 'CASE_NO', 'ADMIT_DATE', 'CASE_TYPE', 'DEPT', 'DESCR', 'TX_TYPE', 'VER_NO', 'SEQ_NO'],
        value_vars=['Anamnézis', 'Jelen panaszok', 'Dekurzus', 'Epikrízis', 'Egyéb vizsgálatok', 'Műtéti leírás', 'Státusz', 'Javaslat', 'Terápia'],
        var_name='TYPE',
        value_name='VALUE'
        ).dropna(subset='VALUE')
    df_melt.insert(
        df_melt.columns.get_loc('TYPE'),
        'TYPE_ORDER',
        df_melt['TYPE'].replace(
            {
                'Anamnézis': 1,
                'Jelen panaszok': 2,
                'Dekurzus': 3,
                'Epikrízis': 4,
                'Egyéb vizsgálatok': 5,
                'Műtéti leírás': 6,
                'Státusz': 7,
                'Javaslat': 8,
                'Terápia': 9
            }
        ).convert_dtypes()
    )

    # Replace multiple whitespace characters in VALUE column
    df_melt['VALUE'] = df_melt['VALUE'].str.replace(r' +', ' ', regex=True)
    df_melt['VALUE'] = df_melt['VALUE'].str.replace(r'[\n\t\r\f]+', '\n', regex=True)

    # Strip VALUE (and interpret empty string '' as NA)
    df_melt['VALUE'] = df_melt['VALUE'].str.strip().replace('', pd.NA)

    #
    df_melt['TX_TYPE'] = df_melt['TX_TYPE'].str.upper()

    # Add TYPE to TEXT
    df_melt['TEXT'] = df_melt['TYPE'].str.upper() + ':\n' + df_melt['VALUE'].str.replace('\n', '\n\t')

    # TEXT2
    df_melt['TEXT2'] = df_melt['TYPE'].str.upper() + ' (' + df_melt['ADMIT_DATE'].astype(str) + '):\n' + df_melt['VALUE'].str.replace('\n', '\n\t')

    # Drop NA, convert dtypes, and sort melt dataframe
    df_melt = df_melt.dropna().convert_dtypes().sort_values(['NPI', 'ADMIT_DATE', 'CASE_NO', 'TYPE_ORDER'], ignore_index=True)

    df_merged = df_melt.groupby(['NPI','ADMIT_DATE','CASE_NO','CASE_TYPE','DEPT','DESCR'])[['TEXT']].agg("\n\n".join).reset_index(level=-1)

    for i in df_merged.index:
        # Get metadata
        npi = i[0]
        admit_date = i[1]
        case_no = i[2]
        case_type = i[3]
        dept = i[4]
        fname = f'{npi}_{admit_date}_{case_no}_{case_type}_{dept}.txt'
        department = df_merged.loc[i, 'DESCR']
        text = df_merged.loc[i, 'TEXT']
        header = f'ZÁRÓJELENTÉS\nPáciens: {npi} | Rögzítési dátum: {admit_date} | Dokumentum: {case_no} | Osztály: {department} ({dept})\n\n'
        file_content = "\n".join([header, text])
        upload_to_blob_storage(blob_storage,container_name, f"{npi}/src/all/{fname}", file_content)
    
    df_anm_all = df_melt[df_melt['TX_TYPE'] == 'ANM'].groupby(['NPI'])[['VALUE']].agg("\n\n".join)

    #creating merged folder
    for i in df_anm_all.index:
        # Get data
        file_content = df_anm_all.loc[i, 'VALUE']
        fname = f'{i}_ANM_MERGED.txt'
        upload_to_blob_storage(blob_storage, container_name, f"{i}/src/merged/{fname}", file_content)

    df_epd_all = df_melt[df_melt['TX_TYPE'] == 'EPD'].groupby(['NPI'])[['VALUE']].agg("\n\n".join)

    for i in df_epd_all.index:
        # Get data
        file_content = df_epd_all.loc[i, 'VALUE']
        fname = f'{i}_EPD_MERGED.txt'
        upload_to_blob_storage(blob_storage, container_name, f"{i}/src/merged/{fname}", file_content)

    df_melt['FINAL_TEXT'] = df_melt.apply(lambda row: row['TEXT'] if row['TX_TYPE'] == 'ANM' else row['TEXT2'], axis=1)

    # Merge ANM and EPD text only
    df_merged_filtered = df_melt[df_melt['TX_TYPE'].isin(('ANM', 'EPD'))].groupby(['NPI','ADMIT_DATE','CASE_NO','CASE_TYPE','DEPT','DESCR'])[['FINAL_TEXT']].agg("\n\n".join).reset_index(level=-1)

    #creating filtered folder
    for i in df_merged_filtered.index:
        # Get metadata
        npi = i[0]
        admit_date = i[1]
        case_no = i[2]
        case_type = i[3]
        dept = i[4]
        fname = f'{npi}_{admit_date}_{case_no}_{case_type}_{dept}_filtered.txt'
        file_content = df_merged_filtered.loc[i, 'FINAL_TEXT']
        upload_to_blob_storage(blob_storage,container_name, f"{npi}/src/filtered/{fname}", file_content)