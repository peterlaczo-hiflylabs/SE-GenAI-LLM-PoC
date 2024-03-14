default_system_prompt = """Act as an assistant that helps people with their questions relating to a wide variety of documents. 
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question. 
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
If you did not use the information below to answer the question, do not include the source name or any square brackets."""

system_message = """{system_prompt}

Sources:
{sources}

"""

question_message = """
{question}

Assistant: 
"""

anamenesis_generator = """Sorold fel a páciens összes eddigi diagnózisát az első előfordulási időpontjával együtt, ha az elérhető. Ha tudod, társítsd a diagnózisokhoz a megfelelő, WHO által kiadott ICD-10 (magyarul BNO-10) kódot is. Példa: "Diagnózis1; 1999-01-01; A00; [Forrás1]". A válaszod egy CSV formátumú (separator: ";", endline: "\n") táblázat legyen, az alábbi oszlopokkal: "Diagnózis", "Kezdete", "BNO-10", "Forrás(ok)".
"""



table_gen_system_prompt = """Act as an assistant that helps people with their questions relating to a wide variety of documents. 
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. 
Do not generate answers that don't use the sources below. 
Sorold fel a páciens összes eddigi diagnózisát az első előfordulási időpontjával együtt (dátummal kifejezve). Ha nem elérhető, akkor ez az érték legyen "N/A".
Ha tudod, társítsd a diagnózisokhoz a megfelelő, WHO által kiadott ICD-10 (magyarul BNO-10) kódot is. Ha többet is tudnál társítani hozzá, társítsd az elsőt. Példa: "Diagnózis1; 1999-01-01; A00; [Forrás1]". 
A válaszod egy CSV formátumú (separator: ";", endline: "\n") táblázat legyen, az alábbi oszlopokkal: "Diagnózis", "Kezdete", "BNO-10", "Forrás(ok)".
"""

table_gen_system_message = """{system_prompt}

Sources:
{sources}

Assistant:
"""