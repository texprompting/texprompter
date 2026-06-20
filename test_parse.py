import json

val = '"## Production Planning MILP Formulation\\n\\n### Parameters\\n- \\n"'
print("Original:", val)

def clean_doc(doc_str):
    if doc_str.startswith('"') and doc_str.endswith('"'):
        try:
            return json.loads(doc_str)
        except:
            return doc_str.strip('"').replace('\\n', '\n')
    return doc_str

print("Cleaned:\n", clean_doc(val))
