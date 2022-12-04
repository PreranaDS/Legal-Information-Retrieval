import pandas as pd
from legalinforetrieval import retrieve
res=retrieve('arms act')
res=res[res['court']=='SC']
print(res['case name'])
