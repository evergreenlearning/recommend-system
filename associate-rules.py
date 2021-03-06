#!-*- coding:utf-8 -*-
#/usr/bin/python

from collections import Counter
import pandas as pd
def association_rules(N):
    data = pd.read_csv("./BreadBasket_DMS.csv")
    transaction_n_tup = sorted(Counter(data.Transaction).items(), key=lambda x:x[1], reverse=True)
    transaction_s_tup = sorted(Counter(data.Transaction).items(), key=lambda x:x[1], reverse=False)
    transaction_n_dic = dict((x,y) for x,y in transaction_n_tup)
    transaction_s_dic = dict((x,y) for x,y in transaction_s_tup)
    
    ind_n = list(transaction_n_dic.values()).index(N)
    ind_s = list(transaction_s_dic.values()).index(N)
    lens = len(transaction_s_dic.values())
    
    N_Item = {}
    transaction_n_slice = {k:transaction_n_dic[k] for k in list(transaction_n_dic.keys())[ind_n:lens-ind_s]}
    for k,v in transaction_n_slice.items():
        Items = "-".join(sorted(list(data.loc[data.Transaction==k].Item)))
        N_Item.setdefault(Items, 0)
        N_Item[Items] += 1
    N_Item_tup = sorted(N_Item.items(), key=lambda x:x[1], reverse=True)
    N_Item_dic = dict((x,y) for x,y in N_Item_tup)
    return N_Item_dic