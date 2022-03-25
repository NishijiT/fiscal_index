import numpy as np
import pandas as pd
import streamlit as st

st.title('都道府県財政指標ダッシュボード')

df = pd.read_csv('fiscal_index_pref.csv', encoding='shift_jis')
#print(df.head())

df_jp_mean = df[df['都道府県名']=='全国']
df_osaka = df[df['都道府県名']=='大阪府'] 
df_jp_mean = df_jp_mean.set_index('集計年')
df_osaka = df_osaka.set_index('集計年')


df_pref = df.set_index('集計年')

#0 財政力指数
st.subheader('財政力指数')
df0 = df_pref[['都道府県名','財政力指数']]
df_line0 = pd.DataFrame(index=[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
df_line0['全国'] = df_jp_mean['財政力指数']
df_line0['大阪府'] = df_osaka['財政力指数']
pref_list0 = df['都道府県名'].unique()
option_pref0 = st.selectbox('都道府県:財政力指数', (pref_list0))
df0 = df0[df0['都道府県名'] == option_pref0]
df_line0[option_pref0] = df0['財政力指数']

st.line_chart(df_line0)
show_df = st.checkbox('財政力指数')
if show_df == True:
       st.write(df_line0)

#1 経常収支比率
st.subheader('経常収支比率')
df1 = df_pref[['都道府県名','経常収支比率']]
df_line1 = pd.DataFrame(index=[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
df_line1['全国'] = df_jp_mean['経常収支比率']
df_line1['大阪府'] = df_osaka['経常収支比率']
pref_list1 = df['都道府県名'].unique()
option_pref1 = st.selectbox('都道府県:経常収支比率', (pref_list1))
df1 = df1[df1['都道府県名'] == option_pref1]
df_line1[option_pref1] = df1['経常収支比率']

st.line_chart(df_line1)
show_df = st.checkbox('経常収支比率')
if show_df == True:
       st.write(df_line1)

#2 実質公債費比率
st.subheader('実質公債費比率')
df2 = df_pref[['都道府県名','実質公債費比率']]
df_line2 = pd.DataFrame(index=[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
df_line2['全国'] = df_jp_mean['実質公債費比率']
df_line2['大阪府'] = df_osaka['実質公債費比率']
pref_list2 = df['都道府県名'].unique()
option_pref2 = st.selectbox('都道府県:実質公債費比率', (pref_list2))
df2 = df2[df2['都道府県名'] == option_pref2]
df_line2[option_pref2] = df2['実質公債費比率']

st.line_chart(df_line2)
show_df = st.checkbox('実質公債費比率')
if show_df == True:
       st.write(df_line2)

#3 将来負担比率
st.subheader('将来負担比率')
df3 = df_pref[['都道府県名','将来負担比率']]
df_line3 = pd.DataFrame(index=[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
df_line3['全国'] = df_jp_mean['将来負担比率']
df_line3['大阪府'] = df_osaka['将来負担比率']
pref_list3 = df['都道府県名'].unique()
option_pref3 = st.selectbox('都道府県:将来負担比率', (pref_list3))
df3 = df3[df3['都道府県名'] == option_pref3]
df_line3[option_pref3] = df3['将来負担比率']

st.line_chart(df_line3)
show_df = st.checkbox('将来負担比率')
if show_df == True:
       st.write(df_line3)

#4 ラスパイレス指数
st.subheader('ラスパイレス指数')
df4 = df_pref[['都道府県名','ラスパイレス指数']]
df_line4 = pd.DataFrame(index=[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
df_line4['全国'] = df_jp_mean['ラスパイレス指数']
df_line4['大阪府'] = df_osaka['ラスパイレス指数']
pref_list4 = df['都道府県名'].unique()
option_pref4 = st.selectbox('都道府県:ラスパイレス指数', (pref_list4))
df4 = df4[df4['都道府県名'] == option_pref4]
df_line4[option_pref4] = df4['ラスパイレス指数']

st.line_chart(df_line4)
show_df = st.checkbox('ラスパイレス指数')
if show_df == True:
       st.write(df_line4)




