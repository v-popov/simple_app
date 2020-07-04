import streamlit as st
import numpy as np
import pandas as pd


st.title('My First App')

st.write('Here is a table:')

st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

