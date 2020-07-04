import streamlit as st
import numpy as np
import pandas as pd


class UIClass:

	def __init__(self):
		self.data_selection_msg_ph = st.subheader('Step 1/N: Select Data')
		self.data_selection_ph = st.empty()
		self.data_upload_ph = st.empty()
		self.data_preview_msg_ph = st.empty()
		self.data_preview_ph = st.empty()
		self.values_column_selector_ph = st.empty()
		self.product_id_column_selector_ph = st.empty()
		self.values_column = None
		self.product_id_column = None
		self.input_df = None

	@staticmethod
	def try_read_df(f):
		try:
			return pd.read_csv(f)
		except:
			return pd.read_excel(f)

	def load_data(self, filename):
		self.input_df = self.try_read_df(filename)
		self.data_preview_msg_ph.subheader('Data Preview:')
		self.data_preview_ph.dataframe(uiclass.input_df.head(5))

	def select_values_col(self):
		self.values_column = self.values_column_selector_ph.selectbox(label='Select column name containing demand values',
																	  options=['Not Selected'] + self.input_df.columns.tolist())
		try:
			self.input_df[self.values_column] = self.input_df[self.values_column].astype(np.int64)
			self.select_product_id_col()
		except:
			self.product_id_column_selector_ph.markdown('**ERROR: Values should be numerical. Please, select another column containing demand values**')

	def select_product_id_col(self):
		if self.product_id_column != 'Not Selected':
			columns = set(self.input_df.columns)
			columns.remove(self.values_column)
			self.product_id_column = self.product_id_column_selector_ph.selectbox(label='Select column name containing product ID',
																				  options=['Not Selected'] + list(columns))
		else:
			pass #add some placeholder for info msg; potentially next useful one


if __name__ == '__main__':
	st.title('Demand Forecasting App')
	st.write('Here you will be able to predict the future demand for every product you have.')

	uiclass = UIClass()
	data_source_selector = uiclass.data_selection_ph.selectbox(label='', options=('Use Example Data', 'Upload Data'))
	if data_source_selector == 'Upload Data':
		uploaded_file = uiclass.data_upload_ph.file_uploader("Choose a CSV or XLSX file (max file size is 200Mb).", type=["csv", "xlsx"])
		if uploaded_file is not None:
			uiclass.load_data(uploaded_file)
			uiclass.select_values_col()
	else:
		uiclass.load_data('default_table.csv')
		uiclass.select_values_col()


