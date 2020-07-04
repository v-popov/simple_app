import streamlit as st
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.layouts import layout
from bokeh.models import Paragraph, PreText, Select


class UIClass:

	def __init__(self):
		#self.data_selection_msg_ph = st.subheader('Step 1/N: Select Data')
		self.data_selection_ph = st.empty()
		self.data_upload_ph = st.empty()
		self.data_preview_msg_ph = st.empty()
		self.data_preview_ph = st.empty()
		self.values_column_selector_ph = st.empty()
		self.product_id_column_selector_ph = st.empty()
		self.product_dashboard_selector = st.empty()
		self.dashboard_ph = st.empty()
		self.values_column = None
		self.product_id_column = None
		self.product_id_to_plot = None
		self.input_df = None

	@staticmethod
	def try_read_df(f):
		try:
			return pd.read_csv(f)
		except:
			return pd.read_excel(f)

	#@st.cache
	def load_data(self, filename):
		self.input_df = self.try_read_df(filename)
		self.data_preview_msg_ph.text('Data Preview:')#subheader
		self.data_preview_ph.dataframe(uiclass.input_df.head(5))
		self.values_column_selector_ph = st.empty()
		self.product_id_column_selector_ph = st.empty()
		self.product_dashboard_selector = st.empty()
		self.dashboard_ph = st.empty()

	#@st.cache
	def select_values_col(self):
		self.values_column = self.values_column_selector_ph.selectbox(label='Select column name containing demand values',
																	  options=['Not Selected'] + self.input_df.columns.tolist())
		try:
			self.input_df[self.values_column] = self.input_df[self.values_column].astype(np.int64)
			#self.select_product_id_col()
			return True
		except Exception as e:
			print(e)
			self.product_id_column_selector_ph.markdown('**ERROR: Values should be numerical. Please, select another column containing demand values**')
			self.product_dashboard_selector = st.empty()
			self.dashboard_ph = st.empty()
			return False

	#@st.cache
	def select_product_id_col(self):
		if self.product_id_column != 'Not Selected':
			columns = set(self.input_df.columns)
			columns.remove(self.values_column)
			self.product_id_column = self.product_id_column_selector_ph.selectbox(label='Select column name containing product ID',
																				  options=['Not Selected'] + list(columns))
			#st.write('self.product_id_column: {}'.format(self.product_id_column))
			if self.product_id_column != 'Not Selected':
				self.select_product_to_display()
		#else:
			#pass #add some placeholder for info msg; potentially next useful one

	def select_product_to_display(self):
		self.product_id_to_plot = self.product_dashboard_selector.selectbox(label='Select product ID to display',
																			options=self.input_df[
																				self.product_id_column].unique())
		self.plot_dashboard(self.product_id_to_plot)

	def plot_dashboard(self, filter_on_id=None):
		p = figure(title="Historical Demand", x_axis_label='Time', y_axis_label='Demand')
		if not filter_on_id:
			filter_on_id = self.input_df[self.product_id_column].unique()[0]
		sub_df = self.input_df[self.input_df[self.product_id_column] == filter_on_id]
		print(sub_df)
		print(np.arange(len(sub_df)))
		print(sub_df[self.values_column])
		p.line(np.arange(len(sub_df)), sub_df[self.values_column], legend_label="Temp.", line_width=2)
		self.dashboard_ph.bokeh_chart(p)


if __name__ == '__main__':
	st.title('Demand Forecasting App')
	st.write('Here you will be able to predict the future demand for every product you have.')

	uiclass = UIClass()
	data_source_selector = uiclass.data_selection_ph.selectbox(label='Select Data', options=('Use Example Data', 'Upload Data'))
	if data_source_selector == 'Upload Data':
		uploaded_file = uiclass.data_upload_ph.file_uploader("Choose a CSV or XLSX file (max file size is 200Mb).", type=["csv", "xlsx"])
		if uploaded_file is not None:
			uiclass.load_data(uploaded_file)
			status = uiclass.select_values_col()
			if status:
				uiclass.select_product_id_col()
	else:
		uiclass.load_data('default_table.csv')
		status = uiclass.select_values_col()
		if status:
			uiclass.select_product_id_col()


