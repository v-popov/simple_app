import streamlit as st
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.layouts import layout
from bokeh.models import Paragraph, PreText, Select


class UIClass:

	def __init__(self):
		#self.data_selection_msg_ph = st.subheader('Step 1/N: Select Data')
		# self.data_selection_ph = st.empty()
		# self.data_upload_ph = st.empty()
		# self.data_preview_msg_ph = st.empty()
		# self.data_preview_ph = st.empty()
		#self.values_column_selector_ph = st.empty()
		#self.product_id_column_selector_ph = st.empty()
		# self.dashboard_ph = st.empty()
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
		self.data_preview_msg_ph.text('Data Preview:')#subheader
		self.data_preview_ph.dataframe(uiclass.input_df.head(5))

	@staticmethod
	def process_values_col(attr, old, new):
		st.write('HERE IS THE VAL: {}; {}; {}'.format(attr, old, new))

	def plot_dashboard(self):
		vals_col = 'col2'
		header = Paragraph(text='Some Dashboard Text')
		select_product_id_col = Select(title="Select product ID column", value="Not Selected", options=["Not Selected"]+self.input_df.columns.tolist())
		select_values_col = Select(title="Select demand values column", value="Not Selected", options=["Not Selected"]+self.input_df.columns.tolist())
		select_values_col.on_change('value', self.process_values_col)
		p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')
		p.line(self.input_df.index, self.input_df[vals_col], legend_label="Temp.", line_width=2)
		dashboard = layout([header], [select_product_id_col, select_values_col], [p])
		self.dashboard_ph.bokeh_chart(dashboard)

	# def select_values_col(self):
	# 	self.values_column = self.values_column_selector_ph.selectbox(label='Select column name containing demand values',
	# 																  options=['Not Selected'] + self.input_df.columns.tolist())
	# 	try:
	# 		self.input_df[self.values_column] = self.input_df[self.values_column].astype(np.int64)
	# 		self.select_product_id_col()
	# 	except:
	# 		self.product_id_column_selector_ph.markdown('**ERROR: Values should be numerical. Please, select another column containing demand values**')
	#
	# def select_product_id_col(self):
	# 	if self.product_id_column != 'Not Selected':
	# 		columns = set(self.input_df.columns)
	# 		columns.remove(self.values_column)
	# 		self.product_id_column = self.product_id_column_selector_ph.selectbox(label='Select column name containing product ID',
	# 																			  options=['Not Selected'] + list(columns))
	# 	else:
	# 		pass #add some placeholder for info msg; potentially next useful one

import plotly.graph_objects as go
from ipywidgets import widgets

def response(change):
	print(8)
	print(change)

if __name__ == '__main__':

	trace1 = go.Histogram(x=[1,2,3,4,5,1,2,3,2,3,2,3], opacity=0.75, name='Trace1')
	trace2 = go.Histogram(x=[1,3,3,4,7,1,5,2,2,3,1,3], opacity=0.75, name='Trace2')
	g = go.FigureWidget(data=[trace1, trace2],
						layout=go.Layout(
							title=dict(
								text='NYC FlightDatabase'
							),
							barmode='overlay'
						))
	origin = widgets.Dropdown(
		options=['q', 'w', 'e'],
		value='q',
		description='Origin',
	)
	origin.observe(response, names="value")

	container = widgets.HBox([origin])
	qqq = widgets.VBox([g])
	st.plotly_chart(figure_or_data=qqq)
	# x = [1, 2, 3, 4, 5]
	# y = [6, 7, 2, 4, 5]
	# p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')
	# p.line(x, y, legend_label="Temp.", line_width=2)
	#
	# #file_input = FileInput(accept=".csv,.xlsx")
	# #file_input.on_change('value', upload_fit_data)
	# layout = ([p])
	# curdoc.add_root(layout)
	#show(p)
	# uiclass = UIClass()
	# data_source_selector = uiclass.data_selection_ph.selectbox(label='Select Data', options=('Use Example Data', 'Upload Data'))
	# if data_source_selector == 'Upload Data':
	# 	uploaded_file = uiclass.data_upload_ph.file_uploader("Choose a CSV or XLSX file (max file size is 200Mb).", type=["csv", "xlsx"])
	# 	if uploaded_file is not None:
	# 		uiclass.load_data(uploaded_file)
	# 		uiclass.plot_dashboard()
	# 		#uiclass.select_values_col()
	# else:
	# 	uiclass.load_data('default_table.csv')
	# 	uiclass.plot_dashboard()
	# 	#uiclass.select_values_col()


