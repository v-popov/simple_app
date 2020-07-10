# source /home/victor/simple_app/bin/activate
# bokeh serve --allow-websocket-origin="*" bokeh_test.py
# ToDo: debug product ID selector (for plotting) not displaying

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CDSView, GroupFilter, \
    Slider, TextInput, FileInput, Paragraph, PreText, Select, DataTable, TableColumn
from bokeh.plotting import figure
import base64
import pandas as pd
from io import StringIO


DATATABLE_PREVIEW_COL_WIDTH = 80  # pixels
DATATABLE_PREVIEW_HEIGHT = 120  # pixels
DATATABLE_PREVIEW_WIDTH = DATATABLE_PREVIEW_COL_WIDTH * 3  # pixels

class UIClass:

    def __init__(self):
        self.input_df = pd.DataFrame({'x': ['a'], 'y': [0]})
        self.inputs = None
        
        self.demand_plot = figure()
        #self.demand_plot.line('x', 'y', source=ColumnDataSource(data=dict(x=[0], y=[0])))
        self.plot_data_source = ColumnDataSource(data=self.input_df) #dict(x=[0], y=[0])
        self.demand_plot.line(x='x', y='y', source=self.plot_data_source)

        # Set up data
        self.N = 200
        self.x = np.linspace(0, 4 * np.pi, self.N)
        self.y = np.sin(self.x)
        self.source = ColumnDataSource(data=dict(x=self.x, y=self.y))

        # Set up widgets
        self.data_source_selector = Select(title='Step 1/5: Select Data', value='Not Selected',
                                       options=['Not Selected', 'Use Example Data', 'Upload Data'])
        self.file_input = FileInput(accept='.csv,.xlsx')
        self.data_table = DataTable(height=DATATABLE_PREVIEW_HEIGHT,
                                    width=DATATABLE_PREVIEW_WIDTH,
                                    fit_columns=False, index_position=None,
                                    margin=(0,15,0,15), #aspect_ratio=0.5,
                                    #default_size=50
                                    )
        self.data_preview_paragraph = Paragraph(text='Data Preview:', margin=(0,15,0,15))
        self.values_col_selector = Select(title='Step 2/5: Select column with demand values', value='Not Selected',
                                       options=['Not Selected'])
        self.product_id_col_selector = Select(title='Step 3/5: Select column with product ID', value='Not Selected',
                                       options=['Not Selected'])
        self.product_selector_plotting = Select(title='Select Product to Display', value='v1', options=['v1', 'v2'])
        self.default_info_msg = 'This window will contain additional information,\nas you interact with the app.'
        self.info_paragraph = PreText(text='Details:\n{}'.format(self.default_info_msg))
        # self.text = TextInput(title='title', value='my sine wave')
        # self.offset = Slider(title='offset', value=0.0, start=-5.0, end=5.0, step=0.1)

        self.widgets = {'data_source_selector': self.data_source_selector,
                        'file_input': self.file_input,
                        'values_col_selector': self.values_col_selector,
                        'product_id_col_selector': self.product_id_col_selector,
                        'data_preview_paragraph': self.data_preview_paragraph,
                        'data_table': self.data_table,
                        'product_selector': self.product_selector_plotting,
                        'demand_plot': self.demand_plot,
                        #'': self.,
        }

        self.values_colname = None
        self.product_id_colname = None
        self.product_ids = []

    def _change_widgets_visibility(self, names, names_show_or_hide='show'):
        displaying = True if names_show_or_hide == 'show' else False
        for widget_name in self.widgets:
            if widget_name in names:
                self.widgets[widget_name].visible = displaying
            else:
                self.widgets[widget_name].visible = not displaying

    def display_all_widgets_except(self, widgets=[]):
        self._change_widgets_visibility(widgets, 'hide')

    def hide_all_widgets_except(self, widgets=[]):
        self._change_widgets_visibility(widgets, 'show')

    def update_details_msg(self, msg):
        self.info_paragraph.text = "Details:\n{}".format(msg)

    def select_data_source(self, attrname, old_val, new_val):
        print('===INSIDE select_data_source: {}, {}, {}'.format(attrname, old_val, new_val))
        if new_val == 'Upload Data':
            self.update_details_msg(msg='Step 1/5: Please upload data in one of the\nfollowing formats: .CSV or .XLSX')
            self.hide_all_widgets_except(['data_source_selector', 'file_input'])
        elif new_val == 'Use Example Data':
            self.update_details_msg(msg='Step 1/5: Using a sample toy data. You can use it\nto test the functionality of this app.')
            self.input_df = pd.read_csv('default_table.csv')
            self.prepare_values_col_selection()
            self.preview_input_df()
            self.hide_all_widgets_except(['data_source_selector', 'values_col_selector', 'data_preview_paragraph', 'data_table'])
        else: # Not Selected
            self.update_details_msg(msg=self.default_info_msg)
            self.hide_all_widgets_except(['data_source_selector'])

    def upload_fit_data(self, attr, old, new):
        print('fit data upload succeeded')
        self.update_details_msg(msg='Step 1/5: Uploading data')
        base64_message = self.file_input.value
        base64_bytes = base64_message.encode('ascii')
        message_bytes = base64.b64decode(base64_bytes)
        message = message_bytes.decode('ascii')
        self.input_df = pd.read_csv(StringIO(message), sep=',')
        self.update_details_msg(msg='Step 1/5: Data has been successfully uploaded!')
        print('Input DF shape: {}'.format(self.input_df.shape))
        self.prepare_values_col_selection()
        self.hide_all_widgets_except(['data_source_selector', 'file_input', 'values_col_selector'])
        self.preview_input_df()

    def preview_input_df(self):
        columns = [TableColumn(field=Ci, title=Ci, width=DATATABLE_PREVIEW_COL_WIDTH) for Ci in self.input_df.columns]
        self.data_table.columns = columns
        self.data_table.source = ColumnDataSource(self.input_df.head(3))
        print('LAYOUT: ', self.layout)
        print('CHILDREN: {}; 1: {}'.format(self.row_data_input.children, self.row_data_input.children[1].children))
        while len(self.row_data_input.children[1].children) != 1:
            self.row_data_input.children[1].children.pop()
        self.row_data_input.children[1].children.append(self.data_table)
        #self.row_data_input.children[1].children.insert(-1, self.data_preview_paragraph)
        self.data_table.visible = True
        self.data_preview_paragraph.visible = True

    def prepare_values_col_selection(self):
        self.values_col_selector.options = ['Not Selected'] + self.input_df.columns.tolist()

    def select_values_colname(self, attrname, old_val, new_val):
        self.update_details_msg(msg='Step 2/5: Please select a column that contains\nthe demand values. Note, that all the values in\nthis column should be numerical.')
        if new_val == 'Not Selected':
            #pass
            self.hide_all_widgets_except(['data_source_selector',
                                          'values_col_selector',
                                          'data_table',
                                          'data_preview_paragraph',
                                          'data_table']+self.get_additional_cols_to_show())
        else:
            self.values_colname = new_val
            try:
                self.input_df[self.values_colname] = self.input_df[self.values_colname].astype(float)
                available_cols = set(self.input_df.columns)
                available_cols.remove(new_val)
                self.product_id_col_selector.options = ['Not Selected'] + list(available_cols)
                self.product_id_col_selector.visible=True
                self.hide_all_widgets_except(['data_source_selector',
                                              'values_col_selector',
                                              'product_id_col_selector',
                                              'data_preview_paragraph',
                                              'data_table'
                                              ]+self.get_additional_cols_to_show())
            except:
                self.update_details_msg(msg='WARNING! Step 2/5: Not all the values\nin selected column are numerical!')
                self.hide_all_widgets_except(['data_source_selector',
                                              'values_col_selector',
                                              'data_table',
                                              'data_preview_paragraph',
                                              'data_table'] + self.get_additional_cols_to_show())

    def get_additional_cols_to_show(self):
        return ['file_input'] if self.data_source_selector.value == 'Upload Data' else []

    def replace_selector_options(self, selector, old_value, new_options):
        selector.options = [old_value] + new_options
        selector.value = new_options[0]
        selector.options = new_options

    # ToDo: replace 'day' with something better
    def select_product_id_colname(self, attrname, old_val, new_val):
        self.update_details_msg(msg="Step 3/5: Please select a column that contains products' identifiers.")
        if new_val == 'Not Selected':
            self.hide_all_widgets_except(['data_source_selector',
                                          'values_col_selector',
                                          'data_table',
                                          'product_id_col_selector']+self.get_additional_cols_to_show())
        else:
            self.product_id_colname = new_val
            self.product_ids = self.input_df[self.product_id_colname].unique().astype(str).tolist()
            self.replace_selector_options(self.product_selector_plotting, 'v1', self.product_ids)
            self.product_selector_plotting.visible = True

            self.plot_data_source = ColumnDataSource(data=self.input_df[self.input_df[self.product_id_colname] ==
                                                                         self.product_ids[0]])
            self.demand_plot.line(x='day', y=self.values_colname, source=self.plot_data_source)
            self.demand_plot.visible = True

    def update_plot(self, attrname, old, new):
        self.plot_data_source.data.update(self.input_df[self.input_df[self.product_id_colname] == new])

    def update_title(self, attrname, old, new):
        self.plot.title.text = self.text.value

    def update_data(self, attrname, old, new):
        # Get the current slider values
        a = self.amplitude.value
        b = self.offset.value
        w = self.phase.value
        k = self.freq.value

        # Generate the new curve
        self.x = np.linspace(0, 4 * np.pi, self.N)
        self.y = a * np.sin(k * self.x + w) + b

        self.source.data=dict(x=self.x, y=self.y)

    def display(self):
        self.file_input.on_change('value', self.upload_fit_data)
        self.plot = figure(plot_height=400, plot_width=400, title='my sine wave',
                      tools='crosshair,pan,reset,save,wheel_zoom',
                      x_range=[0, 4 * np.pi], y_range=[-2.5, 2.5])
        self.plot.line('x', 'y', source=self.source, line_width=3, line_alpha=0.6)

        for w in [self.file_input,
                  self.values_col_selector,
                  self.product_id_col_selector,
                  self.plot]:
            w.visible = False

        # Set up layouts and add to document
        self.inputs = column(self.data_source_selector,
                             self.file_input,
                             self.values_col_selector,
                             self.product_id_col_selector,
                             )

        #self.data_source_selector.visible = True
        self.hide_all_widgets_except(['data_source_selector'])
        self.data_source_selector.on_change('value', self.select_data_source)
        self.values_col_selector.on_change('value', self.select_values_colname)
        self.product_id_col_selector.on_change('value', self.select_product_id_colname)
        self.product_selector_plotting.on_change('value', self.update_plot)

        self.col_left = self.inputs
        self.col_right = column(self.data_preview_paragraph)
        #self.col_info = column()


        self.col_left.width = 300
        #self.col_right.max_width = 500
        #self.col_right.sizing_mode = 'scale_width'

        self.row_data_input = row(self.col_left, self.col_right, self.info_paragraph)
        #self.row_data_input.sizing_mode = 'scale_width'

        self.row_demand_plot = row(self.product_selector_plotting, self.demand_plot)

        self.layout = column(self.row_data_input, self.row_demand_plot)

        curdoc().add_root(self.layout)
        curdoc().title = 'Demand Forecasting'


uiclass = UIClass()
uiclass.display()

# input_df = pd.read_csv('default_table.csv')
# plot_data_filter = GroupFilter(column_name='ccc1', group='qqq')
# sub_df = input_df[input_df['ccc1']=='qqq']
# plot_data_source = ColumnDataSource(data=sub_df)  # move to another method
# plot_data_view = CDSView(source=plot_data_source, filters=[plot_data_filter])
# demand_plot = figure()
# demand_plot.line(x='day', y='ccc2', source=plot_data_source)
# plot_data_source.data=input_df[input_df['ccc1']=='www']
# curdoc().add_root(demand_plot)
# curdoc().title = 'Demand Forecasting'
#show(demand_plot)

#curdoc().add_root(row(uiclass.inputs, uiclass.plot))
#curdoc().title = 'Sliders'
# N = 200
# x = np.linspace(0, 4*np.pi, N)
# y = np.sin(x)
# source = ColumnDataSource(data=dict(x=x, y=y))
#
# # Set up plot
# plot = figure(plot_height=400, plot_width=400, title='my sine wave',
#               tools='crosshair,pan,reset,save,wheel_zoom',
#               x_range=[0, 4*np.pi], y_range=[-2.5, 2.5])
#
# plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
#
# curdoc().add_root(plot)
# curdoc().title = 'Sliders'
#
#
# ############################# WORKING EXAMPLE BELOW ############################################
# def upload_fit_data(attr, old, new):
#     print('fit data upload succeeded')
#     base64_message = file_input.value
#     base64_bytes = base64_message.encode('ascii')
#     message_bytes = base64.b64decode(base64_bytes)
#     message = message_bytes.decode('ascii')
#     df = pd.read_csv(StringIO(message), sep=',')
#     print(df.shape)
# file_input = FileInput(accept='.csv,.xlsx')
# file_input.on_change('value', upload_fit_data)
#
# # Set up data
# N = 200
# x = np.linspace(0, 4*np.pi, N)
# y = np.sin(x)
# source = ColumnDataSource(data=dict(x=x, y=y))
#
# # Set up plot
# plot = figure(plot_height=400, plot_width=400, title='my sine wave',
#               tools='crosshair,pan,reset,save,wheel_zoom',
#               x_range=[0, 4*np.pi], y_range=[-2.5, 2.5])
#
# plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
# plot.sizing_mode = 'scale_both'
#
# # Set up widgets
# text = TextInput(title='title', value='my sine wave')
# offset = Slider(title='offset', value=0.0, start=-5.0, end=5.0, step=0.1)
# amplitude = Slider(title='amplitude', value=1.0, start=-5.0, end=5.0, step=0.1)
# phase = Slider(title='phase', value=0.0, start=0.0, end=2*np.pi)
# freq = Slider(title='frequency', value=1.0, start=0.1, end=5.1, step=0.1)
#
# # Set up callbacks
# def update_title(attrname, old, new):
#     plot.title.text = text.value
#
# text.on_change('value', update_title)
#
# def update_data(attrname, old, new):
#     # Get the current slider values
#     a = amplitude.value
#     b = offset.value
#     w = phase.value
#     k = freq.value
#
#     # Generate the new curve
#     x = np.linspace(0, 4*np.pi, N)
#     y = a*np.sin(k*x + w) + b
#
#     source.data = dict(x=x, y=y)
#
# for w in [offset, amplitude, phase, freq]:
#     w.on_change('value', update_data)
#
# # for w in [text, offset, amplitude, phase, freq]:
# #     w.width = 400
#
# # Set up layouts and add to document
# inputs = column(file_input, text, offset, amplitude, phase, freq)
# inputs.sizing_mode = 'scale_both'
#
# #curdoc().add_root(row(inputs, plot, width=800))
# curdoc().add_root(row(inputs, plot))
# curdoc().title = 'Sliders'