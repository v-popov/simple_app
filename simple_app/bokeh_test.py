import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, FileInput, Paragraph, PreText, Select, DataTable, TableColumn
from bokeh.plotting import figure
import base64
import pandas as pd
from io import StringIO


class UIClass:

    def __init__(self):
        self.status = 'init' # init, data_inputted
        self.used_cols = []
        self.input_df = pd.DataFrame()
        self.inputs = None

        # Set up data
        self.N = 200
        self.x = np.linspace(0, 4 * np.pi, self.N)
        self.y = np.sin(self.x)
        self.source = ColumnDataSource(data=dict(x=self.x, y=self.y))

        # Set up widgets
        self.data_source_selector = Select(title='Step 1/5: Select Data', value='Use Example Data',
                                       options=['Use Example Data', 'Upload Data'])
        self.file_input = FileInput(accept='.csv,.xlsx')
        self.data_table = DataTable()
        self.values_col_selector = Select(title='Step 2/5: Select column with demand values', value='Not Selected',
                                       options=['Not Selected'])
        self.product_id_col_selector = Select(title='Step 3/5: Select column with product ID', value='Not Selected',
                                       options=['Not Selected'])
        # self.text = TextInput(title='title', value='my sine wave')
        # self.offset = Slider(title='offset', value=0.0, start=-5.0, end=5.0, step=0.1)
        # self.amplitude = Slider(title='amplitude', value=1.0, start=-5.0, end=5.0, step=0.1)
        # self.phase = Slider(title='phase', value=0.0, start=0.0, end=2 * np.pi)
        # self.freq = Slider(title='frequency', value=1.0, start=0.1, end=5.1, step=0.1)

        self.widgets = {'data_source_selector': self.data_source_selector,
                        'file_input': self.file_input,
                        'values_col_selector': self.values_col_selector,
                        'product_id_col_selector': self.product_id_col_selector,
                        'data_table': self.data_table,
                        #'': self.,
        }

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

    def select_data_source(self, attrname, old_val, new_val):
        print('===INSIDE select_data_source: {}, {}, {}'.format(attrname, old_val, new_val))
        # print('CHILDREN: {}; 1: {}'.format(self.row.children, self.row.children[1].children))
        # if len(self.row.children[1].children) == 0:
        #     self.plot.visible=True
        #     self.row.children[1].children.insert(-1, self.plot)
        if new_val == 'Upload Data':
            self.hide_all_widgets_except(['data_source_selector', 'file_input'])
            #self.file_input.visible = True
            #self.values_col_selector.visible = False
        else: # Use Example Data
            #self.file_input.visible = False
            self.input_df = pd.read_csv('default_table.csv')
            self.prepare_values_col_selection()
            self.preview_input_df()
            self.hide_all_widgets_except(['data_source_selector', 'values_col_selector', 'data_table'])

    def upload_fit_data(self, attr, old, new):
        print('fit data upload succeeded')
        base64_message = self.file_input.value
        base64_bytes = base64_message.encode('ascii')
        message_bytes = base64.b64decode(base64_bytes)
        message = message_bytes.decode('ascii')
        self.input_df = pd.read_csv(StringIO(message), sep=',')
        print('Input DF shape: {}'.format(self.input_df.shape))
        self.prepare_values_col_selection()
        self.hide_all_widgets_except(['data_source_selector', 'file_input', 'values_col_selector'])
        self.preview_input_df()
        #self.display()
        # if self.offset.visible:
        #     self.offset.visible = False
        # else:
        #     self.offset.visible = True
        #self.prepare_values_col_selection()

    def preview_input_df(self):
        columns = [TableColumn(field=Ci, title=Ci) for Ci in self.input_df.columns]
        self.data_table.columns = columns
        self.data_table.source = ColumnDataSource(self.input_df.head(5))
        self.data_table.visible = True
        #data_table = DataTable(columns=columns, source=ColumnDataSource(self.input_df))
        print('CHILDREN: {}; 1: {}'.format(self.row.children, self.row.children[1].children))
        while len(self.row.children[1].children) != 0:
            self.row.children[1].children.pop()
        #     #self.plot.visible=True
        self.row.children[1].children.insert(-1, self.data_table)

    def prepare_values_col_selection(self):
        self.values_col_selector.options = ['Not Selected'] + self.input_df.columns.tolist()
        #self.values_col_selector.visible = True

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

        #self.text.on_change('value', self.update_title)

        # for w in [self.offset, self.amplitude, self.phase, self.freq]:
        #     w.on_change('value', self.update_data)

        for w in [self.file_input,
                  self.values_col_selector,
                  self.product_id_col_selector,
                  # self.text,
                  # self.offset,
                  # self.amplitude,
                  # self.phase,
                  # self.freq,
                  self.plot]:
            w.visible = False

        # for w in [text, offset, amplitude, phase, freq]:
        #     w.width = 400

        # Set up layouts and add to document
        self.inputs = column(self.data_source_selector,
                             self.file_input,
                             self.values_col_selector,
                             self.product_id_col_selector,
                             # self.text,
                             # self.offset,
                             # self.amplitude,
                             # self.phase,
                             # self.freq
                             )

        #for input in self.inputs:
        #    input.sizing_mode = 'scale_both'


        self.data_source_selector.visible = True
        self.data_source_selector.on_change('value', self.select_data_source)

        # if self.status == 'init':
        #     curdoc().add_root(row(self.inputs))
        #     curdoc().title = 'Demand Forecasting'
        # else:
        self.col_left = column(self.inputs)
        #self.col_right = column()
        self.col_right = column()

        self.col_left.sizing_mode = 'scale_both' # doesn't work
        self.col_right.sizing_mode = 'scale_both' # doesn't work

        self.row = row(self.col_left, self.col_right)
        curdoc().add_root(self.row)
        curdoc().title = 'Demand Forecasting'


#if __name__ == '__main__':
uiclass = UIClass()
uiclass.display()
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