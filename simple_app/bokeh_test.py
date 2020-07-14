# source /home/victor/simple_app/bin/activate
# bokeh serve --allow-websocket-origin="*" bokeh_test.py
# ToDo: remove legend

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CDSView, GroupFilter, Slider, TextInput, FileInput, Paragraph, PreText, \
                         Select, DataTable, TableColumn, DatePicker, Range, CheckboxGroup, DatetimeTickFormatter, \
                         Button, DateFormatter
from bokeh.plotting import figure
import base64
import pandas as pd
from io import StringIO
from fbprophet import Prophet
import datetime
from math import radians

DF_NUM_PREVIEW_ROWS = 4
DATATABLE_PREVIEW_COL_WIDTH = 80  # pixels
DATATABLE_PREVIEW_HEIGHT = 35 * DF_NUM_PREVIEW_ROWS  # pixels
DATATABLE_PREVIEW_WIDTH = DATATABLE_PREVIEW_COL_WIDTH * 3  # pixels


class UIClass:

    def __init__(self):
        self.input_df = pd.DataFrame({'x': ['2010-01-01'] * DF_NUM_PREVIEW_ROWS, 'y': [0] * DF_NUM_PREVIEW_ROWS})
        self.forecasted_df = None
        self.datefmt = DateFormatter(format='%m-%d-%Y')
        self.inputs = None
        self.x_range = [0, 10]
        self.demand_plot = figure(x_range=self.x_range, x_axis_type="datetime", tools = ["pan", 'wheel_zoom'])#,wheel_zoom,box_zoom,reset,resize")

        self.plot_data_source = ColumnDataSource(data=self.input_df) #dict(x=[0], y=[0])
        self.line1 = self.demand_plot.line(x='x', y='y', source=self.plot_data_source, line_color='blue', name='line1')
        self.demand_plot.xaxis.formatter = DatetimeTickFormatter(days="%d %b %Y", hours="")
        self.demand_plot.axis.minor_tick_line_color = None
        self.demand_plot.xaxis[0].ticker.desired_num_ticks=10#num_minor_ticks = 0
        self.demand_plot.xaxis.major_label_orientation = radians(30) # from math import radians

        # Set up widgets
        self.data_source_selector = Select(title='Step 1/5: Select Data',
                                           value='Not Selected',
                                           options=['Not Selected', 'Use Example Data', 'Upload Data'])
        self.file_input = FileInput(accept='.csv,.xlsx')
        self.data_table = DataTable(height=DATATABLE_PREVIEW_HEIGHT,
                                    width=DATATABLE_PREVIEW_WIDTH,
                                    fit_columns=False,
                                    index_position=None,
                                    margin=(0,15,0,15), #aspect_ratio=0.5,
                                    #default_size=50
                                    )
        self.data_preview_paragraph = Paragraph(text='Data Preview:',
                                                margin=(0,15,0,15))
        self.values_col_selector = Select(title='Step 2/5: Select column with demand values',
                                          value='Not Selected',
                                          options=['Not Selected'])
        self.product_id_col_selector = Select(title='Step 3/5: Select column with product ID',
                                              value='Not Selected',
                                              options=['Not Selected'])
        self.date_col_selector = Select(title="Step 4/5: Select date column",
                                        value='Not Selected',
                                        options=['Not Selected'])
        self.last_date_picker = DatePicker(title='Select the date of last observation',
                                           max_date=datetime.datetime.date(pd.to_datetime("today")),
                                           value=datetime.datetime.date(pd.to_datetime("today")))
        self.workdays_checkboxgroup = CheckboxGroup(labels=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
                                                    active=[],
                                                    inline=True,
                                                    margin=(0,15,0,0))
        self.workdays_apply_button = Button(label='Select Business Days', button_type='primary')
        self.product_selector_plotting = Select(title='Select Product to Display',
                                                value='v1',
                                                options=['v1', 'v2'])
        self.prediction_button = Button(label='Forecast Demand for Selected Product ID', button_type='primary')
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
                        'date_col_selector': self.date_col_selector,
                        'last_date_picker': self.last_date_picker,
                        'workdays_checkboxgroup': self.workdays_checkboxgroup,
                        'workdays_apply_button': self.workdays_apply_button,
                        'prediction_button': self.prediction_button,
                        #'': self.,
        }

        self.values_colname = None
        self.product_id_colname = None
        self.date_colname = None
        self.product_ids = []

########## WIDGETS VISIBILITY CONTROLS ##########
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

########## LOGIC ##########
    def set_widget_to_default_value(self, widget_names, default_val='Not Selected'):
        for widget_name in widget_names:
            self.widgets[widget_name].value = default_val

    def prepare_values_col_selection(self):
        self.values_col_selector.options = ['Not Selected'] + self.input_df.columns.tolist()

    def get_additional_cols_to_show(self):
        return ['file_input'] if self.data_source_selector.value == 'Upload Data' else []

    def update_details_msg(self, msg):
        self.info_paragraph.text = "Details:\n{}".format(msg)

    def preview_input_df(self):
        # https://stackoverflow.com/questions/40942168/how-to-create-a-bokeh-datatable-datetime-formatter
        columns = [TableColumn(field=Ci, title=Ci, width=DATATABLE_PREVIEW_COL_WIDTH) for Ci in self.input_df.columns]
        self.data_table.update(columns=columns)
        self.data_table.update(source=ColumnDataSource(self.input_df.head(DF_NUM_PREVIEW_ROWS)))
        self.data_table.visible = True
        self.data_preview_paragraph.visible = True

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

    def replace_selector_options(self, selector, old_value, new_options):
        selector.options = [old_value] + new_options
        selector.value = new_options[0]
        selector.options = new_options

    def date_col_integrity(self, date_colname):
        if not isinstance(self.input_df[date_colname][0], str):
            self.input_df[date_colname] = self.input_df[date_colname].astype(str)
        if '-' in self.input_df[date_colname][0]:
            sep = '-'
        elif '/' in self.input_df[date_colname][0]:
            sep = '/'
        else:
            return 'no separator found'
        date_parts = self.input_df[date_colname].apply(lambda x: x.split(sep))
        if (date_parts.apply(lambda x: len(x)) == 3).all():
            try:
                self.input_df[date_colname] = pd.to_datetime(self.input_df[date_colname])
                return 'ok'
            except:
                return 'error converting to datetime'
        else:
            return 'not all dates have exactly 3 components'


    def display_preview_plot(self):
        self.replace_selector_options(self.product_selector_plotting, 'v1', self.product_ids)
        self.product_selector_plotting.visible = True
        self.prediction_button.visible = True
        self.demand_plot.renderers.remove(self.line1)
        self.plot_data_source = None
        self.plot_data_source = ColumnDataSource(data=self.input_df[self.input_df[self.product_id_colname] ==
                                                                    self.product_ids[0]])
        self.line1 = self.demand_plot.line(x=self.date_colname,
                                           y=self.values_colname,
                                           source=self.plot_data_source,
                                           line_color='blue',
                                           name='line1')
        self.update_plot(None, None, self.product_ids[0])
        self.demand_plot.visible = True

    def generate_dates(self, end_date: datetime.datetime, work_days: list, num_periods: int):
        work_days = ' '.join(work_days)  # 'Sun Mon Tue Wed Fri'
        freq = pd.offsets.CustomBusinessDay(weekmask=work_days)
        return pd.date_range(end=end_date, periods=num_periods, freq=freq)

    def clean_df(self):
        """
        Modifies self.input_df:
        1) Removing duplicates based on [self.date_colname, self.product_id_colname]
        2) Sorting based on self.date_colname
        :return: void
        """
        self.input_df = self.input_df[~self.input_df.duplicated(subset=[self.date_colname, self.product_id_colname],
                                                               keep='first')]
        self.input_df.sort_values(by=self.date_colname, inplace=True)
        print('===RESULTED INPUT_DF SHAPE AFTER CLEANING: ', self.input_df.shape)

########## WIDGETS ON_CHANGE METHODS ##########
    def select_data_source(self, attrname, old_val, new_val):
        self.set_widget_to_default_value(['values_col_selector', 'product_id_col_selector', 'date_col_selector'])
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

    def select_values_colname(self, attrname, old_val, new_val):
        self.update_details_msg(msg='Step 2/5: Please select a column that contains\nthe demand values. Note, that all the values in\nthis column should be numerical.')
        self.set_widget_to_default_value(['product_id_col_selector', 'date_col_selector'])
        self.hide_all_widgets_except(['data_source_selector',
                                      'values_col_selector',
                                      'data_preview_paragraph',
                                      'data_table'] + self.get_additional_cols_to_show())
        if new_val == 'Not Selected':
            pass
        else:
            self.values_colname = new_val
            try:
                self.input_df[self.values_colname] = self.input_df[self.values_colname].astype(float)
                available_cols = set(self.input_df.columns)
                available_cols.remove(self.values_colname)
                if self.date_colname in available_cols:
                    available_cols.remove(self.date_colname)
                self.product_id_col_selector.options = ['Not Selected'] + list(available_cols)
                self.product_id_col_selector.visible = True
            except:
                self.update_details_msg(msg='WARNING! Step 2/5: Not all the values\nin selected column are numerical!')

    def select_product_id_colname(self, attrname, old_val, new_val):
        self.update_details_msg(msg="Step 3/5: Please select a column that contains products' identifiers.")
        self.set_widget_to_default_value(['date_col_selector'])
        self.hide_all_widgets_except(['data_source_selector',
                                      'values_col_selector',
                                      'data_preview_paragraph',
                                      'data_table',
                                      'product_id_col_selector'] + self.get_additional_cols_to_show())
        if new_val == 'Not Selected':
            pass
        else:
            self.product_id_colname = new_val
            self.product_ids = self.input_df[self.product_id_colname].unique().astype(str).tolist()
            available_cols = set(self.input_df.columns)
            for colname in [self.values_colname, self.product_id_colname]:
                available_cols.remove(colname)
            if self.date_colname in available_cols:
                available_cols.remove(self.date_colname)
            self.date_col_selector.options = ['Not Selected'] + list(available_cols)
            self.date_col_selector.visible = True
            self.last_date_picker.visible = True
            self.workdays_checkboxgroup.visible = True
            self.workdays_apply_button.visible = True

    def select_date_column(self, attrname, old_val, new_val):
        self.update_details_msg(msg="Step 4/5: If there is a date column, please select it's name.\n"
                                    "Note: Dates should be in one of the following formats:\n"
                                    "yyyy-mm-dd OR mm-dd-yyyy OR yyyy/mm/dd OR mm/dd/yyyy\n"
                                    "If there is no such column, use 'Not Selected' option.")
        self.hide_all_widgets_except(['data_source_selector',
                                      'values_col_selector',
                                      'data_preview_paragraph',
                                      'data_table',
                                      'product_id_col_selector',
                                      'date_col_selector'] + self.get_additional_cols_to_show())
        if new_val == 'Not Selected':
            self.last_date_picker.visible = True
            self.workdays_checkboxgroup.visible = True
            self.workdays_apply_button.visible = True
        else:
            self.date_colname = new_val
            date_col_integrity_status = self.date_col_integrity(self.date_colname)
            if date_col_integrity_status == 'ok':
                self.clean_df()
                self.display_preview_plot()
            else:
                print('date_col_integrity_status: ', date_col_integrity_status)
                self.update_details_msg(msg="ERROR: selected date column doesn't satisfy specified requirements:\n"
                                            "Dates should be in one of the following formats:\n"
                                            "yyyy-mm-dd OR mm-dd-yyyy OR yyyy/mm/dd OR mm/dd/yyyy\n"
                                            "If there is no such column, use 'Not Selected' option.")

    def select_last_date(self, attrname, old_val, new_val):
        self.update_details_msg(msg="Alright, dates will be automatically generated for you!\n"
                                    "Select days when your business works.")
        self.workdays_checkboxgroup.visible = True
        self.workdays_apply_button.visible = True

    def workdays_button_pressed(self, new):
        if len(self.workdays_checkboxgroup.active) == 0:
            self.update_details_msg(msg="Please select at least one business day.")
        else:
            self.update_details_msg(msg="Generating dates.")
            if 'generated_dates' in self.input_df.columns:
                self.update_details_msg(msg="Please rename the generated_dates column in you table.")
            else:
                self.date_colname = 'generated_date'
                self.input_df[self.date_colname] = ''
                for product_id in self.product_ids:
                    inds = self.input_df[self.product_id_colname] == product_id
                    self.input_df.loc[inds, self.date_colname] = self.generate_dates(end_date=self.last_date_picker.value,
                                                                                     work_days=np.array(self.workdays_checkboxgroup.labels)[self.workdays_checkboxgroup.active],
                                                                                     num_periods=inds.sum())
                self.input_df[self.date_colname] = pd.to_datetime(self.input_df[self.date_colname])
                self.clean_df()
                self.display_preview_plot()
                #self.preview_input_df() # https://stackoverflow.com/questions/40942168/how-to-create-a-bokeh-datatable-datetime-formatter

    # ToDo: different colors for historical and predicted values: https://stackoverflow.com/questions/59017033/bokeh-unable-to-generate-different-line-colours-when-using-multiline-glyph
    # https://docs.bokeh.org/en/latest/docs/user_guide/styling.html
    def prediction_button_pressed(self, new):
        train_dataset = pd.DataFrame()
        print('Preparing forecast for product: ', self.product_selector_plotting.value)
        inds = self.input_df[self.product_id_colname] == self.product_selector_plotting.value
        train_dataset['ds'] = pd.to_datetime(self.input_df.loc[inds, self.date_colname])
        train_dataset['y'] = self.input_df.loc[inds, self.values_colname]
        # train_dataset = train_dataset[train_dataset.duplicated(subset=['ds'],keep='first')]
        #train_dataset.sort_values(by=self.date_colname, inplace=True)
        print('Train Dataset shape: ', train_dataset.shape)
        for q in self.make_predictions(train_dataset):
            if q[0] == 'msg':
                print('Message: ', q[1])
            else:
                self.forecasted_df = q[1]
                self.forecasted_df.columns = ['ds', 'y']
                print('Done; shape: ', self.forecasted_df.shape)
                #self.demand_plot.line(x='ds', y='yhat', source=ColumnDataSource(data=self.forecasted_df, name='line2'))
                #print(self.forecasted_df.tail(30))

                #combined_dataset = train_dataset.append(self.forecasted_df.tail(30), ignore_index=True)
                d = {'ds': train_dataset['ds'].append(self.forecasted_df.tail(30)['ds']),
                     'y': train_dataset['y'].append(self.forecasted_df.tail(30)['y'])}
                combined_dataset = pd.DataFrame(d)

                try:
                    while len(self.demand_plot.legend[0].items) > 0:
                        self.demand_plot.legend[0].items.pop()
                except:
                    print('FAIL: popping legends in prediction_button_pressed()')

                self.demand_plot.renderers.remove(self.line1)
                try:
                    self.demand_plot.renderers.remove(self.line2)
                except:
                    pass

                self.plot_data_source = None
                self.plot_data_source = ColumnDataSource(data=combined_dataset)
                self.line1 = self.demand_plot.line(x=train_dataset['ds'],
                                                   y=train_dataset['y'],
                                                   line_color='blue',
                                                   name='line1',
                                                   legend_label='Historical')
                self.line2 = self.demand_plot.line(x=train_dataset['ds'].tail(1).append(self.forecasted_df['ds'].tail(30)),
                                                   y=train_dataset['y'].tail(1).append(self.forecasted_df['y'].tail(30)),
                                                   line_color='red',
                                                   name='line2',
                                                   legend_label='Forecast')
                #print('QQQ ', self.demand_plot.select(name="line2"))
                self.demand_plot.legend.location = "top_left"

                self.demand_plot.x_range.start = combined_dataset['ds'].min()
                self.demand_plot.x_range.end = combined_dataset['ds'].max()
                self.demand_plot.y_range.start = combined_dataset['y'].min()
                self.demand_plot.y_range.end = combined_dataset['y'].max()

                self.demand_plot.visible = True

########## OTHER ##########
    # https://facebook.github.io/prophet/docs/non-daily_data.html
    def make_predictions(self, df, days_ahead=30):
        yield ['msg', 'training model']
        prophet = Prophet(weekly_seasonality=False, daily_seasonality=False)
        prophet.fit(df)
        yield ['msg', 'making predictions']
        future = prophet.make_future_dataframe(periods=days_ahead)
        forecast = prophet.predict(future)
        yield ['results', forecast[['ds', 'yhat']]]

    def update_plot(self, attrname, old, new):
        try:
            while len(self.demand_plot.legend[0].items) > 0:
                self.demand_plot.legend[0].items.pop()
        except:
            print('FAIL: popping legends in update_plot()')
        try:
            self.demand_plot.renderers.remove(self.line2)
        except:
            pass

        sub_df = self.input_df[self.input_df[self.product_id_colname] == new]

        self.demand_plot.renderers.remove(self.line1)
        self.plot_data_source = None
        self.plot_data_source = ColumnDataSource(data=sub_df)
        self.line1 = self.demand_plot.line(x=self.date_colname,
                                           y=self.values_colname,
                                           source=self.plot_data_source,
                                           line_color='blue',
                                           legend_label='Historical',
                                           name='line1')
        self.demand_plot.legend.location = "top_left"
        self.demand_plot.x_range.start = sub_df[self.date_colname].min()
        self.demand_plot.x_range.end = sub_df[self.date_colname].max()
        self.demand_plot.y_range.start = sub_df[self.values_colname].min()
        self.demand_plot.y_range.end = sub_df[self.values_colname].max()

    def display(self):
        self.file_input.on_change('value', self.upload_fit_data)
        self.plot = figure(plot_height=400, plot_width=400, title='my sine wave',
                      tools='crosshair,pan,reset,save,wheel_zoom')

        # Set up layouts and add to document
        self.inputs = column(self.data_source_selector,
                             self.file_input,
                             self.values_col_selector,
                             self.product_id_col_selector,
                             self.date_col_selector,
                             self.last_date_picker,
                             self.workdays_checkboxgroup,
                             self.workdays_apply_button
                             )

        #self.data_source_selector.visible = True
        self.hide_all_widgets_except(['data_source_selector'])
        self.data_source_selector.on_change('value', self.select_data_source)
        self.values_col_selector.on_change('value', self.select_values_colname)
        self.product_id_col_selector.on_change('value', self.select_product_id_colname)
        self.product_selector_plotting.on_change('value', self.update_plot)
        self.date_col_selector.on_change('value', self.select_date_column)
        self.last_date_picker.on_change('value', self.select_last_date)
        self.workdays_apply_button.on_click(self.workdays_button_pressed)
        self.prediction_button.on_click(self.prediction_button_pressed)

        #self.col_left = self.inputs

        columns = [TableColumn(field=Ci, title=Ci, width=DATATABLE_PREVIEW_COL_WIDTH) for Ci in self.input_df.columns]
        self.data_table.columns = columns
        self.data_table.source = ColumnDataSource(self.input_df.head(DF_NUM_PREVIEW_ROWS))

        self.col_middle = column(self.data_preview_paragraph, self.data_table)
        #self.col_info = column()


        #self.col_left.width = 300
        #self.col_right.max_width = 500
        #self.col_right.sizing_mode = 'scale_width'

        #self.row_data_input = row(self.col_left, self.col_right, self.info_paragraph)
        #self.row_data_input.sizing_mode = 'scale_width'

        #self.row_demand_plot = row(self.product_selector_plotting)#, self.demand_plot)

        #self.layout = column(self.row_data_input, self.row_demand_plot)

        self.layout = column(row(column(self.data_source_selector,
                                        self.file_input,
                                        self.values_col_selector,
                                        self.product_id_col_selector,),
                                 column(self.data_preview_paragraph,
                                        self.data_table),
                                 self.info_paragraph),
                             row(column(self.date_col_selector,
                                        self.last_date_picker,
                                        self.workdays_checkboxgroup,
                                        self.workdays_apply_button,
                                        self.product_selector_plotting,
                                        self.prediction_button),
                                 self.demand_plot))

        curdoc().add_root(self.layout)
        curdoc().title = 'Demand Forecasting'


uiclass = UIClass()
uiclass.display()

# print('LAYOUT: ', self.layout)
# print('CHILDREN: {}; 1: {}'.format(self.row_data_input.children, self.row_data_input.children[1].children))
# while len(self.row_data_input.children[1].children) != 1:
#    self.row_data_input.children[1].children.pop()
# self.row_data_input.children[1].children.append(self.data_table)
# self.row_data_input.children[1].children.insert(-1, self.data_preview_paragraph)