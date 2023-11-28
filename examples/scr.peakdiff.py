#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bokeh.models   import ColumnDataSource, Circle, Div, CustomJS, Span, HoverTool
from bokeh.plotting import figure, show
from bokeh.layouts  import gridplot
from bokeh.io       import curdoc

import msgpack

from configurator import Configurator

# ___/ CONFIG \___
config = Configurator()
with config.enable_auto_create():
    config.path_n_peaks.ds0 = "peaknet.n_peaks.msgpack"
    config.path_m_rates.ds0 = "peaknet.m_rate.msgpack"
    config.path_n_peaks.ds1 = "pyalgo.n_peaks.msgpack"
    config.path_m_rates.ds1 = "pyalgo.m_rate.msgpack"

# ___/ DATA SOURCE \___
# Read the data to visualize...
n_peaks_dict = {}
m_rates_dict = {}
with open(config.path_n_peaks.ds0, 'rb') as f:
    data = f.read()
    n_peaks_dict['peaknet'] = msgpack.unpackb(data, strict_map_key = False)

with open(config.path_n_peaks.ds1, 'rb') as f:
    data = f.read()
    n_peaks_dict['pyalgo'] = msgpack.unpackb(data, strict_map_key = False)

with open(config.path_m_rates.ds0, 'rb') as f:
    data = f.read()
    m_rates_dict['peaknet'] = msgpack.unpackb(data, strict_map_key = False)

with open(config.path_m_rates.ds1, 'rb') as f:
    data = f.read()
    m_rates_dict['pyalgo'] = msgpack.unpackb(data, strict_map_key = False)

# Build the data source (it's pandas dataframe under the hood)...
data_source = dict(
    events    = list(n_peaks_dict['peaknet'].keys()),
    n_peaks_x = list(n_peaks_dict['pyalgo'].values()),
    n_peaks_y = list(n_peaks_dict['peaknet'].values()),
    n_peaks_l = [f"event {event:06d}, pyalgo:{m_pyalgo:.2f}, peaknet:{m_peaknet:.2f}"
                 for event, m_pyalgo, m_peaknet in zip(n_peaks_dict['pyalgo' ].keys(),
                                                       n_peaks_dict['pyalgo' ].values(),
                                                       n_peaks_dict['peaknet'].values())],
    m_rates_x = list(m_rates_dict['pyalgo'].values()),
    m_rates_y = list(m_rates_dict['peaknet'].values()),
    m_rates_l = [f"event {event:06d}, pyalgo:{m_pyalgo:.2f}, peaknet:{m_peaknet:.2f}"
                 for event, m_pyalgo, m_peaknet in zip(m_rates_dict['pyalgo' ].keys(),
                                                       m_rates_dict['pyalgo' ].values(),
                                                       m_rates_dict['peaknet'].values())],
)
data_source = ColumnDataSource(data_source)

# ___/ GUI \___
TOOLS = "box_select,lasso_select,wheel_zoom,pan,reset,help,"

fig = dict(
    n_peaks = figure(## width        =  500,
                     ## height       =  500,
                     tools        =  TOOLS,
                     title        = "Number of peaks comparison",
                     x_axis_label = 'pyalgo',
                     y_axis_label = 'peaknet',
                     match_aspect = True),
    m_rates = figure(## width        =  500,
                     ## height       =  500,
                     tools        =  TOOLS,
                     title        = "Match rate comparison",
                     x_axis_label = 'pyalgo',
                     y_axis_label = 'peaknet',
                     match_aspect = True),
)

scatter_n_peaks = fig['n_peaks'].scatter('n_peaks_x',
                                         'n_peaks_y',
                                         source                  = data_source,
                                         size                    = 10,
                                         fill_color              = "blue",
                                         line_color              =  None,
                                         fill_alpha              = 0.5,
                                         nonselection_fill_alpha = 0.005,
                                         nonselection_fill_color = "blue")
scatter_m_rates = fig['m_rates'].scatter('m_rates_x',
                                         'm_rates_y',
                                         source                  = data_source,
                                         size                    = 10,
                                         fill_color              = "red",
                                         line_color              =  None,
                                         fill_alpha              = 0.5,
                                         nonselection_fill_alpha = 0.005,
                                         nonselection_fill_color = "red")


# CustomJS callback to update Div on selection
selected_div = Div(width=500, height=500, text="Selected indices:")
callback = CustomJS(args=dict(source=data_source, div=selected_div), code="""
    const inds = source.selected.indices;
    let text = "Selected indices:\\n";
    for (let i = 0; i < inds.length; i++) {
        text += inds[i] + "\\n";
    }
    div.text = text;
""")

data_source.selected.js_on_change('indices', callback)

# Show...
curdoc().add_root(gridplot([[fig['n_peaks'], fig['m_rates']], [selected_div]]))
