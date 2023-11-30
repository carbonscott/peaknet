import numpy as np
import os
import h5py

from bokeh.models         import ColumnDataSource, HoverTool, Div, CustomJS, PolyDrawTool, Button
from bokeh.plotting       import figure, show
from bokeh.io             import output_notebook
from bokeh.models.mappers import LogColorMapper
from bokeh.transform      import field
from bokeh.layouts        import row, column, gridplot
from bokeh.io             import curdoc
from bokeh.events         import ButtonClick

path_cxi = "inference_results/peaknet.cxic00318_0123.cxi"

try:
    fh.close()
except:
    pass

fh = h5py.File(path_cxi, "r")

idx = 0
img = fh.get('entry_1/data_1/data')[idx]

fh.close()

# Create ColumnDataSource
H, W = img.shape

data_source = ColumnDataSource(data=dict(
    x=[0],
    y=[0],
    dw=[W],
    dh=[H],
    f=[img],
))

H_viz = 1024 * 4
W_viz = H_viz * W / H
fig = figure(width = int(W_viz), height = int(H_viz), y_range = (0, H), x_range = (0, W), match_aspect=True)

vmin = img.mean()
vmax = img.mean() + 6 * img.std()
color_mapper = LogColorMapper(palette="Viridis256", low=vmin, high=vmax)

fig.image(source = data_source, image = 'f', x = 'x', y = 'y', dw = 'dw', dh = 'dh', color_mapper=color_mapper)
# fig.x_range.range_padding = fig.y_range.range_padding = 0
fig.grid.grid_line_width = 0.5

# Add hover tool with the callback
hover_tool = HoverTool(
    tooltips   = [('x', '$x{%d}'), ('y', '$y{%d}'), ('v', '@f{0.2f}')],
    formatters = {
        '$x': 'printf',
        '$y': 'printf',
        '@v': 'printf'
    })
fig.add_tools(hover_tool)


# Create a ColumnDataSource for polygons
polygons_source = ColumnDataSource(data=dict(xs=[[]], ys=[[]]))

# Add patch glyphs to the figure
fig.patches('xs', 'ys', source=polygons_source, color='red', line_width=2, alpha=0.6)

# Create and add the PolyDrawTool
draw_tool = PolyDrawTool(renderers=[fig.renderers[-1]], empty_value='red')
fig.add_tools(draw_tool)
fig.toolbar.active_drag = draw_tool

# Display the figure
button = Button(label="Get Polygons", button_type="success")

# Function to retrieve data
def get_polygon_data(event):
    xs, ys = polygons_source.data['xs'], polygons_source.data['ys']
    print(xs, ys)  # Or process the data as needed

button.on_event(ButtonClick, get_polygon_data)

# Show...
curdoc().add_root(column(button, fig))
