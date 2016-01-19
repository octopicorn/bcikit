To run, from {bcikit root} folder:
```
$ python run.py  -i octopicorn -d openbci -c conf/viz_demos/eeg.yml
```
And then browse to http://localhost:9999

A regular demo is implemented, as well as one attempting to use the 3d line chart.
Plotly shows some interesting capabilities using 3d canvas rendering, but it was much too hard to get the chart to be aligned correctly.