Although Chart.js is fine library in general, there are some fundamental drawbacks to using it, and these are why work on this demo has
been abandoned:

1. version 1, 2, and 2beta are substantially different in the implementation of the chart redraw
2. The way that data is added doesn't allow for x values in a line chart, only y values. This means you have to pass in every single data point one by one and they must be evenly spaced. Ideally a charting library should be able to take datapoints like (0,10) and (35,100) and draw the line between them. Chart.js however, only takes in an array of y values.
3. It's much too slow for rendering charts with 250-1000 points. See benchmarks here: [https://jsperf.com/amcharts-vs-highcharts-vs-canvasjs/17] 