Although Chart.js is great library in general, there are some fundamental drawbacks to using it, and these are why work on this demo ha sbeen abandoned:

1. version 1, 2, and 2beta are substantially different in the realtime updating implementation of the chart
2. The way that data is added doesn't allow for x values in a line chart, only y values. This means you have to pass in every single data point one by one and they must be evenly spaced. Ideally a charting library should be able to take datapoints like (0,10) and (35,100) and draw the line between them. Chart.js however, only takes in an array of y values.
3. No way to turn off the data point dots
4. the line is curved between points for maximum smoothness, but there doesn't seem to be a way to turn this off, so charts look over-smoothed, making it nonstandard compared to other EEG data displays that users will be familiar with.
 