// random value generator
function getNewData(){
  return Math.random() * 100;
}

// extend Chart class
Chart.prototype.containerId = null;
Chart.prototype.tid = null;





// globals
// set default window dimensions
var defaultDataLength = 100;
var defaultData = [];
var defaultLabels = [];
for(var i=0;i<defaultDataLength;i++){
  defaultData.push(getNewData());
  defaultLabels.push("");
}
var dataLine = {
    labels: defaultLabels,
    datasets: [
        {
            label: "My Second dataset",
            strokeColor: "rgba(151,187,205,1)",
            data: defaultData
        }
    ]
};

// main runtime
$(document).ready(function(){
  // this will hold interval timer object
  var tid;

});


function createChart($canvas){
    var containerId = $canvas.parents('.chartContainer').attr('id');

    // Get context with jQuery - using jQuery's .get() method.
    var ctx = $canvas.get(0).getContext("2d");
    // This will get the first returned node in the jQuery collection.
    //var lineChart = new Chart(ctx).Line(data, options);
    var lineChart = new Chart(ctx).Line(dataLine,optionsLine);
    console.log(lineChart)
    lineChart.containerId = containerId;

    function abortTimer() { // to be called when you want to stop the timer
      clearInterval(tid);
    }

    return lineChart;
}

function bindChartButtons(containerId){
  // BINDS

    // add one data at a time manually
    $('#'+containerId+' .one_more').on('click',function(e){
      $.charts[containerId].addValues([[getNewData()]]);
    });

    // start looping at default 250Hz
    $('#'+containerId+' .start_auto').on('click',function(e){
      $.charts[containerId].tid = setInterval(function() { addValues($.charts[containerId],[[getNewData()]]); }, 10);
      $('#'+containerId+' .start_auto').hide();
      $('#'+containerId+' .stop_auto').show();
    });

    // stop auto-loop
    $('#'+containerId+' .stop_auto').on('click',function(e){
      abortTimer();
      $('#'+containerId+' .stop_auto').hide();
      $('#'+containerId+' .start_auto').show();
    });
}


//// add new values to the end of a chart
function addDataToChart (chart, values) {
    console.log(values[0].length)
    if((chart.datasets.length) && Array.isArray(values)) {
      // loop through each series
        chart.datasets.forEach(function (dataset, i) {
          // if there is a corresponding arrya of new values for this series
          if(Array.isArray(values[i]) && values[i].length){
            // find the boundary index which separates new from old data
            // i.e. if you have an current series with 10 points, and there
            // are 3 new data points coming in, then the logic is as follows:
            // [0] - will be removed
            // [1..7] - will be shifted left
            // [7..9] - will take the 3 new values
            var newValuesStartIndex = (dataset.points.length - values[i].length);
            (dataset.points || dataset.bars).forEach(function (dataItem, j) {
              if(j < newValuesStartIndex){
                // old values to keep
                // shift each element left by copying from right (the one ahead of it)
                // i.e. to shift element 1 back to 0, array[0] = array[0+1]
                dataItem.value = dataset.points[j+1].value;
              } else {
                // otherwise, it's a new value coming in, just copy from input
                dataItem.value = values[i].shift(); // pop the first element off the incoming array
                //console.log('values['+i+']['+j+'] = '+dataItem.value);
              }
            });
          } else {
            console.log("incoming values should be an array of arrays: values[datasetIndex][datapointValue]");
          }
        });
        // console.log(lineChart.datasets[0].points)
        chart.update();
    } else {
      console.log("incoming values should be an array of arrays");
    }
}


// settings
var optionsLine = {
    ///Boolean - Whether grid lines are shown across the chart
    scaleShowGridLines : true,
    //String - Colour of the grid lines
    scaleGridLineColor : "rgba(0,0,0,.05)",
    //Number - Width of the grid lines
    scaleGridLineWidth : 1,
    //Boolean - Whether to show horizontal lines (except X axis)
    scaleShowHorizontalLines: true,
    //Boolean - Whether to show vertical lines (except Y axis)
    scaleShowVerticalLines: true,
    //Boolean - Whether the line is curved between points
    bezierCurve : false,
    bezierCurveTension : 0,
    //Boolean - Whether to show a dot for each point
    pointDot : false,
    //Boolean - Whether to show a stroke for datasets
    datasetStroke : false,
    //Number - Pixel width of dataset stroke
    datasetStrokeWidth : 1,
    //Boolean - Whether to fill the dataset with a colour
    datasetFill : false,
};

Chart.defaults.global = {
    // Boolean - Whether to animate the chart
    animation: true,
    // Boolean - If we should show the scale at all
    showScale: true,

    // Boolean - If we want to override with a hard coded scale
    scaleOverride: false,
    // ** Required if scaleOverride is true **
    // Number - The number of steps in a hard coded scale
    scaleSteps: 2,
    // Number - The value jump in the hard coded scale
    scaleStepWidth: 100,
    // Number - The scale starting value
    scaleStartValue: -100,

    // String - Colour of the scale line
    scaleLineColor: "rgba(0,0,0,.1)",
    // Number - Pixel width of the scale line
    scaleLineWidth: 1,
    // Boolean - Whether to show labels on the scale
    scaleShowLabels: true,
    // Interpolated JS string - can access value
    scaleLabel: "<%=value%>",
    // Boolean - Whether the scale should stick to integers, not floats even if drawing space is there
    scaleIntegersOnly: true,
    // Boolean - Whether the scale should start at zero, or an order of magnitude down from the lowest value
    scaleBeginAtZero: false,

    // String - Scale label font declaration for the scale label
    scaleFontFamily: "'Helvetica Neue', 'Helvetica', 'Arial', sans-serif",
    // Number - Scale label font size in pixels
    scaleFontSize: 10,
    // String - Scale label font weight style
    scaleFontStyle: "normal",
    // String - Scale label font colour
    scaleFontColor: "#666",
    // Boolean - whether or not the chart should be responsive and resize when the browser does.
    responsive: false,
    // Boolean - whether to maintain the starting aspect ratio or not when responsive, if set to false, will take up entire container
    maintainAspectRatio: true,
    // Function - Will fire on animation progression.
    onAnimationProgress: function(){},
    // Function - Will fire on animation completion.
    onAnimationComplete: function(){}
}