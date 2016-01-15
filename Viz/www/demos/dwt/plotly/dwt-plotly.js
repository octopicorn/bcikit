var getRandomData = function (rows,columns) {
	// count is number of times loop runs to generate random dataPoints.
	var data = [];
    var row;
	for (var i = 0; i < columns; i++) {
        row = [];
        for (var j = 0; j < rows; j++) {
            row.push(getNewData());
        }
        data.push(row);
	};
    return data;
};

var randomUpdate = function(){

}

//var drawTest = function ($chart) {
//    var plotDiv = document.getElementById('chartContainer0');
//    // erase the old data
//    Plotly.deleteTraces(plotDiv, 0);
//    // draw the new data
//    plotDiv.data.push(getRandomData($.dataLength))
//	Plotly.redraw(plotDiv);
//};
//


var randomUpdate = function () {

    for(var i=0;i< $.numChannels;i++) {

        var plotDiv = document.getElementById('chartContainer' + i);

        // erase the old data
        Plotly.deleteTraces(plotDiv, 0);

        // draw the new data
        plotDiv.data = [
            {
                z: getRandomData($.chart_rows,$.chart_cols),
                colorscale: $.colorscale,
                type: 'heatmap',
                hoverinfo: 'none',
                scrollZoom: false
            }
        ];

        Plotly.redraw(plotDiv);
    }
}
//var addCoords = function(chart_index, points){
//    var pointsLength = points.length;
//    var x = [];
//    var y = [];
//	for (var i = 0; i < pointsLength; i++) {
//		x.push(points[i][0]);
//        y.push(points[i][1]);
//	};
//    var plotDiv = document.getElementById('chartContainer'+chart_index);
//	// erase the old data
//    Plotly.deleteTraces(plotDiv, 0);
//    // draw the new data
//    plotDiv.data.push(getNewUpdateObject(x,y));
//	Plotly.redraw(plotDiv);
//}


var drawCharts = function(){

    $.chart_rows = 25;
    $.chart_cols = 30;

    $.colorscale = 'Portland';
    //colorscale: 'Jet',
    //colorscale: 'Portland',
    //colorscale: 'Greys',
    //colorscale: 'Picnic',
    // more here: https://plot.ly/javascript/heatmap-and-contour-colorscales/

    // create the chart and save to global jquery scope
	for(var i=0;i< $.numChannels;i++) {

        var layout = {
            title: 'Channel ' + (i+1),
            titlefont:{ size: 12 },
            margin: { l: 25,r: 5, b: 0, t: 20 },
            yaxis: {
              showgrid: false,
              zeroline: false,
              showticklabels: false,
              ticks: ''
            }
        };

        var data = [
          {
            z: getRandomData($.chart_rows,$.chart_cols),
            colorscale: $.colorscale,
            type: 'heatmap',
            hoverinfo:'none',
            scrollZoom: false
          }
        ];

		$.charts[i] = Plotly.newPlot('chartContainer'+i, data, layout, {displayModeBar: false, scrollZoom: false});

	}

}