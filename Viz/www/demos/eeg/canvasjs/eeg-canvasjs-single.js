function drawCharts(){
	//console.log($.dataLength)
	// create the chart and save to global jquery scope
	$.charts[0] = new CanvasJS.Chart("chartContainer0", {
		interactivityEnabled: false,
		title: {
			text: "Channels "
		},
		dataLength: $.dataLength,
		axisX: {
			minimum: 0,
			maximum: $.dataLength,
			interval: $.dataLength
		},
		axisY: {
			minimum: $.offsetPerChannel/2,
			maximum: ($.numChannels * $.offsetPerChannel) + $.offsetPerChannel/2,
			interval: $.offsetPerChannel
		},
		data: getChartInitDatasets()
	});

	// draw chart
	$.charts[0].render();
	//console.log($.charts[0].options.data[0].dataPoints)

}

// generates first set of dataPoints
function getChartInitDatasets(){
	var datasets = [];
	for(var i=0; i<$.numChannels;i++){
		datasets.push({
			type: "line",
			lineThickness: 1,
			color: "#FF0000",
			dataPoints: getInitData($.dataLength, i)
		});
	}
	return datasets;
}

var getInitData = function (count, channel_index) {
	count = count || 1;
	// count is number of times loop runs to generate random dataPoints.
	var data = [];
	for (var i = 0; i < count; i++) {
		//yval = yval +  Math.round(5 + Math.random() *(-5-5)); // uncomment to init with random values
		data.push({	x: i, y: getNewData(channel_index) });
	};
	return data;
};

var addCoordsWindow = function (chart_index, points) {
	var pointsLength = points.length;
	var oldLength = $.charts[0].options.data[chart_index].dataPoints.length;
	for (var i = 0; i < pointsLength; i++) {
		$.charts[0].options.data[chart_index].dataPoints.shift();
		$.charts[0].options.data[chart_index].dataPoints.push({
			x: points[i][0],
			y: points[i][1] + ($.offsetPerChannel * (chart_index+1))
		});
	};
	//for (var i = 0; i<oldLength; i++){
	//	$.charts[0].options.data[chart_index].dataPoints.shift();
	//}
	//console.log($.dps)
	$.charts[0].render();
};



