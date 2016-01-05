function drawCharts(){

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
			interval: 10
		},
		axisY: {
			minimum: -10,
			maximum: 10
		},
		data: getChartInitDatasets()
	});

	// draw chart
	//$.charts[0].render();
	console.log($.charts[0])

}

// generates first set of dataPoints
function getChartInitDatasets(){
	var datasets = [];
	for(var i=0; i<$.numChannels;i++){
		datasets.push({
			type: "line",
			lineThickness: 1,
			color: "#FF0000",
			dataPoints: []//getInitData($.dataLength)
		});
	}
	return datasets;
}

var getInitData = function (count) {
	count = count || 1;
	// count is number of times loop runs to generate random dataPoints.
	var data = [];
	for (var i = 0; i < count; i++) {
		//yval = yval +  Math.round(5 + Math.random() *(-5-5)); // uncomment to init with random values
		data.push({	x: i, y: 0 });
	};

	return data;
};

var addArray = function ($chart, points) {
	var pointsLength = points.length;
	for (var i = 0; i < pointsLength; i++) {
		$.dps.push({
			x: $.xVal,
			y: points[i]
		});
		$.xVal = $.xVal+1;

		if ($.dps.length > $chart.options.dataLength) {
			$.dps.shift();
		}
	};
	$chart.render();
};

var addCoordsWindow = function ($chart, points) {
	var pointsLength = points.length;
	var oldLength = $.dps.length
	for (var i = 0; i < pointsLength; i++) {
		$.dps.push({
			x: points[i][0],
			y: points[i][1]
		});
	};
	for (var i = 0; i<oldLength; i++){
		$.dps.shift();
	}
	//console.log($.dps)
	$chart.render();
};

var addPoint = function ($chart, value) {
	$.dps.push({
		x: $.xVal,
		y: value
	});
	$.xVal = $.xVal+1;

	if ($.dps.length > $chart.options.dataLength) {
		$.dps.shift();
	}
	$chart.render();
};


