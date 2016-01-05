function initCharts(){
    $.dps = []; // dataPoints


	// y-axis labels
        // number of ticks
        Chart.defaults.global.scaleSteps = parseInt($.numChannels);
        // how big is space between each tick
        Chart.defaults.global.scaleStepWidth = $.offsetPerChannel;
        // starting value
        Chart.defaults.global.scaleStartValue = $.offsetPerChannel/2;

    // create the chart and save to global jquery scope
	for(var i=0;i< $.numChannels;i++) {
		$.charts[0] = new CanvasJS.Chart("chartContainer"+i, {
			interactivityEnabled: false,
			title: {
				text: "Channel " + (i+1)
			},
			dataLength: $.dataLength,
			axisX: {
				minimum: 0,
				maximum: $.dataLength,
				interval: $.dataLength
			},
			axisY: {
				minimum: -60,
				maximum: 60
			},
			data: [{
				type: "line",
				lineThickness: 1,
				color: "#FF0000",
				dataPoints: $.dps
			}]
		});

		// generates first set of dataPoints
		initChart($.charts[i], $.dataLength);
	}
	// keep the num of charts for easy for looping
	$.numCharts = $.charts.length;

	openPipe();
}

// random value generator
function getNewData(){
    return Math.floor(Math.random() * 21) - 10;
}

var initChart = function ($chart,count) {
	count = count || 1;
	// count is number of times loop runs to generate random dataPoints.
	var xval = 0;
	var yval = 0;
	for (var j = 0; j < count; j++) {
		//yval = yval +  Math.round(5 + Math.random() *(-5-5)); // uncomment to init with random values
		$chart.options.data[0].dataPoints.push({
			x: xval,
			y: yval
		});
		xval = xval+1;

		if ($chart.options.data[0].dataPoints.length > $chart.options.dataLength) {
			$chart.options.data[0].dataPoints.shift();
		}
	};
	$chart.render();
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


