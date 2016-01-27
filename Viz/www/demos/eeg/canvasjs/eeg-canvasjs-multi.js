function drawCharts(){

	$.testmodeCounter = $.testmodeCounter || [];

    // create the chart and save to global jquery scope
	for(var i=0;i< $.numChannels;i++) {
		$.charts[i] = new CanvasJS.Chart("chartContainer"+i, {
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
				minimum: $.yMin,
				maximum: $.yMax
			},
			data: [{
				type: "line",
				lineThickness: 1,
				color: chartColors[i],//"#FF0000",
				dataPoints: []
			}]
		});

		// generates first set of dataPoints
		initChart($.charts[i], $.dataLength);

		$.testmodeCounter[i] = 0;
	}
}

var initChart = function ($chart,count) {
	count = count || 1;
	// count is number of times loop runs to generate random dataPoints.
	var xval = 0;
	var yval = 0;
	for (var j = 0; j < count; j++) {

		if($.testmode=='marker' && j < ($.dataLength-10)){
			$chart.options.data[0].dataPoints.push({
				x:xval,
				y:yval
			});
		} else {
			$chart.options.data[0].dataPoints.push({
				x: xval,
				y: getNewData()
			});
		}

		xval = xval+1;

		if ($chart.options.data[0].dataPoints.length > $chart.options.dataLength) {
			$chart.options.data[0].dataPoints.shift();
		}
	};
	$chart.render();
};


var updateChart = function (chart_index, points) {
	var pointsLength = points.length;

	if ($.testmode == "marker"){
		/* testmode = marker
			in this mode we only want to update 10 points at the very end of the chart
			so we can visually track the speed of data refresh more easily as it flies by
		 */

		// add new points & remove old points
		for (var i = 0; i < pointsLength; i++) {

			if($.testmodeCounter[chart_index] > ($.dataLength-pointsLength)){
				// add one
				$.charts[chart_index].options.data[0].dataPoints.push({
					x: points[i][0],
					y: points[i][1]
				});
			} else {
				// add one
				$.charts[chart_index].options.data[0].dataPoints.push({
					x: i,
					y: 0
				});
			}

			// remove one
			$.charts[chart_index].options.data[0].dataPoints.shift();

			$.testmodeCounter[chart_index] += 1;

			if($.testmodeCounter[chart_index] > $.dataLength){
				$.testmodeCounter[chart_index] = 0;
			}
		};

		// rebase x values
		for (var j = 0; j < $.dataLength; j++) {
			$.charts[chart_index].options.data[0].dataPoints[j].x = j;
		}

	} else if (1) {
		// add new points & remove old points
		for (var i = 0; i < pointsLength; i++) {

			// remove one
			$.charts[chart_index].options.data[0].dataPoints.shift();

			// add one
			$.charts[chart_index].options.data[0].dataPoints.push({
				x: points[i][0],
				y: points[i][1]
			});
		}
		// rebase x values
		for (var j = 0; j < $.dataLength; j++) {
			$.charts[chart_index].options.data[0].dataPoints[j].x = j;
		}
	} else {
		/* testmode = continuous
			in this mode it's sequential data being sent, so we will update the complete chart
		 */

		// we will need to see how much of the chart to displace, since there could be downsampled values coming in
		// if downsampled, x points are not continuous
		// (example: for a range from 0 to 60, with 60 points downsampled 90% down to 6,
		// x could look like this: x = [ 0, 12, 15, 33, 52, 60])
		// in this case, we'll have to find the range between start and end x values, and shift that many points back
		var range = points[pointsLength-1][0] - points[0][0] + 1;
		var offset = $.dataLength - range;
		// current number of points may be less than $.dataLength if downsampling
		var currentNumPoints = $.charts[chart_index].options.data[0].dataPoints.length;

		//console.log('pointsLength: ' + pointsLength + ', range: '+ range + ', offset: ' + offset)

		// shift old x values back by offset
		for (var k = 0; k < currentNumPoints; k++) {
			//console.log(k)
			$.charts[chart_index].options.data[0].dataPoints[k].x -= range;
		}

		// add new points with x+offset
		for (var i = 0; i < pointsLength; i++) {
			// add one
			$.charts[chart_index].options.data[0].dataPoints.push({
				x: points[i][0] + offset,
				y: points[i][1]
			});
		};

		// remove old points which are now less than x=0
		for (var j = 0; j < range; j++) {
			if($.charts[chart_index].options.data[0].dataPoints[j].x < 0) {
				$.charts[chart_index].options.data[0].dataPoints.shift();
			}
		};
	}

	$.charts[chart_index].render();
}

