// main
var drawCharts = function(){
    $.testmodeCounter = $.testmodeCounter || [];

    // create the chart and save to global jquery scope
	for(var i=0;i< $.numChannels;i++) {

        $.charts[i] = $.plot("#chartContainer"+i, [getRandoms($.dataLength)], {
            canvas: true,
            series: {
                shadowSize: 0,
                color: chartColors[i],
                lines: {
                    lineWidth: 1
                }
            },
            yaxis: {
                min: -1000,
                max: 4000
            },
            xaxis: {show: false}
        });

		$.testmodeCounter[i] = 0;
	}
}

var updateChart = function (chart_index, points) {

    var pointsLength = points.length;

    // get access to existing data points
    //var plotDatas = plot.getData();
    var plotData = $.charts[chart_index].getData()[0].data;

	if ($.testmode == "marker"){
		/* testmode = marker
			in this mode we only want to update 10 points at the very end of the chart
			so we can visually track the speed of data refresh more easily as it flies by
		 */

		// add new points & remove old points
		for (var i = 0; i < pointsLength; i++) {

			if($.testmodeCounter[chart_index] > ($.dataLength-pointsLength)){
				// add one
				plotData.push([points[i][0],points[i][1]]);
			} else {
				// add one
				plotData.push([i,0]);
			}

			// remove one
			plotData.shift();

			$.testmodeCounter[chart_index] += 1;

			if($.testmodeCounter[chart_index] > $.dataLength){
				$.testmodeCounter[chart_index] = 0;
			}
		};

		// rebase x values
		for (var j = 0; j < $.dataLength; j++) {
			plotData[j][0] = j;
		}

	} else {
		/* testmode = continuous
			in this mode it's sequential data being sent, so we will update the complete chart
		 */

		// we will need to see how much of the chart to displace, since there could be downsampled values coming in
		// if diwnsampled, x points are not continuous
		// (example: for a range from 0 to 60, with 60 points downsampled 90% down to 6,
		// x could look like this: x = [ 0, 12, 15, 33, 52, 60])
		// in this case, we'll have to find the range between start and end x values, and shift that many points back
		var range = points[pointsLength-1][0] - points[0][0] + 1;
		var offset = $.dataLength - range;
		// current number of points may be less than $.dataLength if downsampling
		var currentNumPoints = plotData.length;

		//console.log('pointsLength: ' + pointsLength + ', range: '+ range + ', offset: ' + offset + ', currentNumPoints: ' + currentNumPoints)

		// shift old x values back by offset
		for (var k = 0; k < currentNumPoints; k++) {
			//console.log(k)
			plotData[k][0] -= range;
		}

		// add new points with x+offset
		for (var i = 0; i < pointsLength; i++) {
			// add one
			plotData.push([points[i][0] + offset, points[i][1]]);
		};

		// remove old points which are now less than x=0
		for (var j = 0; j < range; j++) {
            if(plotData[0][0] < 0) {
				plotData.shift();
			}
		};
	}

    // refresh chart
    $.charts[chart_index].setData([plotData]);
    $.charts[chart_index].draw();

    // legacy
    //// add new
    //plotData = plotData.concat(newValues);
    //// chop off the old
    //plotData.splice(0, newValues.length);
    //// recalculate x values
    //for(var i=0; i<plotData.length; i++){
    //    plotData[i][0] = i;
    //}
}

// get n number of random points as [0,n] coordinate pairs
var getRandoms = function(totalPoints){
    data = [];
    x = 0;
    while (data.length < totalPoints) {
        if($.testmode=='marker' && x < ($.dataLength-10)){
			data.push([x,0]);
		} else {
            data.push([x, getNewData()]);
        }
        x = x+1;
    }
    return data;
}
