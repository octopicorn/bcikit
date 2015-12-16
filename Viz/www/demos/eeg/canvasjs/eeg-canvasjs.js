// Pipe - convenience wrapper to present data received from an
// object supporting WebSocket API in an html element. And the other
// direction: data typed into an input box shall be sent back.
var pipe = function(ws, el_name) {
    var el_id = '#'+el_name;
    var $connectButton = $(el_id + ' .connect');
    var $disconnectButton = $(el_id + ' .disconnect');
    //ws.onopen    = function()  { console.log('websocket OPEN');}

    ws.onmessage = function(e) {
        // get incoming data as json
        var data = JSON.parse(e.data);

		// use this when updating chart from a raw TIME_SAMPLE
		//addPoint($.charts.plot1,data.channel_1);

		// use this when updating chart from a ModuleWindows output
		// addArray($.charts.plot1,data[0])

		// use this when updating chart from a ModuleConvert (DTYPE_COORD) output
		for(var i=0;i<$.numCharts;i++){
			addCoords($.charts[i],data[i]);
		}

        //console.log(data[0])
    }
    //ws.onclose   = function()  { console.log('websocket CLOSED');};

    $connectButton.on('click', function(e){
        e.preventDefault();
        // get selected metric from dropdown
        var metric = $(el_id + ' .metricSelect').val();
        var jsonRequest = JSON.stringify({
            "type": "subscription",
            "deviceName": "openbci",
            "deviceId": "octopicorn",
            "metric": metric,
            "dataType": "MATRIX",
            "rabbitmq_address":"127.0.0.1"
        });
        ws.send(jsonRequest);
        console.log('subscribed: '+ metric);
        $connectButton.addClass('hidden');
        $disconnectButton.removeClass('hidden');
    });

    $disconnectButton.on('click', function(e){
        e.preventDefault();
        // get selected metric from dropdown
        var metric = $(el_id + ' .metricSelect').val();
        var jsonRequest = JSON.stringify({
            "type": "unsubscription",
            "deviceName": "openbci",
            "deviceId": "octopicorn",
            "metric": metric,
            "dataType": "MATRIX",
            "rabbitmq_address":"127.0.0.1"
        });
        ws.send(jsonRequest);
        console.log('unsubscribed: '+ metric);
        $disconnectButton.addClass('hidden');
        $connectButton.removeClass('hidden');
    });
};

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

var addCoords = function ($chart, points) {
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


$(document).ready(function(){

	$.xVal = 0;
	$.yVal = 0;
	var dataLength = 500; // number of dataPoints visible at any point
    $.dps = []; // dataPoints
	$.numChannels = 16;

	$.charts = [];

    // create the chart and save to global jquery scope
	for(var i=0;i< $.numChannels;i++) {
		$.charts[i] = new CanvasJS.Chart("chartContainer"+i, {
			interactivityEnabled: false,
			title: {
				text: "Channel " + (i+1)
			},
			dataLength: dataLength,
			axisX: {
				minimum: 0,
				maximum: dataLength,
				interval: dataLength
			},
			axisY: {
				minimum: -60,
				maximum: 60
			},
			data: [{
				type: "line",
				dataPoints: $.dps
			}]
		});

		// generates first set of dataPoints
		initChart($.charts[i], dataLength);
	}
	// keep the num of charts for easy for looping
	$.numCharts = $.charts.length;

    // declare
    var sockjs_url = '/echo';
    var sockjs = new SockJS(sockjs_url);

    var multiplexer = new MultiplexedWebSocket(sockjs);
    var ann  = multiplexer.channel('ann');
    pipe(ann,  'first');



});