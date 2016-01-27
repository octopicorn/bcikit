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
		// use this when updating chart from a Convert (DTYPE_COORD) output
		for(var i=0;i<$.numCharts;i++){
			addCoords(i,data[i]);
		}
    }

    // add one data at a time manually
    $(el_id + ' .add').on('click',function(e){
        drawRandomData($.charts[0]);
    });

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
            "rabbitmq_address":"127.0.0.1"
        });
        ws.send(jsonRequest);
        console.log('unsubscribed: '+ metric);
        $disconnectButton.addClass('hidden');
        $connectButton.removeClass('hidden');
    });
};

// random value generator
var getNewData = function(){
    return Math.floor(Math.random() * 71) - 35;
}

var getNewUpdateObject = function(x,y){
    return {x:x, y:y, mode:'lines', hoverinfo:'none', line: { color: 'rgb(128, 0, 128)', width: 1 }};
}

var getRandomData = function (count) {
	count = count || 1;
	// count is number of times loop runs to generate random dataPoints.
	var xval = 0;
    var x = [];
    var y = [];
	for (var j = 0; j < count; j++) {
		x.push(xval++);
        y.push(getNewData());
	};
    return getNewUpdateObject(x,y);
};

var drawRandomData = function ($chart) {
    var plotDiv = document.getElementById('chartContainer0');
    // erase the old data
    Plotly.deleteTraces(plotDiv, 0);
    // draw the new data
    plotDiv.data.push(getRandomData($.dataLength))
	Plotly.redraw(plotDiv);
};

var addCoords = function(chart_index, points){
    var pointsLength = points.length;
    var x = [];
    var y = [];
	for (var i = 0; i < pointsLength; i++) {
		x.push(points[i][0]);
        y.push(points[i][1]);
	};
    var plotDiv = document.getElementById('chartContainer'+chart_index);
	// erase the old data
    Plotly.deleteTraces(plotDiv, 0);
    // draw the new data
    plotDiv.data.push(getNewUpdateObject(x,y));
	Plotly.redraw(plotDiv);
}


$(document).ready(function() {

	$.dataLength = 500; // number of dataPoints visible at any point
	$.numChannels = 16;
	$.charts = [];

    // create the chart and save to global jquery scope
	for(var i=0;i< $.numChannels;i++) {

        var data = [getRandomData($.dataLength)];

        var layout = {
            title: 'Channel ' + (i+1),
            titlefont:{ size: 12 },
            margin: { l: 25,r: 5, b: 0, t: 20 },
            xaxis: {
                fixedrange: false
            },
            yaxis: {
                fixedrange: true,
                autorange: false,
                range: [-500,3000]
            }
        };

		$.charts[i] = Plotly.newPlot('chartContainer'+i, data, layout, {displayModeBar: false, scrollZoom: false});

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