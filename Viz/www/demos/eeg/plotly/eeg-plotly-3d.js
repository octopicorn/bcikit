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
		// use this when updating chart from a ModuleConvert (DTYPE_COORD) output
		for(var i=0;i<$.numCharts;i++){
			addCoords(i,data[i]);
		}
    }

    // add one data at a time manually
    $(el_id + ' .add').on('click',function(e){
        drawRandomData(0);
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

var getNewUpdateObject = function(x,y,z){
    return {
        x:x, y:y, z:z,
        mode:'lines',
        hoverinfo:'none',
        projection: {
            x: { show: true },
            y: { show: true },
            z: { show: 0 }
        },
        line: {
            color: 'rgb(0, 0, 150)',
            width: 1.5
        },
        type: 'scatter3d'
    };
}

var getRandomData = function (count) {
	count = count || 1;
	// count is number of times loop runs to generate random dataPoints.
	var xval = 0;
    var x = [];
    var y = [];
    var z = [];
	for (var j = 0; j < count; j++) {
		x.push(xval++);
        y.push(getNewData());
        z.push(0);
	};
    return getNewUpdateObject(x,z,y);
};

var drawRandomData = function (chart_index) {
    var plotDiv = document.getElementById('chartContainer'+chart_index);
    // overwrite old data with the new data
    plotDiv.data[0] = getRandomData($.dataLength)
	Plotly.redraw(plotDiv);
};

var addCoords = function(chart_index, points){
    var pointsLength = points.length;
    var x = [];
    var y = [];
    var z = [];
	for (var i = 0; i < pointsLength; i++) {
		x.push(points[i][0]);
        y.push(points[i][1]);
        z.push(0);
	};
    var plotDiv = document.getElementById('chartContainer'+chart_index);
	// overwrite old data with the new data
    plotDiv.data[0] = getNewUpdateObject(x,z,y);
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
            margin: { l: 0,r: 0, b: 0, t: 0 },
            autosize: false,
            width: 600,
            height: 300,
            scene: {
                camera: {
                    //center: {x: 0, y: 0, z: 0}
                    eye: {x: 0, y:-1, z:0}
                },
                aspectmode: "manual",
                aspectratio: {x: 1.5, y: 0.01, z:0.5}
            },
            xaxis: {
                fixedrange: true,
                showspikes: false,
                spikesides: false,
            },
            yaxis: {
                autorange: true
            },
            zaxis: {
                fixedrange: true,
                autorange: false,
                range: [-80,80]
            }


        };

		$.charts[i] = Plotly.newPlot('chartContainer'+i, data, layout, {displayModeBar: false, scrollZoom: false});
	}

    var plotDiv = document.getElementById('chartContainer0');
    // overwrite old data with the new data
    console.log(plotDiv.layout)

	// keep the num of charts for easy for looping
	$.numCharts = $.charts.length;

    // declare
    var sockjs_url = '/echo';
    var sockjs = new SockJS(sockjs_url);

    var multiplexer = new MultiplexedWebSocket(sockjs);
    var ann  = multiplexer.channel('ann');
    pipe(ann,  'first');

});