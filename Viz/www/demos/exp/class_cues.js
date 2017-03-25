var drawCharts = function() {
    // generates first set of dataPoints
    initChart();
}

var initChart = function () {
    // hide everything
    $('#cue-left').addClass('hidden');
    $('#cue-right').addClass('hidden');
};


var updateCueDisplay = function (data) {
    // sample data packet:  { class:0,metric:"viz_class_cues",timestamp:1477381352070970}
    //console.log('incoming class',data.class)
    switch (data.class){
        case 0:
            //
            $('#cue-message').text("+");
            $('#cue-left').addClass('hidden');
            $('#cue-right').addClass('hidden');
            break;
        case 2:
            $('#cue-message').text("+");
            $('#cue-left').removeClass('hidden');
            $('#cue-right').addClass('hidden');
            //
            break;
        case 3:
            $('#cue-message').text("+");
            $('#cue-left').addClass('hidden');
            $('#cue-right').removeClass('hidden');
            //
            break;
    }
}


// Pipe - convenience wrapper to present data received from an
// object supporting WebSocket API in an html element. And the other
// direction: data typed into an input box shall be sent back.
var pipe = function(ws) {
    ws.onopen    = function()  { console.log('websocket OPEN');}
    ws.onclose   = function()  { console.log('websocket CLOSED');};

    ws.onmessage = function(e) {
        // get incoming data as json
        var data = JSON.parse(e.data);

        if(data['type'] && data['type'] == 'handshake'){
            // variables attached to handshake will be snake_cased
            console.log("incoming handshake from server");
            //console.log(data)
            updateMetricsDropDown(data.metrics)
            $('#controls').removeClass('hidden');

        } else {
            //console.log(data) // debug
            updateCueDisplay(data)
            // incoming plot data
            //for(var i=0;i<$.numChannels;i++){
            //    updateChart(i,data[i]);
            //}
        }
    }
};





function updateMetricsDropDown(metrics){
    // get the <select> element
    $metricSelect = $('#metricSelect');
    // remove all options
    $metricSelect
        .find('option')
        .remove();
    // append the incoming metrics from the handshake
    for (i in metrics){
        $metricSelect
            .append('<option value="' + metrics[i] + '">'+metrics[i]+'</option>');
    }
}

function initCharts(){
    drawCharts();
}


$(document).ready(function(){
    initCharts();
    // websocket related
    var sockjs_url = '/echo';
    var sockjs = new SockJS(sockjs_url);
    $.multiplexer = new MultiplexedWebSocket(sockjs);
    $.websockets = {};

    var $connectButton = $('#controls button.connect');
    var $disconnectButton = $('#controls button.disconnect');

    function connect(ws,metric,jsonRequest){
        // send request to server across the websocket
        ws.send(jsonRequest);
        console.log('subscribed to metric: '+ metric);


        // show/hide connect buttons
        $connectButton.addClass('hidden');
        $disconnectButton.removeClass('hidden');
    }

    $connectButton.on('click', function(e){
        e.preventDefault();
        // get websocket index
        var ws = $.websockets[$(this).data('websocketIndex')];
        // get selected metric from dropdown
        var metric = $('#metricSelect').val();
        var jsonRequest = JSON.stringify({
            "type": "subscription",
            "deviceName": $('#deviceTypeSelect').val(),
            "deviceId": $('#metricName').val(),
            "metric": metric,
            "dataType": "TIME",
            "rabbitmq_address":"127.0.0.1"
        });

        // hide cues
        $('#cue-left').fadeOut( "slow");
        $('#cue-right').fadeOut( "slow");

        // start countdown
        var timer = 10;
        var interval = setInterval(function(){
            $('#cue-message').text(timer);
            if (--timer < 0) {
                clearInterval(interval);
                // place center point indicator
                $('#cue-message').text("+");
                connect(ws,metric,jsonRequest);
            }
        },1000)

    });

    $disconnectButton.on('click', function(e){
        e.preventDefault();
        // get websocket index
        var ws = $.websockets[$(this).data('websocketIndex')];
        // get selected metric from dropdown
        var metric = $('#metricSelect').val();
        var jsonRequest = JSON.stringify({
            "type": "unsubscription",
            "deviceName": $('#deviceTypeSelect').val(),
            "deviceId": $('#metricName').val(),
            "metric": metric,
            "dataType": "TIME",
            "rabbitmq_address":"127.0.0.1"
        });
        // send request to server across the websocket
        ws.send(jsonRequest);
        console.log('unsubscribed: '+ metric);

        // show/hide connect buttons
        $(this).addClass('hidden');
        $connectButton.removeClass('hidden');
    });



    // once all the bindings are set, let's open up the websocket connection
    var websocketReadConnectionIndex = 'a';
    $.websockets[websocketReadConnectionIndex] = $.multiplexer.channel('ann');
    // assign the websocket connection to the "connect" and "disconnect" buttons
    $connectButton.data('websocketIndex',websocketReadConnectionIndex);
    $disconnectButton.data('websocketIndex',websocketReadConnectionIndex);
    // further setup of message actions for the pipe
    pipe($.websockets[websocketReadConnectionIndex]);

});

