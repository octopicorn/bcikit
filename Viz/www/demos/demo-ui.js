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
            //console.log("incoming handshake from server");
            //console.log(data)
            updateMetricsDropDown(data.metrics)
            if(data.num_channels){
                updateNumChannelsDropDown(data.num_channels)
            }
            $('#controls').removeClass('hidden');

        } else {
            //console.log(data) // debug
            // incoming plot data
            for(var i=0;i<$.numChannels;i++){
                updateChart(i,data[i]);
            }
        }
    }
};

// random value generator (pick between -10 and 10)
function getNewData(channel_index){
    //channel_index = channel_index || 0;
    // add the offset specific to the channel to the random value
    // for example, if it's channel 3, we will add offset * (3+1)
    return Math.floor(Math.random() * 91) - 50; // + ($.offsetPerChannel * (channel_index+1));
}

function updateNumChannelsDropDown(numChannels){
    // get the <select> element
    $numChannelsSelect = $('#numChannelsSelect');
    // remove all options
    $numChannelsSelect
        .find('option')
        .remove();
    // append the incoming metrics from the handshake
    for (var i=1;i<=numChannels;i++){
        $numChannelsSelect
            .append('<option value="' + i + '">'+ i+'</option>');
    }
    // select the first one by default (1 channel)
    $numChannelsSelect
        .val(1)
        .change();
}

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

var chartColors = [
'#ff0000',
'#00ccff',
'#66ff00',
'#228800',
'#ff6600',
'#00ff00',
'#EA5532',
'#6600ff',
'#B72F8C',
'#00ff66',
'#30475B',
'#cc00ff',
'#00ffcc',
'#0066ff',
'#0000ff',
'#00AB84'];

$(document).ready(function(){

    // websocket related
    var sockjs_url = '/echo';
    var sockjs = new SockJS(sockjs_url);
    $.multiplexer = new MultiplexedWebSocket(sockjs);
    $.websockets = {};

    $.testmode = 'continuous';
    $.charts = [];
    $.dataLength = parseInt($('#dataLength').val());
    $.offsetPerChannel = 0;
    $.numChannels = 1;

    // bind numChannels dropdown
    $('#numChannelsSelect').change(function(){
        $.numChannels = parseInt($(this).val());
        drawCharts();
    });

    // bind testmode
    $('input[type=checkbox][name=testmode]').on('change',function(){
        if($(this).is(':checked')){
            $.testmode = 'marker';
        } else {
            $.testmode = 'continuous';
        }
        drawCharts();
    });

    // bind datalength
    $('#dataLength').on('change', function(e){
        // update datalength global
        $.dataLength = parseInt($(this).val());
        // if chart is drawn already, re-initialize it
        if($('#numChannelsSelect').val() != ''){
            initCharts();
        }
    });

    // bind TEST connect button
    $('#starttest').on('click', function(e){
        if($.numChannels != "") {
            e.preventDefault();

            var fps = parseInt($('#fps').val());
            var timeToSleep = parseInt(1000 / fps);

            $('#stoptest').show();
            $('#starttest').hide();

            console.log('connect: looping at ' + fps + ' times per sec. will sleep ' + timeToSleep + ' msec between each loop');
            $.tid = setInterval(randomUpdate, timeToSleep);
        }
    });

    // bind TEST disconnect button
    $('#stoptest').on('click', function(e){
        $('#starttest').show();
        $('#stoptest').hide();
        e.preventDefault();
        clearInterval($.tid);
    });

    var $connectButton = $('#controls button.connect');
    var $disconnectButton = $('#controls button.disconnect');

    $connectButton.on('click', function(e){
        e.preventDefault();
        // get websocket index
        var ws = $.websockets[$(this).data('websocketIndex')];
        // get selected metric from dropdown
        var metric = $('#metricSelect').val();
        var jsonRequest = JSON.stringify({
            "type": "subscription",
            "deviceName": "openbci",
            "deviceId": $('#metricName').val(),
            "metric": metric,
            "dataType": "MATRIX",
            "rabbitmq_address":"127.0.0.1"
        });
        // send request to server across the websocket
        ws.send(jsonRequest);
        console.log('subscribed to metric: '+ metric);

        // show/hide connect buttons
        $(this).addClass('hidden');
        $disconnectButton.removeClass('hidden');
    });

    $disconnectButton.on('click', function(e){
        e.preventDefault();
        // get websocket index
        var ws = $.websockets[$(this).data('websocketIndex')];
        // get selected metric from dropdown
        var metric = $('#metricSelect').val();
        var jsonRequest = JSON.stringify({
            "type": "unsubscription",
            "deviceName": "openbci",
            "deviceId": $('#metricName').val(),
            "metric": metric,
            "dataType": "MATRIX",
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