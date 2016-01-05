// random value generator
function getNewData(){
    return Math.floor(Math.random() * 21) - 10;
}

function updateCoords(chart, points) {
    var pointsLength = points.length;
    var newData = [];
    var nextPoint = null;
	for (var i = 0; i < pointsLength; i++) {
        nextPoint = points[i];
        newData.push(points[i][1]);
    }
    // add new
    chart.data.datasets[0].data = newData;
    // redraw
    chart.update();
}

// to be called when you want to stop the timer
// only used for the dummy data looping function
function abortTimer() {
  clearInterval(tid);
}

// Pipe - convenience wrapper to present data received from an
// object supporting WebSocket API in an html element. And the other
// direction: data typed into an input box shall be sent back.
var pipe = function(ws, el_name) {
    var el_id = '#'+el_name;
    var $connectButton = $(el_id + ' .connect');
    var $disconnectButton = $(el_id + ' .disconnect');
    //ws.onopen    = function()  { console.log('websocket OPEN');}

    var bufferData = [];
    var bufferSize = 1;

    ws.onmessage = function(e) {

        // get incoming data as json
        var data = JSON.parse(e.data);
        //updatePlot($.charts.plot1,[data.channel_1]);


        // put it in buffer
        //bufferData.push(data.channel_1);

        //if(bufferData.length >= bufferSize){
        //    updatePlot($.charts.plot1, bufferData);
        //    //console.log(bufferData);
        //    bufferData = [];
        //}
        for(var i=0;i<$.numChannels;i++){
            updateCoords($.charts[i], data[i]);
		}

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

var getRandomData = function(length){
    var data = {
                    labels: [],
                    datasets: [{
                        label: "",
                        data: [],
                        fill: false,
                        borderColor: "#000000",
                    }]
                };
    for(var i=0;i< $.dataLength;i++){
        data.datasets[0].data.push(getNewData());
        data.labels.push(getNewData());
    }
    return data;
}

$(document).ready(function(){
    $.charts = {};
    $.numChannels = 3;
    $.dataLength = 500;

    // globals
    // set default window dimensions


    var config = {
        scales: {
            xAxes: [{
              display: true,
                gridlines: {
                    display:false,
                    drawTicks: false,
                    lineWidth: 0.5
                },
              ticks: {
                  userCallback: function(dataLabel, index) {
                      return '';
                  }
              }
            }],
            yAxes: [{
                gridlines: {
                    display:false,
                    drawTicks: false,
                    lineWidth: 0.5
                },
              display: true,
              ticks: {
                  display: false,
                  suggestedMin: -60,
                  suggestedMax: 60

              }
            }]
        },
        responsive: true,
        responsiveAnimationDuration: 0,
        animation: {
            duration: 0,
            easing: "easeOutQuart",
            onProgress: function() {},
            onComplete: function() {},
        },
        tooltips:{
          enabled: false,
          custom: null
        }
    };

    Chart.defaults.global.elements.line.tension = 0;
    Chart.defaults.global.elements.line.borderWidth = 1;
    Chart.defaults.global.elements.point.radius = 0;

    for(var i=0;i< $.numChannels;i++) {
        // create the chart and save to global jquery scope
        ctx = document.getElementById("chartContainer"+i);
        $.charts[i] = new Chart(ctx, {type: 'line', data: getRandomData(), options: config});
    }

    // declare
    var sockjs_url = '/echo';
    var sockjs = new SockJS(sockjs_url);

    var multiplexer = new MultiplexedWebSocket(sockjs);
    var ann  = multiplexer.channel('ann');
    pipe(ann,  'first');


});
