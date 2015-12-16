// random value generator
function getNewData(){
    return Math.floor(Math.random() * 21) - 10;
}

// globals
// set default window dimensions
var defaultDataLength = 1000;
var defaultData = [];
var defaultLabels = [];
for(var i=0;i<defaultDataLength;i++){
  defaultData.push(getNewData());
  defaultLabels.push(getNewData());
}


var config = {
    type: 'line',
    data: {
        labels: defaultLabels,
        datasets: [{
            label: "My First dataset",
            data: defaultData,
            fill: true,
        }]
    },
    options: {

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
                      //return index % 2 === 0 ? dataLabel : '';
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
        responsive: false,
        responsiveAnimationDuration: 0,
        animation: {
            duration: 0,
            easing: "easeOutQuart",
            onProgress: function() {},
            onComplete: function() {},
        },
        line: {
            tension: 0,
            backgroundColor: Chart.defaults.global.defaultColor,
            borderWidth: 0,
            borderColor: Chart.defaults.global.defaultColor,
            borderCapStyle: 'butt',
            borderDash: [],
            borderDashOffset: 0.0,
            borderJoinStyle: 'miter',
            fill: false, // do we fill in the area between the line and its base axis
            skipNull: true,
            drawNull: false,
        },
        point: {
            display: 0,
            radius: 0,
            backgroundColor: Chart.defaults.global.defaultColor,
            borderWidth: 0,
            borderColor: Chart.defaults.global.defaultColor,
            // Hover
            hitRadius: 0,
            hoverRadius: 0,
            hoverBorderWidth: 0,
        },
        tooltips:{
          enabled: false,
          custom: null
        }
    }
};

function addData (chart, values) {
  var maxlen = defaultDataLength;
  if (config.data.datasets.length > 0) {
      //config.data.labels.push('dataset #' + config.data.labels.length);

      $.each(config.data.datasets, function(i, dataset) {
          dataset.data.push(values);
          if(dataset.data.length > maxlen){
            dataset.data.shift();
          }
      });
      chart.update();
  }
}

function updatePlot(chart, newValues) {
    //var chartData = chart.data.datasets[0].data;

    // add new
    chart.data.datasets[0].data = chart.data.datasets[0].data.concat(newValues);
    // chop off the old
    chart.data.datasets[0].data.splice(0, newValues.length);
    // redraw
    chart.update();
}

function updateCoords(chart, points) {

    var newData = Array(1000).fill(null);
    var pointsLength = points.length;
    var nextPoint = null;
	for (var i = 0; i < pointsLength; i++) {
        nextPoint = points[i];
        newData[points[i][0]] = points[i][1];
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

        updateCoords($.charts.plot1, data[0]);
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


$(document).ready(function(){
    $.charts = {};

    // create the chart and save to global jquery scope
    var ctx = document.getElementById("placeholder");
    $.charts.plot1 = new Chart(ctx, config);

    console.log($.charts.plot1)

    // default 250Hz
    //var tid = setInterval(function() { addData($.charts.plot1,getNewData()); }, 10);


    // add one data at a time manually
    var $oneMoreButton = $('.add');
    $oneMoreButton.on('click',function(e){
        addData($.charts.plot1,getNewData());
        console.log($.charts.plot1)
    });

    // declare
    var sockjs_url = '/echo';
    var sockjs = new SockJS(sockjs_url);

    var multiplexer = new MultiplexedWebSocket(sockjs);
    var ann  = multiplexer.channel('ann');
    pipe(ann,  'first');


});
