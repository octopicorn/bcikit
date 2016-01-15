// Pipe - convenience wrapper to present data received from an
// object supporting WebSocket API in an html element. And the other
// direction: data typed into an input box shall be sent back.
var pipe = function(ws, el_name) {
    var el_id = '#'+el_name;
    var $connectButton = $(el_id + ' .connect');
    var $disconnectButton = $(el_id + ' .disconnect');
    ws.onopen    = function()  { console.log('websocket OPEN');}
    ws.onclose   = function()  { console.log('websocket CLOSED');};

    ws.onmessage = function(e) {
        // get incoming data as json
        var data = JSON.parse(e.data);

        if(data['type'] && data['type'] == 'handshake'){
            //console.log("incoming handshake from server");
            //console.log(data)
            updateMetricsDropDown(data.metrics)
        } else {
            // incoming plot data
            for(var i=0;i<$.numChannels;i++){
                updateChart(i,data[i]);
            }
        }
    }

    var updateMetricsDropDown = function(metrics){
        // get the <select> element
        $metrics = $(el_id + ' .metricSelect');
        // remove all options
        $metrics
            .find('option')
            .remove();
        // append the incoming metrics from the handshake
        for (i in metrics){
            $metrics
                .append('<option value="' + metrics[i] + '">'+metrics[i]+'</option>');
        }
        // make the metrics/connect display visible
        $(el_id).removeClass('hidden');
    }

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
        var foo = ws.send(jsonRequest);
        console.log('subscribed to metric: '+ metric);
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
'#00AB84']

/**
 * HSV to RGB color conversion
 *
 * H runs from 0 to 360 degrees
 * S and V run from 0 to 100
 *
 * Ported from the excellent java algorithm by Eugene Vishnevsky at:
 * http://www.cs.rit.edu/~ncs/color/t_convert.html
 */
function hsvToRgb(h, s, v) {
	var r, g, b;
	var i;
	var f, p, q, t;

	// Make sure our arguments stay in-range
	h = Math.max(0, Math.min(360, h));
	s = Math.max(0, Math.min(100, s));
	v = Math.max(0, Math.min(100, v));

	// We accept saturation and value arguments from 0 to 100 because that's
	// how Photoshop represents those values. Internally, however, the
	// saturation and value are calculated from a range of 0 to 1. We make
	// That conversion here.
	s /= 100;
	v /= 100;

	if(s == 0) {
		// Achromatic (grey)
		r = g = b = v;
		return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
	}

	h /= 60; // sector 0 to 5
	i = Math.floor(h);
	f = h - i; // factorial part of h
	p = v * (1 - s);
	q = v * (1 - s * f);
	t = v * (1 - s * (1 - f));

	switch(i) {
		case 0:
			r = v;
			g = t;
			b = p;
			break;

		case 1:
			r = q;
			g = v;
			b = p;
			break;

		case 2:
			r = p;
			g = v;
			b = t;
			break;

		case 3:
			r = p;
			g = q;
			b = v;
			break;

		case 4:
			r = t;
			g = p;
			b = v;
			break;

		default: // case 5:
			r = v;
			g = p;
			b = q;
	}

	return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function componentToHex(c) {
    var hex = c.toString(16);
    return hex.length == 1 ? "0" + hex : hex;
}

function rgbToHex(r, g, b) {
    return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
}

function randomColors(total)
{
    var i = 360 / (total - 1); // distribute the colors evenly on the hue range
    var r = []; // hold the generated colors
    for (var x=0; x<total; x++)
    {

        foo = hsvToRgb(i * x, 100, 100);
        r.push(rgbToHex(foo[0],foo[1],foo[2])); // you can also alternate the saturation and value for even more contrast between the colors
    }
    return r;
}

function openPipe(){
    // declare
    var sockjs_url = '/echo';
    var sockjs = new SockJS(sockjs_url);

    var multiplexer = new MultiplexedWebSocket(sockjs);
    var ann  = multiplexer.channel('ann');
    pipe(ann,  'first');
}

// random value generator (pick between -10 and 10)
function getNewData(channel_index){
    //channel_index = channel_index || 0;
    // add the offset specific to the channel to the random value
    // for example, if it's channel 3, we will add offset * (3+1)
    return Math.floor(Math.random() * 91) - 50; // + ($.offsetPerChannel * (channel_index+1));
}

function initCharts(){
    drawCharts();
    openPipe();
}

$(document).ready(function(){

    $.testmode = 'continuous';
    $.charts = [];
    $.dataLength = parseInt($('#dataLength').val());
    $.offsetPerChannel = 0;
    $.numChannels = 1;

    // bind numChannels dropdown
    $('#numChannelsSelect').change(function(){
        $.numChannels = parseInt($(this).val());
        initCharts();
    });


    // bind testmode
    $('input[type=checkbox][name=testmode]').on('change',function(){
        if($(this).is(':checked')){
            console.log('check')
        } else {
            console.log('not check')
        }
        $.testmode = $(this).val();

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

});