$(function () {

    /**
     * This plugin extends Highcharts in two ways:
     * - Use HTML5 canvas instead of SVG for rendering of the heatmap squares. Canvas
     *   outperforms SVG when it comes to thousands of single shapes.
     * - Add a K-D-tree to find the nearest point on mouse move. Since we no longer have SVG shapes
     *   to capture mouseovers, we need another way of detecting hover points for the tooltip.
     */
    (function (H) {
        var Series = H.Series,
            each = H.each;

        /**
         * Create a hidden canvas to draw the graph on. The contents is later copied over
         * to an SVG image element.
         */
        Series.prototype.getContext = function () {
            if (!this.canvas) {
                this.canvas = document.createElement('canvas');
                this.canvas.setAttribute('width', this.chart.chartWidth);
                this.canvas.setAttribute('height', this.chart.chartHeight);
                this.image = this.chart.renderer.image('', 0, 0, this.chart.chartWidth, this.chart.chartHeight).add(this.group);
                this.ctx = this.canvas.getContext('2d');
            }
            return this.ctx;
        };

        /**
         * Draw the canvas image inside an SVG image
         */
        Series.prototype.canvasToSVG = function () {
            this.image.attr({ href: this.canvas.toDataURL('image/png') });
        };

        /**
         * Wrap the drawPoints method to draw the points in canvas instead of the slower SVG,
         * that requires one shape each point.
         */
        H.wrap(H.seriesTypes.heatmap.prototype, 'drawPoints', function () {

            var ctx = this.getContext();

            if (ctx) {

                // draw the columns
                each(this.points, function (point) {
                    var plotY = point.plotY,
                        shapeArgs;

                    if (plotY !== undefined && !isNaN(plotY) && point.y !== null) {
                        shapeArgs = point.shapeArgs;

                        ctx.fillStyle = point.pointAttr[''].fill;
                        ctx.fillRect(shapeArgs.x, shapeArgs.y, shapeArgs.width, shapeArgs.height);
                    }
                });

                this.canvasToSVG();

            } else {
                this.chart.showLoading('Your browser doesn\'t support HTML5 canvas, <br>please use a modern browser');

                // Uncomment this to provide low-level (slow) support in oldIE. It will cause script errors on
                // charts with more than a few thousand points.
                // arguments[0].call(this);
            }
        });
        H.seriesTypes.heatmap.prototype.directTouch = false; // Use k-d-tree
    }(Highcharts));


});



////////////////////////////////////////////

var getRandomData = function (rows,columns,header_row) {
	// count is number of times loop runs to generate random dataPoints.
	//var data = "x,y,z\n";
    //for (var i = 1; i <= columns; i++) {
     //   for (var j = 1; j <= rows; j++) {
     //       data += i + "," + j + "," + getNewData() + "\n";
     //   }
	//};


    var data = [];
    if(header_row){
        data.push([null,null,null]);
    }
    for (var i = 1; i <= columns; i++) {
        for (var j = 1; j <= rows; j++) {
            data.push([i,j,getNewData()]);
        }
	};

//console.log(data)
    return data;
};


var randomUpdate = function () {
    for(var i=0;i< $.numChannels;i++) {
        //var pointsLength = $.charts[i].series[0].data.length;
        //for(p=0;p<pointsLength;p++){
        //    $.charts[i].series[0].data[p].update(getNewData());
        //}

        // setData (Array<Mixed> data, [Boolean redraw], [Mixed animation], [Boolean updatePoints])
        $.charts[i].series[0].setData(getRandomData($.chart_rows, $.chart_cols), true, false, true);

        //$.charts[i].userOptions.data.rows = getRandomData($.chart_rows, $.chart_cols);
        //$.charts[i].redraw();
    }
}
//var addCoords = function(chart_index, points){
//    var pointsLength = points.length;
//    var x = [];
//    var y = [];
//	for (var i = 0; i < pointsLength; i++) {
//		x.push(points[i][0]);
//        y.push(points[i][1]);
//	};
//    var plotDiv = document.getElementById('chartContainer'+chart_index);
//	// erase the old data
//    Plotly.deleteTraces(plotDiv, 0);
//    // draw the new data
//    plotDiv.data.push(getNewUpdateObject(x,y));
//	Plotly.redraw(plotDiv);
//}


var drawCharts = function(){

    // got these backwards
    $.chart_rows = 25;
    $.chart_cols = 30;

    // create the chart and save to global jquery scope
	for(var i=0;i< $.numChannels;i++) {
        var start;

        $('#chartContainer'+i).highcharts({

            data: {
                //csv: getRandomData($.chart_rows, $.chart_cols),
                rows: getRandomData($.chart_rows, $.chart_cols, true),
                parsed: function () {
                    start = +new Date();
                }
            },

            chart: {
                type: 'heatmap',
                margin: [60, 10, 80, 50]
            },

            title: {
                text: 'Highcharts extended heat map',
                align: 'left',
                x: 40
            },

            xAxis: {
                //type: 'datetime',
                //min: 1,
                //max: 5,
                //labels: {
                //    align: 'left',
                //    x: 5,
                //    y: 14,
                //    format: '{value:%B}' // long month
                //},
                //showLastLabel: false,
                //tickLength: 16
            },

            yAxis: {
                title: {
                    text: null
                },
                //labels: {
                //    format: '{value}:00'
                //},
                //minPadding: 0,
                //maxPadding: 0,
                //startOnTick: false,
                //endOnTick: false,
                //tickPositions: [0, 6, 12, 18, 24],
                //tickWidth: 1,
                //min: 1,
                //max: 5,
                //reversed: true
            },

            colorAxis: {
                stops: [
                    [0, '#3060cf'],
                    [0.5, '#fffbbc'],
                    [0.9, '#c4463a'],
                    [1, '#c4463a']
                ],
                //min: -15,
                //max: 25,
                startOnTick: false,
                endOnTick: false,
            },

            series: [{
                animation: false,
                borderWidth: 0,
                enableMouseTracking: false,
                //nullColor: '#EFEFEF',
                //colsize: 24 * 36e5, // one day
                states: {
                    hover: {enabled: false}
                },
                stickyTracking: false,
                tooltip: {
                    animation: false,
                    enabled: false
                },
                turboThreshold: Number.MAX_VALUE, // #3404, remove after 4.0.5 release
                //data: [getRandomData($.chart_rows, $.chart_cols)]
            }],

            credits: {enabled: false},
            tooltip: {enabled: false},


        });
        console.log('Rendered in ' + (new Date() - start) + ' ms'); // eslint-disable-line no-console

        $.charts[i] = $('#chartContainer'+i).highcharts();
        //console.log($.charts[i]);
    }


    //
     //   var layout = {
     //       title: 'Channel ' + (i+1),
     //       titlefont:{ size: 12 },
     //       margin: { l: 25,r: 5, b: 0, t: 20 },
     //       yaxis: {
     //         showgrid: false,
     //         zeroline: false,
     //         showticklabels: false,
     //         ticks: ''
     //       }
     //   };
    //
     //   var data = [
     //     {
     //       z: getRandomData($.chart_rows,$.chart_cols),
     //       colorscale: $.colorscale,
     //       type: 'heatmap',
     //       hoverinfo:'none',
     //       scrollZoom: false
     //     }
     //   ];
    //
	//	$.charts[i] = Plotly.newPlot('chartContainer'+i, data, layout, {displayModeBar: false, scrollZoom: false});
    //
	//}

}