function httpGetAsync(theUrl, callback) {
                var xmlHttp = new XMLHttpRequest();
                xmlHttp.onreadystatechange = function () {
                    if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
                        callback(JSON.parse(xmlHttp.responseText));
                }
                xmlHttp.open("GET", theUrl, true); 
                xmlHttp.send(null);
}
//-------------------------------------------------------------------------------------------------------------------------------------
            window.onload = function () {
                var dataHumd = [];
                var dataTemp = [];

                var Chart1 = new CanvasJS.Chart("line-chart-1", {
                    zoomEnabled: true,          
                    axisX: {
                        title: "chart updates every 2 secs" 
                    },
                    data: [{
                       
                        type: "line", 
                        xValueType: "dateTime", 
                        dataPoints: dataTemp 
                    }
                    ],
                });

                var Chart2 = new CanvasJS.Chart("line-chart-2", {
                    zoomEnabled: true, 
                    axisX: {
                        title: "chart updates every 2 secs" 
                    },
                    data: [
                        {
                            type: "line",
                            xValueType: "dateTime",
                            showInLegend: true,
                            name: "humd",
                            dataPoints: dataHumd
                        }
                    ],
                });

                var updateInterval = 2000; 
                var time = new Date();

                var yHumdVal = 0; 
                var yTempVal = 0; 
               
                var updateChart = function () {
                    httpGetAsync('/get', function (data) {
                        time.setTime(time.getTime() + updateInterval);
                        yTempVal = parseInt(data[0].temp);
                        yHumdVal = parseInt(data[0].humd);
                        dataTemp.push({ 
                            x: time.getTime(),
                            y: yTempVal
                        });
                        dataHumd.push({
                            x: time.getTime(),
                            y: yHumdVal
                        });
                        Chart1.render();
                        Chart2.render(); 
                    });
                };
      
                setInterval(function () {
                    updateChart()
                }, updateInterval);

            }

