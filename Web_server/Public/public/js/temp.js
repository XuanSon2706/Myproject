function httpGetAsync(theUrl, callback) {
                var xmlHttp = new XMLHttpRequest();
                xmlHttp.onreadystatechange = function () {
                    if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
                        callback(JSON.parse(xmlHttp.responseText));
                }
                xmlHttp.open("GET", theUrl, true); // true for asynchronous
                xmlHttp.send(null);
            }

            window.onload = function () {
                var dataTemp = [];
 		var Chart = new CanvasJS.Chart("line-chart-1", {
                    zoomEnabled: true, // Dùng thuộc tính có thể zoom vào graph          
                    axisX: {
                        title: "chart updates every 2 secs" // Chú thích cho trục X
                    },
                    data: [{
                       
                        type: "line", 
                        xValueType: "dateTime", 
                        dataPoints: dataTemp 
                    }
                    ],
                });
                
                var yTempVal = 0; // Biến lưu giá trị nhiệt độ (theo trục Y)
                var updateInterval = 2000; // Thời gian cập nhật dữ liệu 2000ms = 2s
                var time = new Date(); // Lấy thời gian hiện tại

                var updateChart = function () {
                    httpGetAsync('/get', function (data) {
                        time.setTime(time.getTime() + updateInterval);
                        yTempVal = parseInt(data[0].temp);
              
                        dataTemp.push({ // cập nhât dữ liệu mới từ server
                            x: time.getTime(),
                            y: yTempVal
                        });
                        
                        Chart.render(); // chuyển đổi dữ liệu của của graph thành mô hình đồ họa
                    });
                };
                updateChart(); // Chạy lần đầu tiên
                setInterval(function () { // Cập nhật lại giá trị graph sau thời gian updateInterval
                    updateChart()
                }, updateInterval);
            }