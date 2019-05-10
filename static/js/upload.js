$(function() {
    $('#upload-file-btn').click(function() {
        var form_data = new FormData($('#upload-file')[0]);
        
        $(".status").empty()
        $(".predict").empty()
        $(".upload-image").empty()

        // add <p> tag and status
        $(".status").append(`<p> Processing... </p>`);

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                console.log('Success!');
                console.log(data);


                // remove existing <p> tag
                $(".predict").empty()

                // change status to complete
                $(".status").empty()
                $(".status").append(`<p> Done! </p>`);

                var prediction = eval(JSON.stringify(data.prediction));
                var image = eval(JSON.stringify(data.filepath));

                console.log(prediction);
                console.log(image)

                // add <p> tag and prediction
                $(".predict").append(`<p> Our cloud prediction: ${prediction} </p>`);

                $(".upload-image").append(`<img class="img-fluid d-block mx-auto" src='${image}' ></img>`);

                
            },
        });
    });
});