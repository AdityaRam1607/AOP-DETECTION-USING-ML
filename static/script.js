$(document).ready(function() {
    $('#aopForm').submit(function(e) {
        e.preventDefault();
        var form = $(this);
        var formData = {
            'input_features': [
                parseFloat($('#hole_dia').val()),
                parseInt($('#num_holes').val()),
                parseFloat($('#hole_depth').val()),
                parseFloat($('#burden_spacing').val()),
                parseFloat($('#deck').val()),
                parseFloat($('#top_stemming').val()),
                parseFloat($('#avg_explosive').val()),
                parseFloat($('#total_explosives_weight').val())
            ]
        };
        $.ajax({
            type: 'POST',
            url: '/predict',
            contentType: 'application/json',
            data: JSON.stringify(formData),
            success: function(response) {
                $('#predictionResult').html('Predicted AOP (dB): ' + response.predicted_aop);
            },
            error: function(xhr, status, error) {
                console.error(xhr.responseText);
            }
        });
    });
});
