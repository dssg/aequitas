$(document).ready(function () {
    $('#id1').click(function () {
        $('#div2').hide();
        $('#div1').show();
    });
    $('#id2').click(function () {
        $('#div1').hide();
        $('#div2').show();
    });

    if ($('#id1').prop("checked") == true) {
        $('#div2').hide();
        $('#div1').show();
    } else {
        $('#div1').hide();
        $('#div2').show();
    }
});
