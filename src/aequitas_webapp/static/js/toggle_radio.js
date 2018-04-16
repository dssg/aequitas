document.addEventListener("DOMContentLoaded", function(event) {
    const customGroupRadio = document.getElementById('id1');
    const majorityGroupRadio = document.getElementById('id2');
    const minimumGroupRadio = document.getElementById('id3');

    const dropDownElement = document.getElementById('div1');
    const regularElement = document.getElementById('div2');

    customGroupRadio.addEventListener('click', () => {
        dropDownElement.style.display = 'block';
        regularElement.style.display = 'none';
    });

    majorityGroupRadio.addEventListener('click', () => {
        dropDownElement.style.display = 'none';
        regularElement.style.display = 'block';
    });

    minimumGroupRadio.addEventListener('click', () => {
        dropDownElement.style.display = 'none';
        regularElement.style.display = 'block';
    });
});
