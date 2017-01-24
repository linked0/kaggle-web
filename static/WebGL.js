"use strict"

// <script src="{{url_for('static', filename='somejavascriptfile.js')}}"></script>

var gGL = null;

function initializeGL() {
    var canvas = document.getElementById("GLCanvas");
    gGL = canvas.getContext("webgl") ||
            canvas.getContext("experimental-webgl");
    if (gGL !== null) {
        gGL.clearColor(0.0, 0.8, 0.0, 1.0); // set the color to be cleared
    } else {
        document.write("<br><b>WebGL is not supported!</b>");
    }
}

function clearCanvas() {
    gGL.clear(gGL.COLOR_BUFFER_BIT); // clear to the color previously set
}

function doGLDraw() {
    initializeGL();
    clearCanvas();
}