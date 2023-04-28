const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
let isDrawing = false;

canvas.addEventListener("mousedown", start);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stop);
canvas.addEventListener("mouseleave", stop);
canvas.width = window.innerWidth - 64;
canvas.height = window.innerHeight - 64;

context.lineWidth = 15;
setPathColor();

let paths = [];
let path = null;

document.getElementById("sequentialBtn").addEventListener("click", execSequential);
document.getElementById("clearBtn").addEventListener("click", clearCanvas);
document.getElementById("themeBtn").addEventListener("click", changeTheme);

window.addEventListener("resize", function () {
    clearCanvas();
    canvas.width = window.innerWidth - 64;
    canvas.height = window.innerHeight - 64;
});

function start(event) {
    isDrawing = true;
    path = {
        coordinates: [],
    };

    context.beginPath();
    context.moveTo(
        event.clientX - canvas.offsetLeft,
        event.clientY - canvas.offsetTop
    );
}

function draw(event) {
    if (!isDrawing) return;

    context.lineTo(
        event.clientX - canvas.offsetLeft,
        event.clientY - canvas.offsetTop
    );
    context.stroke();

    let x = event.clientX - canvas.offsetLeft;
    let y = event.clientY - canvas.offsetTop;
    if ((path != null) | undefined) path.coordinates.push({ x: x, y: y });
}

function stop() {
    isDrawing = false;
    paths.push(path);
}

function clearCanvas() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    paths = [];
    path = null;
    console.clear()
}

function changeTheme() {
    const body = document.querySelector("body");
    const icon = document.getElementById("themeIcon");

    icon.textContent = body.classList[0] + "_mode";

    body.classList.toggle("light");
    body.classList.toggle("dark");

    setPathColor();
    redrawPaths();
}

function setPathColor() {
    const body = document.querySelector("body");
    let pathColor = body.classList.contains("dark") ? "#ffffff" : "#000000";
    context.strokeStyle = pathColor;
}

function redrawPaths() {
    context.clearRect(0, 0, canvas.width, canvas.height);

    paths.forEach(function (path) {
        if ((path != null) | undefined) {
            context.beginPath();
            context.moveTo(path.coordinates[0].x, path.coordinates[0].y);
            for (let i = 1; i < path.coordinates.length; i++) {
                context.lineTo(path.coordinates[i].x, path.coordinates[i].y);
            }
            context.stroke();
        }
    });
}

function execSequential() {
    predecir();
}

async function predecir() {
    const sequentialModelUrl = "models/sequential/model.json";
    const convolutionalModelUrl = "models/convolutional/model.json";
    const sequential = await tf.loadLayersModel(sequentialModelUrl);
    const convolutional = await tf.loadLayersModel(convolutionalModelUrl);

    //Pasar canvas a version 28x28
    let smallcanvas = document.createElement("canvas");
    let ctx2 = smallcanvas.getContext('2d');

    smallcanvas.width = 28;
    smallcanvas.height = 28;

    resample_single(canvas, 28, 28, smallcanvas);



    let imgData = ctx2.getImageData(0, 0, 28, 28);
    let arr = []; //El arreglo completo
    let arr28 = []; //Al llegar a 28 posiciones se pone en 'arr' como un nuevo indice

    for (let p = 0, i = 0; p < imgData.data.length; p += 4) {
        let valor = imgData.data[p + 3] / 255;
        arr28.push([valor]); //Agregar al arr28 y normalizar a 0-1. Aparte queda dentro de un arreglo en el indice 0... again

        if (arr28.length == 28) {
            arr.push(arr28);
            arr28 = [];
        }

    }




    arr = [arr]; //Meter el arreglo en otro arreglo por que si no tio tensorflow se enoja >:(

    //Nah basicamente Debe estar en un arreglo nuevo en el indice 0, por ser un tensor4d en forma 1, 28, 28, 1
    let tensor4 = tf.tensor4d(arr);

    let prediction1 = sequential.predict(tensor4).dataSync();
    let prediction2 = convolutional.predict(tensor4).dataSync();

    let result1 = prediction1.indexOf(Math.max.apply(null, prediction1));
    let result2 = prediction2.indexOf(Math.max.apply(null, prediction2));


    console.log("Prediccion secuencial: ", result1);
    console.log("Prediccion convolucional: ", result2);

}

function resample_single(canvas, width, height, resize_canvas) {

    let width_source = canvas.width;
    let height_source = canvas.height;
    width = Math.round(width);
    height = Math.round(height);
    let ratio_w = width_source / width;
    let ratio_h = height_source / height;
    let ratio_w_half = Math.ceil(ratio_w / 2);
    let ratio_h_half = Math.ceil(ratio_h / 2);

    let ctx = canvas.getContext("2d");
    let ctx2 = resize_canvas.getContext("2d");
    let img = ctx.getImageData(0, 0, width_source, height_source);
    let img2 = ctx2.createImageData(width, height);
    let data = img.data;
    let data2 = img2.data;

    for (let j = 0; j < height; j++) {
        for (let i = 0; i < width; i++) {
            let x2 = (i + j * width) * 4;
            let weight = 0;
            let weights = 0;
            let weights_alpha = 0;
            let gx_r = 0;
            let gx_g = 0;
            let gx_b = 0;
            let gx_a = 0;
            let center_y = (j + 0.5) * ratio_h;
            let yy_start = Math.floor(j * ratio_h);
            let yy_stop = Math.ceil((j + 1) * ratio_h);

            for (let yy = yy_start; yy < yy_stop; yy++) {
                let dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
                let center_x = (i + 0.5) * ratio_w;
                let w0 = dy * dy; //pre-calc part of w
                let xx_start = Math.floor(i * ratio_w);
                let xx_stop = Math.ceil((i + 1) * ratio_w);

                for (let xx = xx_start; xx < xx_stop; xx++) {
                    let dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
                    let w = Math.sqrt(w0 + dx * dx);

                    if (w >= 1) {
                        //pixel too far
                        continue;
                    }

                    //hermite filter
                    weight = 2 * w * w * w - 3 * w * w + 1;
                    let pos_x = 4 * (xx + yy * width_source);
                    //alpha
                    gx_a += weight * data[pos_x + 3];
                    weights_alpha += weight;

                    //colors
                    if (data[pos_x + 3] < 255)
                        weight = weight * data[pos_x + 3] / 250;

                    gx_r += weight * data[pos_x];
                    gx_g += weight * data[pos_x + 1];
                    gx_b += weight * data[pos_x + 2];
                    weights += weight;

                }

            }

            data2[x2] = gx_r / weights;
            data2[x2 + 1] = gx_g / weights;
            data2[x2 + 2] = gx_b / weights;
            data2[x2 + 3] = gx_a / weights_alpha;
        }

    }

    //Ya que esta, exagerarlo. Blancos blancos y negros negros..?

    for (let p = 0; p < data2.length; p += 4) {
        let gris = data2[p]; //Esta en blanco y negro

        if (gris < 100) {
            gris = 0; //exagerarlo
        } else {
            gris = 255; //al infinito
        }

        data2[p] = gris;
        data2[p + 1] = gris;
        data2[p + 2] = gris;
    }

    ctx2.putImageData(img2, 0, 0);
}
