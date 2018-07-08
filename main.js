const imgSource = document.getElementById("source")
const output = document.getElementById("output")

document.getElementById("fileInput").addEventListener("change", (e) => {
    imgSource.onload = () => processImage(imgSource)
    setImageSource(imgSource, e.target.files[0])
}, false);

const log = x => { document.getElementById("log").textContent = x };

const setImageSource = (img, source) => { img.src = URL.createObjectURL(source) }

const processImage = (imgElement) => {
    let image = cv.imread(imgElement)
    convertToGrayScale(image)
    applyBlur(image)
    resizeImage(image, 400)
    applyThreshold(image)


    let zones = findZones(image);
    adjustRotation(image, zones);
    zones = findZones(image); // find zones again to update coords after rotation

    cv.imshow('processed_image', image)

    log(zones.length + " zones detected");

    convertZonesToAscii(zones, image.size());

    image.delete()
}

const resizeImage = (mat, width) => {
    const
        ratio = width / mat.cols,
        dsize = new cv.Size(Math.round(ratio * mat.cols), Math.round(ratio * mat.rows));

    cv.resize(mat, mat, dsize, 0, 0, cv.INTER_AREA);
}

const convertToGrayScale = mat => {
    cv.cvtColor(mat, mat, cv.COLOR_RGBA2GRAY, 0);
}

const applyBlur = mat => {
    const {width, height} = mat.size(),
        blurRatio = 1 / 250,
        [blurWidth, blurHeight] = [width, height].map(n => Math.round(n*blurRatio)).map(n => n % 2 === 0 ? n + 1 : n)

    /*
Parameters
    src	input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
    dst	output image of the same size and type as src.
    ksize	blurring kernel size.
    sigmaX	Gaussian kernel standard deviation in X direction.
    sigmaY	Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
    borderType	pixel extrapolation method(see cv.BorderTypes).
    */

    cv.GaussianBlur(mat, mat, new cv.Size(blurWidth, blurHeight), 0, 0, cv.BORDER_DEFAULT);
}

const applyThreshold = mat => {
    /*
Parameters
    src	source 8-bit single-channel image.
    dst	destination image of the same size and the same type as src.
    maxValue	non-zero value assigned to the pixels for which the condition is satisfied
    adaptiveMethod	adaptive thresholding algorithm to use.
    thresholdType	thresholding type that must be either cv.THRESH_BINARY or cv.THRESH_BINARY_INV.
    blockSize	size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
    C	constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
 */
    cv.adaptiveThreshold(mat, mat, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 3);
}

const findContours = mat => {
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    /*
    Parameters
image	source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as binary.
contours	detected contours.
hierarchy	containing information about the image topology. It has as many elements as the number of contours.
mode	contour retrieval mode(see cv.RetrievalModes).
method	contour approximation method(see cv.ContourApproximationModes).
offset	optional offset by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context.
     */
    cv.findContours(mat, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    return contours
}

const average = arr => arr.reduce( ( p, c ) => p + c, 0 ) / arr.length;

const adjustRotation = (mat, zones) => {
    const meanAngle = average(zones.map(zone => (zone.angle + 90) % 90).map(d => d > 45 ? 90-d : d))

    let dsize = new cv.Size(mat.cols, mat.rows);
    let center = new cv.Point(mat.cols / 2, mat.rows / 2);
    // You can try more different parameters
    let M = cv.getRotationMatrix2D(center, -meanAngle, 1);
    cv.warpAffine(mat, mat, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
    M.delete();
}

const findZones = (image) => {
    let contours = findContours(image);
    let mat = cv.Mat.zeros(image.rows, image.cols, cv.CV_8UC3);
    let zones = [];

    for(let i = 0; i < contours.size(); i++){
        const
            c = contours.get(i),
            peri = cv.arcLength(c, true),
            approx = new cv.Mat();

        cv.approxPolyDP(c, approx, 0.04 * peri, true);

        if (approx.rows === 4) { // detect rectangle
            let zone = cv.minAreaRect(approx);
            if(zone.size.width < 10 || zone.size.height < 10) continue;

            zone.corners = cv.RotatedRect.points(zone)
            zones.push(zone);

            cv.drawContours(mat, contours, i, new cv.Scalar(0,255,0), -1, 8);

            const
                M = cv.moments(c),
                pos = new cv.Point(Math.round(M["m10"] / M["m00"]) - 20, Math.round(M["m01"] / M["m00"]));
            cv.putText(mat, "zone", pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, new cv.Scalar(255, 255, 255), 2)
        }
    }

    cv.imshow('contours', mat);
    mat.delete();

    return zones
}

const convertZonesToAscii = (zones, total) => {
    console.log(zones);

    const gridSize = 12;

    for(let zone of zones) {
        let
            x = zone.corners[0].x / total.width * gridSize,
            y = zone.corners[0].y / total.height * gridSize,
            x2 = zone.corners[2].x / total.width * gridSize,
            y2 = zone.corners[2].y / total.height * gridSize;

        [x, x2] = [x, x2].sort((a, b) => a - b);
        [y, y2] = [y, y2].sort((a, b) => a - b);

        Object.assign(zone, {x, x2, y, y2});
    }

    // precision adjustment
    for(let a of zones) {
        ['x','x2','y','y2'].forEach(coord => {
            let aligned = zones.filter(b => Math.abs(a[coord] - b[coord]) < 1.5);
            if(aligned.length > 1){
                let mean = average(aligned.map(z => z[coord]))
                console.log("aligning",coord, mean, aligned);
                aligned.forEach(z => { z[coord] = mean })
            }
        })
    }

    const aspectRatio = 3; // x / y
    const grid = Array(gridSize+1).fill(0).map(() => Array((gridSize+1)*aspectRatio).fill(" "))

    for(let zone of zones) {
        let
            x = Math.round(zone.x) * aspectRatio,
            y = Math.round(zone.y),
            x2 = Math.round(zone.x2) * aspectRatio,
            y2 = Math.round(zone.y2);

        grid[y][x] = "+"
        grid[y][x2] = "+"
        grid[y2][x] = "+"
        grid[y2][x2] = "+"

        for(let i=x+1; i<x2; i++){
            grid[y][i] = "-"
            grid[y2][i] = "-"
        }

        for(let j=y+1; j<y2; j++){
            grid[j][x] = "|"
            grid[j][x2] = "|"
        }
    }

    output.value = 'grid-kiss:\n'+grid.map(row => '  "' + row.join("") + '"').join("\n")+";"
}