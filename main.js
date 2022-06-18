const imgSource = document.getElementById("source")

document.getElementById("fileinput").addEventListener("change", (e) => {
    imgSource.onload = () => processImage(imgSource)
    setImageSource(imgSource, e.target.files[0])
}, false);

const log = x => { document.getElementById("log").textContent += x };

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

    const size = image.size()
    const aspectRatio = size.width / size.height * 2 // *2 because line height is twice big than character width
    const gridWidth = Number(document.getElementById("gridwidth").value)    
    const gridHeight = Math.round(gridWidth / aspectRatio)

    zones = adjustZones(zones, size, gridWidth, gridHeight)

    convertZonesToAscii(zones, gridWidth, gridHeight);

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

function adjustZones(zones, size, gridWidth, gridHeight){
    console.log(zones);

    for(let zone of zones) {
        let
            x = zone.corners[0].x / size.width * gridWidth,
            y = zone.corners[0].y / size.height * gridHeight,
            x2 = zone.corners[2].x / size.width * gridWidth,
            y2 = zone.corners[2].y / size.height * gridHeight;

        [x, x2] = [x, x2].sort((a, b) => a - b);
        [y, y2] = [y, y2].sort((a, b) => a - b);

        Object.assign(zone, {x, x2, y, y2});
    }

    // precision adjustment
    for(let a of zones) {
        ['x','x2','y','y2'].forEach(coord => {
            const deltaPrecision = coord.startsWith('x') ? gridWidth / 12 : gridHeight / 12
            let aligned = zones.filter(b => Math.abs(a[coord] - b[coord]) < deltaPrecision);
            if(aligned.length > 1){
                let mean = average(aligned.map(zone => zone[coord]))
                console.log("aligning", coord, mean, aligned);
                aligned.forEach(zone => { zone[coord] = mean })
            }
        })
    }

    // rounding to indexes

    const indexCols = new Set()
    const indexRows = new Set()
    for(let zone of zones){
        zone.x = Math.round(zone.x)
        indexCols.add(zone.x)

        zone.y = Math.round(zone.y)
        indexRows.add(zone.y)
    }

    console.log({ indexCols, indexRows })

    // forcing index separation between x-x2 y-y2
    for(let zone of zones){
        zone.x2 = Math.round(zone.x2)
        if(indexCols.has(zone.x2)) zone.x2 -= 1
        zone.y2 = Math.round(zone.y2)
        if(indexRows.has(zone.y2)) zone.y2 -= 1
    }

    return zones
}

const convertZonesToAscii = (zones, gridWidth, gridHeight) => {
    let grid = Array(gridHeight+1).fill(0).map(() => Array((gridWidth+1)).fill(" "))

    for(let zone of zones) {
        grid[zone.y][zone.x] = "+"
        grid[zone.y][zone.x2] = "+"
        grid[zone.y2][zone.x] = "+"
        grid[zone.y2][zone.x2] = "+"

        for(let i=zone.x+1; i<zone.x2; i++){
            grid[zone.y][i] = "-"
            grid[zone.y2][i] = "-"
        }

        for(let j=zone.y+1; j<zone.y2; j++){
            grid[j][zone.x] = "|"
            grid[j][zone.x2] = "|"
        }
    }    

    // remove empty rows
    grid = grid.filter(row => !row.every(char => char === " "))

    const output = document.getElementById("output")
    output.value = 'grid-kiss:\n'+grid.map(row => '  "' + row.join("") + '"').join("\n")+";"
}