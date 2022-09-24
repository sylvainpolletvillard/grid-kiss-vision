import cv, { Mat } from "opencv-ts";

export function processImage(imgElement: HTMLImageElement) {
    const image: Mat = cv.imread(imgElement)
    convertToGrayScale(image)
    applyBlur(image)
    resizeImage(image, 400)
    applyThreshold(image)
    return image
}

function resizeImage (mat: Mat, width: number) {
    const
        ratio = width / mat.cols,
        dsize = new cv.Size(Math.round(ratio * mat.cols), Math.round(ratio * mat.rows));

    cv.resize(mat, mat, dsize, 0, 0, cv.INTER_AREA);
}

function convertToGrayScale (mat: any) {
    cv.cvtColor(mat, mat, cv.COLOR_RGBA2GRAY, 0);
}

function applyBlur (mat: Mat) {
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

const applyThreshold = (mat: any) => {
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

export function findContours(mat: Mat) {
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