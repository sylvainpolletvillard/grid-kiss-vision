import { createWorker } from 'tesseract.js';
import cv, { Mat, MatVector, Point, RotatedRect } from "opencv-ts";
import { average } from './helpers';
import { findContours, processImage } from './image_processing';

const imgSource = document.getElementById("source")! as HTMLImageElement
const fileInput = document!.getElementById("fileinput")! as HTMLInputElement
const logContainer = document.getElementById("log")!

const log = (msg: string) => { logContainer.textContent += msg };

imgSource.onload = () => startConverting(imgSource)
fileInput.addEventListener("change", () => {    
    if(fileInput.files){
        imgSource.src = URL.createObjectURL(fileInput.files[0])
    }
}, false);

interface ZoneBeforeAdjustment {
    angle: number;
    size: { width: number, height: number }
    corners: Point[];
    text?: string;
    contours: MatVector
}

interface Zone extends ZoneBeforeAdjustment {
    x: number;
    y: number;
    x2: number;
    y2: number;
    text?: string;
}


async function startConverting(imgElement: HTMLImageElement){
    const image = processImage(imgElement)
    cv.imshow('processed_image', image) 

    const zonesBeforeRotation = findZones(image) as ZoneBeforeAdjustment[];
    adjustRotation(image, zonesBeforeRotation);
    const zonesBeforeAdjustment = findZones(image) as Zone[]; // find zones again to update coords after rotation
    
    const size = image.size()
    const aspectRatio = size.width / size.height * 2 // *2 because line height is twice big than character width
    const gridWidthInput = document.getElementById("gridwidth")! as HTMLInputElement
    const gridWidth = Number(gridWidthInput.value)
    const gridHeight = Math.round(gridWidth / aspectRatio)

    log(zonesBeforeAdjustment.length + " zones detected");
    const zones: Zone[] = adjustZones(zonesBeforeAdjustment, size, gridWidth, gridHeight)
    await recognizeTexts(image, zones)
    
    drawZones(zones, image)

    convertZonesToAscii(zones, gridWidth, gridHeight);

    image.delete()
}

const adjustRotation = (mat: Mat, zones: ZoneBeforeAdjustment[]) => {
    const meanAngle = average(
        zones.map((zone: ZoneBeforeAdjustment) => (zone.angle + 90) % 90)
             .map((d: number) => d > 45 ? 90-d : d)
    )

    let dsize = new cv.Size(mat.cols, mat.rows);
    let center = new cv.Point(mat.cols / 2, mat.rows / 2);
    // You can try more different parameters
    let M = cv.getRotationMatrix2D(center, -meanAngle, 1);
    cv.warpAffine(mat, mat, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
    M.delete();
}

const findZones = (image: Mat) => {
    const contours = findContours(image);
    
    let zones: ZoneBeforeAdjustment[] = [];

    for(let i = 0; i < contours.size(); i++){
        const
            c = contours.get(i),
            peri = cv.arcLength(c, true),
            approx = new cv.Mat();

        cv.approxPolyDP(c, approx, 0.04 * peri, true);

        if (approx.rows === 4) { // detect rectangle
            let rect: RotatedRect = cv.minAreaRect(approx);
            if(rect.size.width < 10 || rect.size.height < 10) continue;

            const corners = cv.RotatedRect.points(rect)
            const zone: ZoneBeforeAdjustment = { ...rect, corners, contours }        
            zones.push(zone);  
        }
    }

    return zones
}

function drawZones(zones: Zone[], image: Mat){
    const mat = new cv.Mat.zeros(image.rows, image.cols, cv.CV_8UC3);
    zones.forEach((zone, i) => {
        cv.drawContours(mat, zone.contours, i, new cv.Scalar(0,255,0), -1, 8);
        const c = zone.contours.get(i);
        const M = cv.moments(c);
        const pos = new cv.Point(Math.round(M["m10"] / M["m00"]) - 20, Math.round(M["m01"] / M["m00"]));
        cv.putText(mat, zone.text ?? zone.text ?? "zone", pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, new cv.Scalar(255, 255, 255), 2)
    })
    cv.imshow('contours', mat);
    mat.delete();
}

function adjustZones(zones: Zone[], size: { width: number; height: number; }, gridWidth: number, gridHeight: number): Zone[]{
    console.log(zones);

    for(let zone of zones) {
        let
            x = zone.corners[1].x / size.width * gridWidth,
            y = zone.corners[1].y / size.height * gridHeight,
            x2 = zone.corners[3].x / size.width * gridWidth,
            y2 = zone.corners[3].y / size.height * gridHeight;

        [x, x2] = [x, x2].sort((a, b) => a - b);
        [y, y2] = [y, y2].sort((a, b) => a - b);

        Object.assign(zone, {x, x2, y, y2});
    }

    // precision adjustment
    for(let a of zones) {
        const coords: (keyof Zone)[] = ['x','x2','y','y2']
        coords.forEach((coord: keyof Zone) => {
            const deltaPrecision = coord.startsWith('x') ? gridWidth / 12 : gridHeight / 12
            let aligned = zones.filter(b => {
                const coordA = a[coord] as number
                const coordB = b[coord] as number
                return Math.abs(coordA - coordB) < deltaPrecision
            });
            if(aligned.length > 1){
                let mean = average(aligned.map((zone: { [x: string]: any; }) => zone[coord]))
                console.log("aligning", coord, mean, aligned);
                aligned.forEach(zone => { zone[coord] = mean as never })
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

async function recognizeTexts(image: Mat, zones: Zone[]){
    const worker = createWorker({
        logger: m => console.log(m)
      });
      
      await worker.load();
      await worker.loadLanguage('eng');
      await worker.initialize('eng');

    const zoneNames = document.getElementById("zone_names")! as HTMLDivElement
    zoneNames.innerHTML = ""

    for(let i=0; i<zones.length; i++){
        const zone = zones[i]
        console.log({ corners: zone.corners })

        const x1 = Math.min(...zone.corners.map(point => point.x))
        const x2 = Math.max(...zone.corners.map(point => point.x))
        const y1 = Math.min(...zone.corners.map(point => point.y))
        const y2 = Math.max(...zone.corners.map(point => point.y))

        const width = x2 - x1
        const height = y2 - y1
        const margin = Math.floor(0.1 * Math.min(width, height))
        let rect = new cv.Rect(x1+margin, y1+margin, width-margin*2, height-margin*2)
        const crop: Mat = image.roi(rect);
        const canvas = document.createElement("canvas") as HTMLCanvasElement
        const pname = document.createElement("p") as HTMLParagraphElement
        canvas.id = `zone_${i+1}_text`
        zoneNames.appendChild(canvas)
        cv.imshow(canvas.id, crop)        
        const { data: { text } } = await worker.recognize(canvas);
        zone.text = text.replaceAll(/\s/g, "")
        pname.textContent = zone.text
        zoneNames.appendChild(pname)
    }

    await worker.terminate();
}


const convertZonesToAscii = (zones: any[], gridWidth: number, gridHeight: number) => {
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

        let texty = zone.y + Math.floor((zone.y2-zone.y)/2)
        let textx = zone.x + Math.floor((zone.x2-zone.x)/2 - zone.text.length / 2)
        grid[texty].splice(textx, zone.text.length, ...zone.text)
    }    

    // remove empty rows
    grid = grid.filter(row => !row.every(char => char === " "))

    const output = document.getElementById("output")! as HTMLTextAreaElement
    output.value = 'grid-kiss:\n'+grid.map(row => '  "' + row.join("") + '"').join("\n")+";"
}