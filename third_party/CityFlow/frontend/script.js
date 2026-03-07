/**
 * Draw Road Network
 */
id = Math.random().toString(36).substring(2, 15);

BACKGROUND_COLOR = 0xe8ebed;
LANE_COLOR = 0x586970;
LANE_BORDER_WIDTH = 1;
LANE_BORDER_COLOR = 0x82a8ba;
LANE_INNER_COLOR = 0xbed8e8;
LANE_DASH = 10;
LANE_GAP = 12;
TRAFFIC_LIGHT_WIDTH = 3;
MAX_TRAFFIC_LIGHT_NUM = 100000;
ROTATE = 90;

CAR_LENGTH = 5;
CAR_WIDTH = 2;
CAR_COLOR = 0xe8bed4;

CAR_COLORS = [0xf2bfd7, // pink
            0xb7ebe4,   // cyan
            0xdbebb7,   // blue
            0xf5ddb5, 
            0xd4b5f5];
CAR_COLORS_NUM = CAR_COLORS.length;

NUM_CAR_POOL = 150000;
DISTRICT_BORDER_COLOR = 0x34495e;
GATEWAY_NODE_COLOR = 0xf39c12;
GATEWAY_EDGE_COLOR = 0xe67e22;
DISTRICT_PALETTE = [
    0x1f77b4, 0xff7f0e, 0x2ca02c, 0xd62728, 0x9467bd, 0x8c564b,
    0xe377c2, 0x7f7f7f, 0xbcbd22, 0x17becf, 0x393b79, 0x637939
];

LIGHT_RED = 0xdb635e;
LIGHT_GREEN = 0x85ee00;

TURN_SIGNAL_COLOR = 0xFFFFFF;
TURN_SIGNAL_WIDTH   = 1;
TURN_SIGNAL_LENGTH  = 5;

var simulation, roadnet, steps;
var nodes = {};
var edges = {};
var logs;
var gettingLog = false;

let Application = PIXI.Application,
    Sprite = PIXI.Sprite,
    Graphics = PIXI.Graphics,
    Container = PIXI.Container,
    ParticleContainer = PIXI.particles.ParticleContainer,
    Texture = PIXI.Texture,
    Rectangle = PIXI.Rectangle
;

var controls = new function () {
    this.replaySpeedMax = 1;
    this.replaySpeedMin = 0.01;
    this.replaySpeed = 0.5;
    this.paused = false;
};

var trafficLightsG = {};

var app, viewport, renderer, simulatorContainer, carContainer, trafficLightContainer;
var overlayContainer;
var turnSignalContainer;
var carPool;

var cnt = 0;
var frameElapsed = 0;
var totalStep;

var nodeCarNum = document.getElementById("car-num");
var nodeProgressPercentage = document.getElementById("progress-percentage");
var nodeTotalStep = document.getElementById("total-step-num");
var nodeCurrentStep = document.getElementById("current-step-num");
var nodeSelectedEntity = document.getElementById("selected-entity");

var SPEED = 3, SCALE_SPEED = 1.01;
var LEFT = 37, UP = 38, RIGHT = 39, DOWN = 40;
var MINUS = 189, EQUAL = 187, P = 80;
var LEFT_BRACKET = 219, RIGHT_BRACKET = 221; 
var ONE = 49, TWO = 50;
var SPACE = 32;

var keyDown = new Set();

var turnSignalTextures = [];

let pauseButton = document.getElementById("pause");
let nodeCanvas = document.getElementById("simulator-canvas");
let replayControlDom = document.getElementById("replay-control");
let replaySpeedDom = document.getElementById("replay-speed");

let loading = false;
let infoDOM = document.getElementById("info");
let selectedDOM = document.getElementById("selected-entity");

function infoAppend(msg) {
    infoDOM.innerText += "- " + msg + "\n";
}

function infoReset() {
    infoDOM.innerText = "";
}

/**
 * Upload files
 */
let ready = false;

let roadnetData = [];
let replayData = [];
let chartData = [];
let districtDataRaw = [];
let districtMap = null;
let showDistrictOverlay = true;
let showGatewayOverlay = true;

function handleChooseFile(v, label_dom) {
    return function(evt) {
        let file = evt.target.files[0];
        label_dom.innerText = file.name;
    }
}

function uploadFile(v, file, callback) {
    let reader = new FileReader();
    reader.onloadstart = function () {
        infoAppend("Loading " + file.name);
    };
    reader.onerror = function() {
        infoAppend("Loading " + file.name + "failed");
    }
    reader.onload = function (e) {
        infoAppend(file.name + " loaded");
        v[0] = e.target.result;
        callback();
    };
    try {
        reader.readAsText(file);
    } catch (e) {
        infoAppend("Loading failed");
        console.error(e.message);
    }
}

let debugMode = false;
let chartLog;
let showChart = false;
let chartConainterDOM = document.getElementById("chart-container");
let showDistrictOverlayDom = document.getElementById("show-district-overlay");
let showGatewayOverlayDom = document.getElementById("show-gateway-overlay");

function parseDistrictMap() {
    if (!districtDataRaw[0]) {
        districtMap = null;
        return;
    }
    try {
        districtMap = JSON.parse(districtDataRaw[0]);
    } catch (e) {
        districtMap = null;
        infoAppend("Parsing district map file failed");
    }
}

function uploadOptionalDistrictThenChart(done) {
    if (DistrictFileDom.value) {
        uploadFile(districtDataRaw, DistrictFileDom.files[0], function() {
            parseDistrictMap();
            if (ChartFileDom.value) {
                showChart = true;
                uploadFile(chartData, ChartFileDom.files[0], done);
            } else {
                showChart = false;
                done();
            }
        });
    } else {
        districtMap = null;
        if (ChartFileDom.value) {
            showChart = true;
            uploadFile(chartData, ChartFileDom.files[0], done);
        } else {
            showChart = false;
            done();
        }
    }
}

function start() {
    if (loading) return;
    loading = true;
    infoReset();
    showDistrictOverlay = showDistrictOverlayDom.checked;
    showGatewayOverlay = showGatewayOverlayDom.checked;
    uploadFile(roadnetData, RoadnetFileDom.files[0], function(){
    uploadFile(replayData, ReplayFileDom.files[0], function(){
        let after_update = function() {
            infoAppend("drawing roadnet");
            ready = false;
            document.getElementById("guide").classList.add("d-none");
            hideCanvas();
            try {
                simulation = JSON.parse(roadnetData[0]);
            } catch (e) {
                infoAppend("Parsing roadnet file failed");
                loading = false;
                return;
            }
            try {
                logs = replayData[0].split('\n');
                logs.pop();
            } catch (e) {
                infoAppend("Reading replay file failed");
                loading = false;
                return;
            }

            totalStep = logs.length;
            if (showChart) {
                chartConainterDOM.classList.remove("d-none");
                let chart_lines = chartData[0].split('\n');
                if (chart_lines.length == 0) {
                    infoAppend("Chart file is empty");
                    showChart = false;
                }
                chartLog = [];
                for (let i = 0 ; i < totalStep ; ++i) {
                    step_data = chart_lines[i + 1].split(/[ \t]+/);
                    chartLog.push([]);
                    for (let j = 0; j < step_data.length; ++j) {
                        chartLog[i].push(parseFloat(step_data[j]));
                    }
                }
                chart.init(chart_lines[0], chartLog[0].length, totalStep);
            }else {
                chartConainterDOM.classList.add("d-none");
            }

            controls.paused = false;
            cnt = 0;
            debugMode = document.getElementById("debug-mode").checked;
            setTimeout(function () {
                try {
                    drawRoadnet();
                } catch (e) {
                    infoAppend("Drawing roadnet failed");
                    console.error(e.message);
                    loading = false;
                    return;
                }
                ready = true;
                loading = false;
                infoAppend("Start replaying");
            }, 200);
        };

        uploadOptionalDistrictThenChart(after_update);

    }); // replay callback
    }); // roadnet callback
}

let RoadnetFileDom = document.getElementById("roadnet-file");
let ReplayFileDom = document.getElementById("replay-file");
let ChartFileDom = document.getElementById("chart-file");
let DistrictFileDom = document.getElementById("district-file");

RoadnetFileDom.addEventListener("change",
    handleChooseFile(roadnetData, document.getElementById("roadnet-label")), false);
ReplayFileDom.addEventListener("change",
    handleChooseFile(replayData, document.getElementById("replay-label")), false);
ChartFileDom.addEventListener("change",
    handleChooseFile(chartData, document.getElementById("chart-label")), false);
DistrictFileDom.addEventListener("change",
    handleChooseFile(districtDataRaw, document.getElementById("district-label")), false);

showDistrictOverlayDom.addEventListener("change", function(evt) {
    showDistrictOverlay = evt.target.checked;
    if (ready) drawRoadnet();
});
showGatewayOverlayDom.addEventListener("change", function(evt) {
    showGatewayOverlay = evt.target.checked;
    if (ready) drawRoadnet();
});

document.getElementById("start-btn").addEventListener("click", start);

document.getElementById("slow-btn").addEventListener("click", function() {
    updateReplaySpeed(controls.replaySpeed - 0.1);
})

document.getElementById("fast-btn").addEventListener("click", function() {
    updateReplaySpeed(controls.replaySpeed + 0.1);
})

function updateReplaySpeed(speed){
    speed = Math.min(speed, 1);
    speed = Math.max(speed, 0);
    controls.replaySpeed = speed;
    replayControlDom.value = speed * 100;
    replaySpeedDom.innerHTML = speed.toFixed(2);
}

updateReplaySpeed(0.5);

replayControlDom.addEventListener('change', function(e){
    updateReplaySpeed(replayControlDom.value / 100);
});

document.addEventListener('keydown', function(e) {
    if (e.keyCode == P) {
        controls.paused = !controls.paused;
    } else if (e.keyCode == ONE) {
        updateReplaySpeed(Math.max(controls.replaySpeed / 1.5, controls.replaySpeedMin));
    } else if (e.keyCode == TWO ) {
        updateReplaySpeed(Math.min(controls.replaySpeed * 1.5, controls.replaySpeedMax));
    } else if (e.keyCode == LEFT_BRACKET) {
        cnt = (cnt - 1) % totalStep;
        cnt = (cnt + totalStep) % totalStep;
        drawStep(cnt);
    } else if (e.keyCode == RIGHT_BRACKET) {
        cnt = (cnt + 1) % totalStep;
        drawStep(cnt);
    } else {
        keyDown.add(e.keyCode)
    }
});

document.addEventListener('keyup', (e) => keyDown.delete(e.keyCode));

nodeCanvas.addEventListener('dblclick', function(e){
    controls.paused = !controls.paused;
});

pauseButton.addEventListener('click', function(e){
    controls.paused = !controls.paused;
});

function initCanvas() {
    app = new Application({
        width: nodeCanvas.offsetWidth,
        height: nodeCanvas.offsetHeight,
        transparent: false,
        backgroundColor: BACKGROUND_COLOR
    });

    nodeCanvas.appendChild(app.view);
    app.view.classList.add("d-none");

    renderer = app.renderer;
    renderer.interactive = true;
    renderer.autoResize = true;

    renderer.resize(nodeCanvas.offsetWidth, nodeCanvas.offsetHeight);
    app.ticker.add(run);
}

function showCanvas() {
    document.getElementById("spinner").classList.add("d-none");
    app.view.classList.remove("d-none");
}

function hideCanvas() {
    document.getElementById("spinner").classList.remove("d-none");
    app.view.classList.add("d-none");
}

function drawRoadnet() {
    if (simulatorContainer) {
        simulatorContainer.destroy(true);
    }
    app.stage.removeChildren();
    viewport = new Viewport.Viewport({
        screenWidth: window.innerWidth,
        screenHeight: window.innerHeight,
        interaction: app.renderer.plugins.interaction
    });
    viewport
        .drag()
        .pinch()
        .wheel()
        .decelerate();
    app.stage.addChild(viewport);
    simulatorContainer = new Container();
    viewport.addChild(simulatorContainer);

    roadnet = simulation.static;
    nodes = [];
    edges = [];
    trafficLightsG = {};

    for (let i = 0, len = roadnet.nodes.length;i < len;++i) {
        node = roadnet.nodes[i];
        node.point = new Point(transCoord(node.point));
        nodes[node.id] = node;
    }

    for (let i = 0, len = roadnet.edges.length;i < len;++i) {
        edge = roadnet.edges[i];
        edge.from = nodes[edge.from];
        edge.to = nodes[edge.to];
        for (let j = 0, len = edge.points.length;j < len;++j) {
            edge.points[j] = new Point(transCoord(edge.points[j]));
        }
        edges[edge.id] = edge;
    }

    /**
     * Draw Map
     */
    trafficLightContainer = new ParticleContainer(MAX_TRAFFIC_LIGHT_NUM, {tint: true});
    let mapContainer, mapGraphics;
    if (debugMode) {
        mapContainer = new Container();
        simulatorContainer.addChild(mapContainer);
    }else {
        mapGraphics = new Graphics();
        simulatorContainer.addChild(mapGraphics);
    }

    for (nodeId in nodes) {
        if (!nodes[nodeId].virtual) {
            let nodeGraphics;
            if (debugMode) {
                nodeGraphics = new Graphics();
                mapContainer.addChild(nodeGraphics);
            } else {
                nodeGraphics = mapGraphics;
            }
            drawNode(nodes[nodeId], nodeGraphics);
        }
    }
    for (edgeId in edges) {
        let edgeGraphics;
        if (debugMode) {
            edgeGraphics = new Graphics();
            mapContainer.addChild(edgeGraphics);
        } else {
            edgeGraphics = mapGraphics;
        }
        drawEdge(edges[edgeId], edgeGraphics);
    }
    drawOverlayLayers();
    let bounds = simulatorContainer.getBounds();
    simulatorContainer.pivot.set(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
    simulatorContainer.position.set(renderer.width / 2, renderer.height / 2);
    simulatorContainer.addChild(trafficLightContainer);

    /**
     * Settings for Cars
     */
    TURN_SIGNAL_LENGTH = CAR_LENGTH;
    TURN_SIGNAL_WIDTH  = CAR_WIDTH / 2;

    var carG = new Graphics();
    carG.lineStyle(0);
    carG.beginFill(0xFFFFFF, 0.8);
    carG.drawRect(0, 0, CAR_LENGTH, CAR_WIDTH);

    let carTexture = renderer.generateTexture(carG);

    let signalG = new Graphics();
    signalG.beginFill(TURN_SIGNAL_COLOR, 0.7).drawRect(0,0, TURN_SIGNAL_LENGTH, TURN_SIGNAL_WIDTH)
           .drawRect(0, 3 * CAR_WIDTH - TURN_SIGNAL_WIDTH, TURN_SIGNAL_LENGTH, TURN_SIGNAL_WIDTH).endFill();
    let turnSignalTexture = renderer.generateTexture(signalG);

    let signalLeft = new Texture(turnSignalTexture, new Rectangle(0, 0, TURN_SIGNAL_LENGTH, CAR_WIDTH));
    let signalStraight = new Texture(turnSignalTexture, new Rectangle(0, CAR_WIDTH, TURN_SIGNAL_LENGTH, CAR_WIDTH));
    let signalRight = new Texture(turnSignalTexture, new Rectangle(0, CAR_WIDTH * 2, TURN_SIGNAL_LENGTH, CAR_WIDTH));
    turnSignalTextures = [signalLeft, signalStraight, signalRight];


    carPool = [];
    if (debugMode)
        carContainer = new Container();
    else
        carContainer = new ParticleContainer(NUM_CAR_POOL, {rotation: true, tint: true});


    turnSignalContainer = new ParticleContainer(NUM_CAR_POOL, {rotation: true, tint: true});
    simulatorContainer.addChild(carContainer);
    simulatorContainer.addChild(turnSignalContainer);
    for (let i = 0, len = NUM_CAR_POOL;i < len;++i) {
        //var car = Sprite.fromImage("images/car.png")
        let car = new Sprite(carTexture);
        let signal = new Sprite(turnSignalTextures[1]);
        car.anchor.set(1, 0.5);

        if (debugMode) {
            car.interactive = true;
            car.on('mouseover', function () {
                selectedDOM.innerText = car.name;
                car.alpha = 0.8;
            });
            car.on('mouseout', function () {
                // selectedDOM.innerText = "";
                car.alpha = 1;
            });
        }
        signal.anchor.set(1, 0.5);
        carPool.push([car, signal]);
    }
    showCanvas();

    return true;
}

function appendText(id, text) {
    let p = document.createElement("span");
    p.innerText = text;
    document.getElementById("info").appendChild(p);
    document.getElementById("info").appendChild(document.createElement("br"));
}

var statsFile = "";
var withRange = false;
var nodeStats, nodeRange;

initCanvas();


function transCoord(point) {
    return [point[0], -point[1]];
}

function getDistrictColor(districtId) {
    return DISTRICT_PALETTE[stringHash(districtId) % DISTRICT_PALETTE.length];
}

function cross2D(o, a, b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function convexHull(points) {
    if (!points || points.length <= 2) return points ? points.slice() : [];
    let sorted = points
        .slice()
        .sort((p, q) => (p.x === q.x ? p.y - q.y : p.x - q.x));
    let lower = [];
    for (let i = 0; i < sorted.length; ++i) {
        while (lower.length >= 2 && cross2D(lower[lower.length - 2], lower[lower.length - 1], sorted[i]) <= 0) {
            lower.pop();
        }
        lower.push(sorted[i]);
    }
    let upper = [];
    for (let i = sorted.length - 1; i >= 0; --i) {
        while (upper.length >= 2 && cross2D(upper[upper.length - 2], upper[upper.length - 1], sorted[i]) <= 0) {
            upper.pop();
        }
        upper.push(sorted[i]);
    }
    lower.pop();
    upper.pop();
    return lower.concat(upper);
}

function drawDistrictRegionFills(graphics, districtToPoints) {
    for (let districtId in districtToPoints) {
        let pts = districtToPoints[districtId];
        if (!pts || pts.length === 0) continue;
        let color = getDistrictColor(districtId);
        if (pts.length >= 3) {
            let hull = convexHull(pts);
            if (hull.length >= 3) {
                graphics.lineStyle(1.2, color, 0.35);
                graphics.beginFill(color, 0.11);
                graphics.moveTo(hull[0].x, hull[0].y);
                for (let i = 1; i < hull.length; ++i) {
                    graphics.lineTo(hull[i].x, hull[i].y);
                }
                graphics.lineTo(hull[0].x, hull[0].y);
                graphics.endFill();
                continue;
            }
        }
        if (pts.length === 2) {
            graphics.lineStyle(18, color, 0.10);
            graphics.drawLine(pts[0], pts[1]);
            continue;
        }
        graphics.lineStyle(0);
        graphics.beginFill(color, 0.12);
        graphics.drawCircle(pts[0].x, pts[0].y, 18);
        graphics.endFill();
    }
}

function drawEdgePolyline(graphics, edge, width, color, alpha) {
    graphics.lineStyle(width, color, alpha);
    for (let i = 1; i < edge.points.length; ++i) {
        graphics.drawLine(edge.points[i - 1], edge.points[i]);
    }
}

function drawOverlayLayers() {
    if (overlayContainer) {
        overlayContainer.destroy(true);
        overlayContainer = null;
    }
    if (!showDistrictOverlay && !showGatewayOverlay) {
        return;
    }
    overlayContainer = new Container();
    simulatorContainer.addChild(overlayContainer);

    let overlayGraphics = new Graphics();
    overlayContainer.addChild(overlayGraphics);

    let intersectionToDistrict = {};
    if (districtMap && districtMap.intersection_to_district) {
        intersectionToDistrict = districtMap.intersection_to_district;
    }
    let gatewayNodeIds = new Set(
        districtMap && districtMap.gateway_intersections
            ? districtMap.gateway_intersections
            : []
    );
    let gatewayEdgeIds = new Set(
        districtMap && districtMap.gateway_roads
            ? districtMap.gateway_roads
            : []
    );
    let nodeDegree = {};
    for (let edgeId in edges) {
        let edge = edges[edgeId];
        nodeDegree[edge.from.id] = (nodeDegree[edge.from.id] || 0) + 1;
        nodeDegree[edge.to.id] = (nodeDegree[edge.to.id] || 0) + 1;
    }
    for (let nodeId in nodes) {
        if ((nodeDegree[nodeId] || 0) <= 1) {
            gatewayNodeIds.add(nodeId);
        }
    }

    if (showDistrictOverlay && districtMap && districtMap.intersection_to_district) {
        let districtToPoints = {};
        for (let nodeId in nodes) {
            let districtId = intersectionToDistrict[nodeId];
            if (!districtId) continue;
            if (!districtToPoints[districtId]) districtToPoints[districtId] = [];
            districtToPoints[districtId].push(nodes[nodeId].point);
        }
        drawDistrictRegionFills(overlayGraphics, districtToPoints);

        for (let edgeId in edges) {
            let edge = edges[edgeId];
            let fromDistrict = intersectionToDistrict[edge.from.id];
            let toDistrict = intersectionToDistrict[edge.to.id];
            if (!fromDistrict || !toDistrict) continue;
            if (fromDistrict === toDistrict) {
                drawEdgePolyline(
                    overlayGraphics,
                    edge,
                    1.2,
                    getDistrictColor(fromDistrict),
                    0.45
                );
            } else {
                drawEdgePolyline(
                    overlayGraphics,
                    edge,
                    1.8,
                    DISTRICT_BORDER_COLOR,
                    0.8
                );
            }
        }
        for (let nodeId in nodes) {
            let districtId = intersectionToDistrict[nodeId];
            if (!districtId) continue;
            let point = nodes[nodeId].point;
            overlayGraphics.beginFill(getDistrictColor(districtId), 0.55);
            overlayGraphics.drawCircle(point.x, point.y, 2.0);
            overlayGraphics.endFill();
        }
    }

    if (showGatewayOverlay) {
        for (let edgeId in edges) {
            let edge = edges[edgeId];
            let fromGateway = gatewayNodeIds.has(edge.from.id) || String(edge.from.id).indexOf("g_") === 0;
            let toGateway = gatewayNodeIds.has(edge.to.id) || String(edge.to.id).indexOf("g_") === 0;
            let edgeLooksGateway = edgeId.indexOf("_g_") >= 0 || edgeId.indexOf("r_g_") === 0;
            if (
                gatewayEdgeIds.has(edgeId)
                || fromGateway
                || toGateway
                || edgeLooksGateway
            ) {
                drawEdgePolyline(
                    overlayGraphics,
                    edge,
                    5.6,
                    GATEWAY_EDGE_COLOR,
                    0.82
                );
            }
        }
        for (let nodeId in nodes) {
            if (gatewayNodeIds.has(nodeId) || String(nodeId).indexOf("g_") === 0) {
                let point = nodes[nodeId].point;
                overlayGraphics.lineStyle(0);
                overlayGraphics.beginFill(GATEWAY_NODE_COLOR, 0.28);
                overlayGraphics.drawCircle(point.x, point.y, 34.0);
                overlayGraphics.endFill();
                overlayGraphics.lineStyle(2.2, 0x1f2d3d, 0.98);
                overlayGraphics.beginFill(GATEWAY_NODE_COLOR, 0.96);
                overlayGraphics.drawCircle(point.x, point.y, 13.0);
                overlayGraphics.endFill();
                overlayGraphics.lineStyle(0);
                overlayGraphics.beginFill(0xffffff, 0.92);
                overlayGraphics.drawCircle(point.x, point.y, 5.2);
                overlayGraphics.endFill();
            }
        }
    }
}

PIXI.Graphics.prototype.drawLine = function(pointA, pointB) {
    this.moveTo(pointA.x, pointA.y);
    this.lineTo(pointB.x, pointB.y);
}

PIXI.Graphics.prototype.drawDashLine = function(pointA, pointB, dash = 16, gap = 8) {
    let direct = pointA.directTo(pointB);
    let distance = pointA.distanceTo(pointB);

    let currentPoint = pointA;
    let currentDistance = 0;
    let length;
    let finish = false;
    while (true) {
        this.moveTo(currentPoint.x, currentPoint.y);
        if (currentDistance + dash >= distance) {
            length = distance - currentDistance;
            finish = true;
        } else {
            length = dash
        }
        currentPoint = currentPoint.moveAlong(direct, length);
        this.lineTo(currentPoint.x, currentPoint.y);
        if (finish) break;
        currentDistance += length;

        if (currentDistance + gap >= distance) {
            break;
        } else {
            currentPoint = currentPoint.moveAlong(direct, gap);
            currentDistance += gap;
        }
    }
};

function drawNode(node, graphics) {
    graphics.beginFill(LANE_COLOR);
    let outline = node.outline;
    for (let i = 0 ; i < outline.length ; i+=2) {
        outline[i+1] = -outline[i+1];
        if (i == 0)
            graphics.moveTo(outline[i], outline[i+1]);
        else
            graphics.lineTo(outline[i], outline[i+1]);
    }
    graphics.endFill();

    if (debugMode) {
        graphics.hitArea = new PIXI.Polygon(outline);
        graphics.interactive = true;
        graphics.on("mouseover", function () {
            selectedDOM.innerText = node.id;
            graphics.alpha = 0.5;
        });
        graphics.on("mouseout", function () {
            graphics.alpha = 1;
        });
    }

}

function drawEdge(edge, graphics) {
    let from = edge.from;
    let to = edge.to;
    let points = edge.points;

    let pointA, pointAOffset, pointB, pointBOffset;
    let prevPointBOffset = null;

    let roadWidth = 0;
    edge.laneWidths.forEach(function(l){
        roadWidth += l;
    }, 0);

    let coords = [], coords1 = [];

    for (let i = 1;i < points.length;++i) {
        if (i == 1){
            pointA = points[0].moveAlongDirectTo(points[1], from.virtual ? 0 : from.width);
            pointAOffset = points[0].directTo(points[1]).rotate(ROTATE);
        } else {
            pointA = points[i-1];
            pointAOffset = prevPointBOffset;
        }
        if (i == points.length - 1) {
            pointB = points[i].moveAlongDirectTo(points[i-1], to.virtual ? 0 : to.width);
            pointBOffset = points[i-1].directTo(points[i]).rotate(ROTATE);
        } else {
            pointB = points[i];
            pointBOffset = points[i-1].directTo(points[i+1]).rotate(ROTATE);
        }
        prevPointBOffset = pointBOffset;

        lightG = new Graphics();
        lightG.lineStyle(TRAFFIC_LIGHT_WIDTH, 0xFFFFFF);
        lightG.drawLine(new Point(0, 0), new Point(1, 0));
        lightTexture = renderer.generateTexture(lightG);

        // Draw Traffic Lights
        if (i == points.length-1 && !to.virtual) {
            edgeTrafficLights = [];
            prevOffset = offset = 0;
            for (lane = 0;lane < edge.nLane;++lane) {
                offset += edge.laneWidths[lane];
                var light = new Sprite(lightTexture);
                light.anchor.set(0, 0.5);
                light.scale.set(offset - prevOffset, 1);
                point_ = pointB.moveAlong(pointBOffset, prevOffset);
                light.position.set(point_.x, point_.y);
                light.rotation = pointBOffset.getAngleInRadians();
                edgeTrafficLights.push(light);
                prevOffset = offset;
                trafficLightContainer.addChild(light);
            }
            trafficLightsG[edge.id] = edgeTrafficLights;
        }

        // Draw Roads
        graphics.lineStyle(LANE_BORDER_WIDTH, LANE_BORDER_COLOR, 1);
        graphics.drawLine(pointA, pointB);

        pointA1 = pointA.moveAlong(pointAOffset, roadWidth);
        pointB1 = pointB.moveAlong(pointBOffset, roadWidth);

        graphics.lineStyle(0);
        graphics.beginFill(LANE_COLOR);

        coords = coords.concat([pointA.x, pointA.y, pointB.x, pointB.y]);
        coords1 = coords1.concat([pointA1.y, pointA1.x, pointB1.y, pointB1.x]);

        graphics.drawPolygon([pointA.x, pointA.y, pointB.x, pointB.y, pointB1.x, pointB1.y, pointA1.x, pointA1.y]);
        graphics.endFill();

        offset = 0;
        for (let lane = 0, len = edge.nLane-1;lane < len;++lane) {
            offset += edge.laneWidths[lane];
            graphics.lineStyle(LANE_BORDER_WIDTH, LANE_INNER_COLOR);
            graphics.drawDashLine(pointA.moveAlong(pointAOffset, offset), pointB.moveAlong(pointBOffset, offset), LANE_DASH, LANE_GAP);
        }

        offset += edge.laneWidths[edge.nLane-1];

        // graphics.lineStyle(LANE_BORDER_WIDTH, LANE_BORDER_COLOR);
        // graphics.drawLine(pointA.moveAlong(pointAOffset, offset), pointB.moveAlong(pointBOffset, offset));
    }

    if (debugMode) {
        coords = coords.concat(coords1.reverse());
        graphics.interactive = true;
        graphics.hitArea = new PIXI.Polygon(coords);
        graphics.on("mouseover", function () {
            graphics.alpha = 0.5;
            selectedDOM.innerText = edge.id;
        });

        graphics.on("mouseout", function () {
            graphics.alpha = 1;
        });
    }
}

function run(delta) {
    let redraw = false;

    if (ready && (!controls.paused || redraw)) {
        try {
            drawStep(cnt);
        }catch (e) {
            infoAppend("Error occurred when drawing");
            ready = false;
        }
        if (!controls.paused) {
            frameElapsed += 1;
            if (frameElapsed >= 1 / controls.replaySpeed ** 2) {
                cnt += 1;
                frameElapsed = 0;
                if (cnt == totalStep) cnt = 0;
            }
        }
    }
}

function _statusToColor(status) {
    switch (status) {
        case 'r':
            return LIGHT_RED;
        case 'g':
            return LIGHT_GREEN;
        default:
            return 0x808080;  
    }
}

function stringHash(str) {
    let hash = 0;
    let p = 127, p_pow = 1;
    let m = 1e9 + 9;
    for (let i = 0; i < str.length; i++) {
        hash = (hash + str.charCodeAt(i) * p_pow) % m;
        p_pow = (p_pow * p) % m;
    }
    return hash;
}

function drawStep(step) {
    if (showChart && (step > chart.ptr || step == 0)) {
        if (step == 0) {
            chart.clear();
        }
        chart.ptr = step;
        chart.addData(chartLog[step]);
    }

    let [carLogs, tlLogs] = logs[step].split(';');

    tlLogs = tlLogs.split(',');
    carLogs = carLogs.split(',');
    
    let tlLog, tlEdge, tlStatus;
    for (let i = 0, len = tlLogs.length;i < len;++i) {
        tlLog = tlLogs[i].split(' ');
        tlEdge = tlLog[0];
        tlStatus = tlLog.slice(1);
        for (let j = 0, len = tlStatus.length;j < len;++j) {
            trafficLightsG[tlEdge][j].tint = _statusToColor(tlStatus[j]);
            if (tlStatus[j] == 'i' ) {
                trafficLightsG[tlEdge][j].alpha = 0;
            }else{
                trafficLightsG[tlEdge][j].alpha = 1;
            }
        }
    }

    carContainer.removeChildren();
    turnSignalContainer.removeChildren();
    let carLog, position, length, width;
    for (let i = 0, len = carLogs.length - 1;i < len;++i) {
        carLog = carLogs[i].split(' ');
        position = transCoord([parseFloat(carLog[0]), parseFloat(carLog[1])]);
        length = parseFloat(carLog[5]);
        width = parseFloat(carLog[6]);
        carPool[i][0].position.set(position[0], position[1]);
        carPool[i][0].rotation = 2*Math.PI - parseFloat(carLog[2]);
        carPool[i][0].name = carLog[3];
        let carColorId = stringHash(carLog[3]) % CAR_COLORS_NUM;
        carPool[i][0].tint = CAR_COLORS[carColorId];
        carPool[i][0].width = length;
        carPool[i][0].height = width;
        carContainer.addChild(carPool[i][0]);

        let laneChange = parseInt(carLog[4]) + 1;
        carPool[i][1].position.set(position[0], position[1]);
        carPool[i][1].rotation = carPool[i][0].rotation;
        carPool[i][1].texture = turnSignalTextures[laneChange];
        carPool[i][1].width = length;
        carPool[i][1].height = width;
        turnSignalContainer.addChild(carPool[i][1]);
    }
    nodeCarNum.innerText = carLogs.length-1;
    nodeTotalStep.innerText = totalStep;
    nodeCurrentStep.innerText = cnt+1;
    nodeProgressPercentage.innerText = (cnt / totalStep * 100).toFixed(2) + "%";
    if (statsFile != "") {
        if (withRange) nodeRange.value = stats[step][1];
        nodeStats.innerText = stats[step][0].toFixed(2);
    }
}

/*
Chart
 */
let chart = {
    max_steps: 3600,
    data: {
        labels: [],
        series: [[]]
    },
    options: {
        showPoint: false,
        lineSmooth: false,
        axisX: {
            showGrid: false,
            showLabel: false
        }
    },
    init : function(title, series_cnt, max_step){
        document.getElementById("chart-title").innerText = title;
        this.max_steps = max_step;
        this.data.labels = new Array(this.max_steps);
        this.data.series = [];
        for (let i = 0 ; i < series_cnt ; ++i)
            this.data.series.push([]);
        this.chart = new Chartist.Line('#chart', this.data, this.options);
    },
    addData: function (value) {
        for (let i = 0 ; i < value.length; ++i) {
            this.data.series[i].push(value[i]);
            if (this.data.series[i].length > this.max_steps) {
                this.data.series[i].shift();
            }
        }
        this.chart.update();
    },
    clear: function() {
        for (let i = 0 ; i < this.data.series.length ; ++i)
            this.data.series[i] = [];
    },
    ptr: 0
};
