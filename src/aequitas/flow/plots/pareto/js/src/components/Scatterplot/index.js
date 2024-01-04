/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React, { useMemo, useState, useEffect } from "react";
import Select from "react-select";
import { Group } from "@visx/group";
import { scaleLinear } from "@visx/scale";
import { AxisLeft, AxisBottom } from "@visx/axis";
import { Grid } from "@visx/grid";
import { Zoom } from "@visx/zoom";
import { extent } from "d3-array";
import { find } from "lodash";

import ParetoLine from "./components/ParetoLine";
import Points from "./components/Points";
import RangeSlider from "./components/RangeSlider";
import labels from "~/enums/labels";
import { formatFairnessLabel, formatPerformanceLabel } from "~/utils/formatters";
import { useChartDimensions } from "~/utils/hooks";
import { useAppDispatch, useAppState, useTunerState } from "~/utils/context";
import { ACTIONS } from "~/utils/reducer";
import { isPareto } from "~/utils/models";
import chartSettings from "~/constants/scatterplot";
import { translation, rescaleAxis, adjustDomain, calculateDomain } from "~/utils/zoom";
import { roundDomain, domainSize } from "~/utils/axisDomain";

import "./Scatterplot.scss";

const getDropdownOptions = (metrics, formatter) =>
metrics.sort().map((metric) => {
    return { label: formatter(metric), value: metric };
});

function Scatterplot() {
    const { isParetoVisible } = useAppState();
    const { models, optimizedPerformanceMetric, optimizedFairnessMetric, fairnessMetrics, performanceMetrics } =
        useTunerState();

    const dispatch = useAppDispatch();

    const [wrapperDivRef, dimensions] = useChartDimensions(chartSettings.dimensions);
    const [performanceMetric, setPerformanceMetric] = useState(optimizedPerformanceMetric);
    const [fairnessMetric, setFairnessMetric] = useState(optimizedFairnessMetric);

    const [performanceDomain, setPerformanceDomain] =  useState(roundDomain(extent(models.map((model) => model[performanceMetric]))));
    const [fairnessDomain, setFairnessDomain] = useState(roundDomain(extent(models.map((model) => model[fairnessMetric]))));

    const xScale = useMemo(
        () =>
            scaleLinear({
                domain: performanceDomain,
                range: [0, dimensions.boundedWidth],
            }),
        [dimensions.boundedWidth, performanceDomain],
    );

    const yScale = useMemo(
        () =>
            scaleLinear({
                domain: fairnessDomain,
                range: [dimensions.boundedHeight, 0],
            }),
        [dimensions.boundedHeight, fairnessDomain],
    );

    let zoomedScaleX = xScale;
    let zoomedScaleY = yScale;

    const setPerformance = (option, zoom) => {
        zoom.setTransformMatrix(chartSettings.initialTransform);
        setPerformanceMetric(option);
        setPerformanceDomain(roundDomain(extent(models.map((model) => model[option]))));
        setFairnessDomain(roundDomain(extent(models.map((model) => model[fairnessMetric]))));
        zoom.reset();
    }

    const setFairness = (option, zoom) => {
        zoom.setTransformMatrix(chartSettings.initialTransform);
        setFairnessMetric(option);
        setFairnessDomain(roundDomain(extent(models.map((model) => model[option]))));
        setPerformanceDomain(roundDomain(extent(models.map((model) => model[performanceMetric]))));
        zoom.reset();
    }

    const handleDomainChange = (newDomainRange, oldDomain, otherAxisDomain, 
        setThisAxisDomain, setOtherAxisDomain, zoom) => {

        // thisAxisDomain corresponds the the domain of the current axis being updated
        // if the axis being updated is the X axis, thisAxisDomain corresponds to domainX and otherAxisDomain to domainY

        if (Array.isArray(newDomainRange)) {
            const newDomain = [Math.max(Math.min(newDomainRange[0], oldDomain[1] - chartSettings.minDistance), 0), 
                Math.min(Math.max(newDomainRange[1], oldDomain[0] + chartSettings.minDistance), 1)];
            
            zoom.setTransformMatrix(chartSettings.initialTransform);

            setThisAxisDomain(newDomain);
            setOtherAxisDomain(otherAxisDomain);
            
            zoom.reset();
        }
    };

    const constrain = (transformMatrix, prevTransformMatrix) => {
        var newTransformMatrix = transformMatrix,
         newDomainX = calculateDomain(xScale, transformMatrix.translateX, transformMatrix.scaleX),
         newDomainY = calculateDomain(yScale, transformMatrix.translateY, transformMatrix.scaleY);

        // X axis
        // pan
        if (transformMatrix.scaleX === prevTransformMatrix.scaleX) {
            const limitMinX = chartSettings.minDomain;
            const limitMaxX = chartSettings.maxDomain - (domainSize(performanceDomain) / transformMatrix.scaleX);
            
            newTransformMatrix.translateX = 
                translation(newDomainX, prevTransformMatrix.translateX, transformMatrix.translateX, 
                    transformMatrix.scaleX, limitMinX, limitMaxX, xScale, performanceDomain[0]);
        }
        
        // zoom
        else {
            newDomainX = adjustDomain(newDomainX, zoomedScaleX.domain(), transformMatrix.scaleX, prevTransformMatrix.scaleX);

            if (domainSize(newDomainX) >= chartSettings.minDistance) {
                const scaleX = domainSize(performanceDomain) / domainSize(newDomainX);
                const translateX = xScale(performanceDomain[0] + (performanceDomain[0] - newDomainX[0])*scaleX);

                newTransformMatrix.scaleX = scaleX;
                newTransformMatrix.translateX = translateX;
            }
            
            else {
                newTransformMatrix.scaleX = prevTransformMatrix.scaleX;
                newTransformMatrix.translateX = prevTransformMatrix.translateX;
            }
        }

        // Y axis
        // pan
        if (transformMatrix.scaleY === prevTransformMatrix.scaleY) {
            const limitMinY =  (domainSize(fairnessDomain) / transformMatrix.scaleY) + chartSettings.minDomain;
            const limitMaxY = chartSettings.maxDomain;

            newTransformMatrix.translateY = 
                translation(newDomainY, prevTransformMatrix.translateY, transformMatrix.translateY, 
                    transformMatrix.scaleY, limitMinY, limitMaxY, yScale, fairnessDomain[1]);
        }
        // zoom
        else {
            newDomainY = adjustDomain(newDomainY, zoomedScaleY.domain(), transformMatrix.scaleY, prevTransformMatrix.scaleY);

            if (domainSize(newDomainY) >= chartSettings.minDistance) {
                const scaleY = domainSize(fairnessDomain) / domainSize(newDomainY);
                const translateY = yScale(fairnessDomain[1] + (fairnessDomain[1] - newDomainY[1])*scaleY);

                newTransformMatrix.scaleY = scaleY;
                newTransformMatrix.translateY = translateY;
            }

            else {
                newTransformMatrix.scaleY = prevTransformMatrix.scaleY;
                newTransformMatrix.translateY = prevTransformMatrix.translateY;
            }
        }

        return newTransformMatrix;
    }

    useEffect(() => {
        if (fairnessMetric === optimizedFairnessMetric && performanceMetric === optimizedPerformanceMetric) {
            dispatch({ type: ACTIONS.ENABLE_PARETO });
        } else {
            dispatch({ type: ACTIONS.DISABLE_PARETO });
        }
    }, [fairnessMetric, performanceMetric, optimizedFairnessMetric, optimizedPerformanceMetric, dispatch]);

    const fairnessDropdownOptions = getDropdownOptions(fairnessMetrics, formatFairnessLabel);
    const performanceDropdownOptions = getDropdownOptions(performanceMetrics, formatPerformanceLabel);

    return (
        <div className="scatterplot">
            <Zoom
                width={dimensions.width}
                height={dimensions.height}
                transformMatrix={chartSettings.initialTransform}
                constrain={constrain}
            >

            {(zoom) => {
                zoomedScaleX = rescaleAxis(xScale, zoom.transformMatrix.translateX, zoom.transformMatrix.scaleX);
                zoomedScaleY = rescaleAxis(yScale, zoom.transformMatrix.translateY, zoom.transformMatrix.scaleY);

                return (
                    <div>
                        <div className="axis-title left">
                            <h4>{labels.FAIRNESS}</h4>
                            <Select
                                value={find(fairnessDropdownOptions, (option) => option.value === fairnessMetric)}
                                options={fairnessDropdownOptions}
                                onChange={(option) => setFairness(option.value, zoom)}
                                className="axis-select bold"
                                data-testid="dropdown-fairness"
                            />
                        </div>

                        <div className="scatterplot-slider-wrapper">
                            <div className="y-axis-slider">
                                <RangeSlider
                                    domainRange={zoomedScaleY.domain()}
                                    handleDomainChange={handleDomainChange}
                                    isHorizontal={false}
                                    dimensions={dimensions.boundedHeight}
                                    otherDomain={zoomedScaleX.domain()}
                                    setThisAxisDomain={setFairnessDomain}
                                    setOtherAxisDomain={setPerformanceDomain}
                                    zoom={zoom}
                                />
                            </div>

                            <div className="scatterplot-wrapper" ref={wrapperDivRef}>

                                <svg 
                                    width={dimensions.width + 5} 
                                    height={dimensions.height}
                                    style={{
                                        cursor: zoom.isDragging ? "grabbing" : "grab",
                                    }}
                                    ref={zoom.containerRef}> 

                                    <rect
                                        x={dimensions.marginLeft}
                                        y={dimensions.marginTop}
                                        width={dimensions.boundedWidth}
                                        height={dimensions.boundedHeight}
                                        fill="transparent"
                                        onTouchStart={zoom.dragStart}
                                        onTouchMove={zoom.dragMove}
                                        onTouchEnd={zoom.dragEnd}
                                        onMouseDown={zoom.dragStart}
                                        onMouseMove={zoom.dragMove}
                                        onMouseUp={zoom.dragEnd}
                                        onMouseLeave={() => {if (zoom.isDragging) zoom.dragEnd();}}
                                    />

                                    <clipPath id="clip-points">
                                        <rect 
                                            x="-5" 
                                            y="-10" 
                                            width={dimensions.boundedWidth+10} 
                                            height={dimensions.boundedHeight+15} 
                                        />
                                    </clipPath>

                                    <clipPath id="clip-pareto-line">
                                        <rect 
                                            x="0" 
                                            y="-10" 
                                            width={dimensions.boundedWidth} 
                                            height={dimensions.boundedHeight+10} 
                                        />
                                    </clipPath>
                                    
                                    <Group top={dimensions.marginTop} left={dimensions.marginLeft}>
                                        <AxisBottom
                                            scale={zoomedScaleX}
                                            top={dimensions.boundedHeight}
                                            axisClassName="axis"
                                            numTicks={chartSettings.numTicks}
                                            tickFormat={zoomedScaleX.tickFormat(chartSettings.numTicks, "%")}
                                        />
                                        <AxisLeft
                                            scale={zoomedScaleY}
                                            axisClassName="axis"
                                            numTicks={chartSettings.numTicks}
                                            tickFormat={zoomedScaleY.tickFormat(chartSettings.numTicks, "%")}
                                        />
                                        <Group>
                                        <Grid
                                            xScale={zoomedScaleX}
                                            yScale={zoomedScaleY}
                                            width={dimensions.boundedWidth}
                                            height={dimensions.boundedHeight}
                                            className="axis-gridlines"
                                            numTicksRows={chartSettings.numTicks}
                                            numTicksColumns={chartSettings.numTicks}
                                        />
                                        <Group>
                                            <Group clipPath="url(#clip-pareto-line)">
                                                {isParetoVisible ? (
                                                    <ParetoLine
                                                    paretoPoints={models.filter((model) => isPareto(model))}
                                                    xScale={zoomedScaleX}
                                                    yScale={zoomedScaleY}
                                                    />
                                                    ) : null}
                                            </Group>
                                            <Group clipPath="url(#clip-points)">
                                                <Points
                                                    xScale={zoomedScaleX}
                                                    yScale={zoomedScaleY}
                                                    fairnessMetric={fairnessMetric}
                                                    performanceMetric={performanceMetric}
                                                    width={dimensions.width}
                                                    />
                                            </Group>
                                        </Group>
                                        </Group>
                                    </Group>
                                </svg>
                            </div>
                        </div>

                        <div className="x-axis-slider">
                            <RangeSlider
                                domainRange={zoomedScaleX.domain()}
                                handleDomainChange={handleDomainChange}
                                isHorizontal={true}
                                dimensions={dimensions.boundedWidth}
                                otherDomain={zoomedScaleY.domain()}
                                setThisAxisDomain={setPerformanceDomain}
                                setOtherAxisDomain={setFairnessDomain}
                                zoom={zoom}
                            />
                        </div>

                        <div className="axis-title bottom">
                            <h4>{labels.PERFORMANCE}</h4>
                            <Select
                                value={find(performanceDropdownOptions, (option) => option.value === performanceMetric)}
                                options={performanceDropdownOptions}
                                onChange={(option) => setPerformance(option.value, zoom)}
                                className="axis-select bold"
                                menuPlacement="top"
                                />
                        </div>
                    </div>

            )}}
            </Zoom>
        </div>
    );
}

export default Scatterplot;
