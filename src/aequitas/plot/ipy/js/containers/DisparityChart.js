import React from "react";
import { toTitleCase } from "../utils/helpers";
import { scaleLinear, scalePow, scaleOrdinal } from "d3-scale";
import { max } from "d3-array";
import { Text } from "@vx/text";

import sizesEnum from "../enums/sizes";
import { CATEGORICAL_COLOR_PALETTE } from "../enums/colors";

import DisparityAxis from "../components/DisparityAxis";

export default function DisparityChart(props) {
  console.log(props.data);
  const chartAreaHeight = sizesEnum.ROW_HEIGHT * props.metrics.length;
  const fullHeight = chartAreaHeight;
  sizesEnum.MARGIN.top + sizesEnum.MARGIN.bottom;
  const maxAbsoluteDisparity = max(
    props.metrics.map((metric) =>
      max(props.data, (row) => Math.abs(row[`${metric}_disparity_scaled`]))
    )
  );

  const maxGroupSize = max(
    props.metrics.map((metric) => max(props.data, (row) => row["group_size"]))
  );

  const disparityScale = scaleLinear()
    .domain([-maxAbsoluteDisparity, maxAbsoluteDisparity])
    .range([sizesEnum.MARGIN.left, sizesEnum.WIDTH - sizesEnum.MARGIN.right])
    .nice();

  const bubbleSizeScale = scalePow()
    .exponent(0.5)
    .domain([0, maxGroupSize])
    .range([0, sizesEnum.ROW_HEIGHT / 4]);

  const groups = props.data.map((row) => row["attribute_value"]);
  console.log(groups);
  const bubbleColorScale = scaleOrdinal()
    .domain(groups)
    .range(CATEGORICAL_COLOR_PALETTE);

  return (
    <div className="aequitas-chart">
      <h1>Disparities on {toTitleCase(props.attribute)}</h1>
      <svg width={sizesEnum.WIDTH} height={fullHeight}>
        <DisparityAxis
          scale={disparityScale}
          chartAreaHeight={chartAreaHeight}
        />
        {props.metrics.map((metric, index) => {
          const metricAxisY = sizesEnum.ROW_HEIGHT * (index + 0.5);
          return (
            <g height={sizesEnum.ROW_HEIGHT} key={`row-${metric}`}>
              <Text x={0} y={metricAxisY} className="aequitas-row-names">
                {metric.toUpperCase()}
              </Text>
              <line
                x1={sizesEnum.MARGIN.left}
                y1={metricAxisY}
                x2={sizesEnum.WIDTH - sizesEnum.MARGIN.right}
                y2={metricAxisY}
                className="aequitas-row-line"
              />
              {props.data.map((group) => {
                return (
                  <g>
                    <circle
                      className="aequitas-bubble"
                      cx={disparityScale(group[`${metric}_disparity_scaled`])}
                      cy={metricAxisY}
                      r={bubbleSizeScale(group["group_size"])}
                      fill={bubbleColorScale(group["attribute_value"])}
                    />
                    <circle
                      cx={disparityScale(group[`${metric}_disparity_scaled`])}
                      cy={metricAxisY}
                      r={4}
                      fill={bubbleColorScale(group["attribute_value"])}
                    />
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
